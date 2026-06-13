# 最小可用 Agent — 系统设计 Spec

- **日期**: 2026-06-13
- **项目**: agent-framework
- **状态**: Draft（待用户 review）
- **对应笔试题**: 从零实现一个最小可用的 Agent

---

## 0. 目标与非目标

### 目标
1. 自实现 Agent 核心 runtime（不依赖 LangChain / OpenHands 等现成 Agent 框架）。
2. 支持多轮对话 + session 维护，且**跨轮次继续执行**（基于已有状态，非每轮全新问题）。
3. 提供基本循环：接收输入 → 判断直接答/调工具 → 执行工具 → 读结果 → 继续，直到最终答案。
4. 至少 3 个工具（calculator / search / todo）。
5. 最大步数限制 + 基本异常处理 + 工具调用 trace。
6. 使用真实 LLM API（DeepSeek，OpenAI 兼容协议）。

### 非目标（README 标注为"可扩展"，本期不做）
- 多 Agent 协作 / 生产级并发 / 鉴权 / 前端工程化框架（React/Vue）。
- ToT、LLMCompiler 等推理搜索类架构（见 §1.4 设计空间，不实现）。

### 关于 "minimal" 与 "分层 runtime" 的张力
笔试题要求 "最小可用"，本设计采用**分层 runtime**（Router + Planner + Executor + Reflexion + FSM 五族叠加），代码量高于纯最小实现。这是用户在 brainstorming 阶段的明确选择（选项："全烙实现，分层 runtime"）。分层的好处：每一族职责单一、可独立测试、README 能讲清架构选型；代价是代码量约为纯最小实现的 2–3 倍。

---

## 1. 架构

### 1.1 分层总览

```
┌─────────────────────────────────────────────────────────┐
│  FastAPI Server (HTTP + 静态 HTML)                       │
│    POST /chat   GET /sessions   GET /trace/{sid}        │
└────────────────────────┬────────────────────────────────┘
                         │
┌────────────────────────▼────────────────────────────────┐
│  Agent Orchestrator (runtime/agent.py)                   │
│  ───────────────────────────────────────────────────    │
│   FSM(F) 驱动状态转移，串联下面四层                      	 │
│                                                          │
│   ┌─────────┐   ┌──────────┐   ┌──────────┐   ┌───────┐	│
│   │ Router  │──▶│ Planner  │──▶│ Executor │──▶│Reply │ │
│   │  (I)    │   │   (D)    │   │   (C+E)  │   │      	│ │
│   └─────────┘   └──────────┘   └──────────┘   └───────┘	│
│        │              │              │                   │
│        │         生成 plan      function-calling loop    │
│        │        + Replanner     + Reflexion 自纠 (E)      │
│        │          (D′) 重规划   + ReWOO (C′) 并行簇       │
└────────┬──────────────┬──────────────┬──────────────────┘
         │              │              │
┌────────▼──────────────▼──────────────▼──────────────────┐
│  SessionStore (session/)   ToolRegistry (tools/)         │
│  TraceLogger (trace/)      LLMClient (llm/)              │
└─────────────────────────────────────────────────────────┘
```

### 1.2 五族职责映射

| 族 | 模块 | 职责 | 触发时机 |
|----|------|------|---------|
| **I — Router** | `runtime/router.py` | 第一道判断：直接答 / 简单工具 / 需规划 | 每轮输入后最先执行。实现：一次轻量 LLM 调用（不带 tools，约束输出为 enum 三选一）或基于关键词的兜底规则，二者可叠加（LLM 主 + 规则保底） |
| **D — Planner** | `runtime/planner.py` | 复杂任务生成有序步骤列表（Plan-and-Execute） | Router 判 `plan_required` |
| **C — Executor** | `runtime/executor.py` | Function-calling 循环：LLM ↔ 工具分发 ↔ 结果回填 | 每步执行；简单工具路径直入 |
| **E — Reflexion** | `runtime/reflexion.py` | 工具失败/低置信时自评，产出 lesson 存 memory，带教训重试（微观自纠，同一步重试） | Executor 内捕获异常时 |
| **D′ — Replanner** | `runtime/replanner.py` | 宏观自纠：基于已执行步骤结果修订剩余 plan（族 D 变体，区别于 Reflexion 的微观重试） | EXECUTING 触 `replan_needed`：Reflexion 重试耗尽 / step 结果推翻 plan 假设 / `MAX_REPLANS` 未超 |
| **C′ — ReWOO** | `runtime/rewoo.py` | 微观并行：独立步骤簇打成 DAG（带变量占位符 `E1/E2`），planner 一次推理，worker 绑变量执行，solver 一次合成（族 C 子模式，嵌在 PAE 内） | Planner 检测到独立可批步骤簇 |
| **F — FSM** | `runtime/fsm.py` | session 级状态机，驱动阶段转移 + memory 召回时机 | 贯穿整轮 |

> 备注：**function calling 是 DeepSeek（OpenAI 兼容）API 的接口协议，不是 Agent 框架**。loop、工具分发、session、memory、trace、状态机全部自实现，满足"核心 runtime 自己实现"。

### 1.3 状态机（F）定义

Session 级主状态：

```
IDLE ──input──▶ ROUTING ──┬─▶ RESPONDING(direct) ─▶ IDLE
                          ├─▶ EXECUTING(simple) ──▶ RESPONDING ─▶ IDLE
                          └─▶ PLANNING ─▶ EXECUTING ─▶ RESPONDING ─▶ IDLE
                                              │
                                  (tool fail) └─▶ REFLECTING ─▶ EXECUTING
                                              │
                                  (replan)    └─▶ REPLANNING ─┬─▶ EXECUTING
                                                              └─▶ RESPONDING(abort)
```

合法转移（非法转移抛 `InvalidTransition`）：

| from | event | to |
|------|-------|-----|
| IDLE | input | ROUTING |
| ROUTING | route=direct | RESPONDING |
| ROUTING | route=simple | EXECUTING |
| ROUTING | route=plan | PLANNING |
| PLANNING | plan_ready | EXECUTING |
| EXECUTING | tool_error | REFLECTING |
| REFLECTING | retry | EXECUTING |
| EXECUTING | replan_needed | REPLANNING |
| REFLECTING | retry_exhausted | REPLANNING |
| REPLANNING | plan_updated | EXECUTING |
| REPLANNING | abort | RESPONDING |
| EXECUTING | done | RESPONDING |
| RESPONDING | replied | IDLE |

> **REFLECTING vs REPLANNING 区分**：REFLECTING = 微观自纠，同一步学 lesson 重试（族 E）；REPLANNING = 宏观自纠，修订剩余 plan 步骤（族 D′）。链路：`tool_error → REFLECTING → 重试 → retry_exhausted → REPLANNING → 改 plan → EXECUTING 新步`。`MAX_REPLANS`（默认 2）防爆循环。

Todo 子状态（memory 内）：`NONE → PLANNED → IN_PROGRESS → DONE`，由 `todo` 工具驱动。

### 1.4 设计空间（已评估，部分纳入）

- **ToT / LLMCompiler**：推理搜索类，agent 场景杀鸡牛刀。不实现。
- **ReAct 文本解析**：function-calling 已覆盖等价能力且更稳，故不采用 prompt 解析路径。
- **ReWOO**：**已纳入**（族 C′，Phase B）。微观并行子模式：Planner 检测独立步骤簇 → 打成 DAG（`E1=...`, `E2=...(依赖E1)`）→ worker 绑变量执行（不走 LLM）→ solver 一次合成。收益是规划/执行解耦省 N 次 LLM round-trip，非纯并行（function-calling 本就支持单步并行 tool_call）。Replanner（族 D′）为其兜底：solver 判证据不足 → 升级 REPLANNING 重规划。
- **Multi-Agent / CodeAct**：超出 minimal，不做。

---

## 2. 模块清单与职责

```
agent_framework/
├── main.py                 # FastAPI app, 路由, 挂载 static/
├── config.py               # DEEPSEEK_API_KEY, BASE_URL, MODEL, MAX_STEPS, SESSION_DIR
├── llm/
│   └── client.py           # DeepSeek 封装 (openai SDK + base_url), chat_with_tools()
├── tools/
│   ├── base.py             # Tool 协议 + ToolRegistry (注册/dispatch/schema 导出)
│   ├── calculator.py       # 安全表达式求值 (ast, 白名单运算符)
│   ├── search.py           # mock 搜索 (预设语料)
│   └── todo.py             # 跨轮次核心: create/list/update, 写 session.memory
├── runtime/
│   ├── router.py           # Router.classify(input, memory) → Route(Enum)
│   ├── planner.py          # Planner.make_plan(input, memory) → List[Step] (族 D，主路径)
│   ├── replanner.py        # Replanner.revise(plan, results, memory) → List[Step] (族 D′，宏观自纠)
│   ├── rewoo.py            # ReWOO: planner DAG + worker 绑变量 + solver 合成 (族 C′，微观并行)
│   ├── executor.py         # Executor.run(session, input) → Outcome (function-calling loop，返回 needs_replan 标志)
│   ├── reflexion.py        # Reflexion.reflect(call, error, memory) → lesson(str) (族 E，微观自纠)
│   ├── fsm.py              # SessionFSM: 状态 + 转移 + 非法转移检测
│   └── agent.py            # Agent 类: 编排五族 + D′/C′，暴露 async chat(session_id, input)
├── session/
│   ├── models.py           # Session, Memory, TodoItem, Message dataclass
│   └── store.py            # JSON 持久化 (load/save/list), utf-8
├── trace/
│   └── logger.py           # 每步落 trace/{sid}.jsonl: step/llm_call/tool_call/result
├── static/
│   └── index.html          # 极简聊天 UI (vanilla JS fetch)
├── tests/
│   ├── test_tools.py
│   ├── test_fsm.py
│   ├── test_executor.py    # mock LLMClient 跑 loop
│   └── test_session_cross_turn.py  # 跨轮次场景
├── README.md
└── PROMPTS.md              # AI Prompt 与问题解决记录
```

**单一职责边界**（每个模块应能独立理解 + 独立测试）：
- `router` 只决定走哪条路径，不执行。
- `planner` 只产出步骤列表，不执行。
- `replanner` 只修订剩余 plan，不执行步骤；不产出 lesson（那是 reflexion）。
- `rewoo` 只在独立步骤簇内做 DAG 规划/执行/合成，不决定是否进入簇（由 planner 检测）。
- `executor` 只跑 function-calling loop，不决定路由；返回 `needs_replan` 标志，不自行重规划。
- `reflexion` 只产出 lesson，不重试（重试由 executor 控制）。
- `fsm` 只管状态合法性，不业务逻辑。
- `store` 只读写 JSON，不解析语义。

---

## 3. 核心循环（笔试要求的最小 loop，落在 Executor + Agent 编排）

### 3.1 Agent 顶层编排（runtime/agent.py）

```python
async def chat(self, session_id: str, user_input: str) -> str:
    session = self.store.load(session_id)          # 1. 召回 (load)
    session.fsm.transition(ROUTING)
    trace = self.trace.open(session_id)

    route = self.router.classify(user_input, session.memory)

    if route == Route.DIRECT:
        session.fsm.transition(RESPONDING)
        answer = await self.llm.respond(session.messages, user_input)

    elif route == Route.SIMPLE_TOOL:
        session.fsm.transition(EXECUTING)
        answer = await self.executor.run(session, user_input, trace)
        session.fsm.transition(RESPONDING)

    else:  # PLAN_REQUIRED
        session.fsm.transition(PLANNING)
        plan = await self.planner.make_plan(user_input, session.memory)
        # planner 可标记独立步骤簇为 is_rewoo_cluster，交 ReWOO（族 C′）处理
        session.memory.plan = plan
        session.fsm.transition(EXECUTING)

        results, replans, i = {}, 0, 0
        while i < len(plan):
            step = plan[i]
            if getattr(step, "is_rewoo_cluster", False):
                outcome = await self.rewoo.run(session, step, trace)   # DAG: plan→worker→solver
            else:
                outcome = await self.executor.run(session, step, trace)  # 内含 reflexion (族 E)
            results[i] = outcome

            if outcome.needs_replan and replans < self.config.MAX_REPLANS:
                session.fsm.transition(REPLANNING)
                plan = plan[:i] + await self.replanner.revise(plan, results, session.memory)
                replans += 1
                session.memory.plan = plan
                trace.log_replan(replans)
                session.fsm.transition(EXECUTING)
                continue                                            # 重跑修订后当前步
            i += 1

        answer = await self.llm.synthesize(session.memory.plan, results)

    session.messages.append(Message(role="assistant", content=answer))
    session.fsm.transition(IDLE)
    self.store.save(session)                        # 2. 持久化 (save)
    trace.close()
    return answer
```

### 3.2 Executor function-calling loop（runtime/executor.py）

```python
async def run(self, session, user_input, trace) -> Outcome:
    messages = self._build_messages(session, user_input)   # 注入 memory (见 §5)
    for step in range(self.config.MAX_STEPS):              # 最大步数限制
        trace.log_step(step)
        resp = await self.llm.chat_with_tools(messages, self.tools.schemas())

        if not resp.tool_calls:                            # 无 tool_call → 最终答案
            return Outcome(text=resp.text, needs_replan=False)

        for call in resp.tool_calls:
            trace.log_tool_call(step, call.name, call.args)
            try:
                result = await self.tools.dispatch(call, session)   # todo 写 memory
            except Exception as e:
                result = f"ERROR: {e}"
                lesson = await self.reflexion.reflect(call, e, session.memory)
                session.memory.lessons.append(lesson)
                trace.log_reflexion(step, lesson)
                if lesson.reflexion_exhausted:                      # 重试耗尽 → 触发 replan
                    return Outcome(text=result, needs_replan=True)
            trace.log_tool_result(step, result)
            messages.append(tool_result_message(call, result))

    trace.log_truncated()                                  # max_steps 触顶
    return Outcome(text="（已达最大步数，强制结束）", needs_replan=True)
```

> `Outcome = (text: str, needs_replan: bool)`。`needs_replan=True` 时 Agent.chat 走 REPLANNING 分支（族 D′）。`needs_replan` 也可由「step 结果推翻 plan 假设」触发（如 search 返空且无备选步）。

### 3.3 满足笔试基本循环映射

| 笔试要求 | 实现位置 |
|---------|---------|
| 接收用户输入 | `Agent.chat(session_id, input)` |
| 判断直接答/调工具 | `Router.classify` (族 I) |
| 执行工具 | `ToolRegistry.dispatch` (族 C) |
| 读取工具结果 | Executor loop 回填 `tool_result_message` |
| 继续直到最终答案 | Executor `for step in range(MAX_STEPS)` |

---

## 4. 工具定义（≥3 个）

所有工具实现统一协议（`tools/base.py`）：

```python
class Tool(Protocol):
    name: str
    description: str
    parameters: dict          # JSON Schema, 传给 LLM function calling
    async def run(self, args: dict, session: "Session") -> str: ...
```

| 工具 | 功能 | 跨轮次 | 备注 |
|------|------|--------|------|
| `calculator` | 算术表达式求值 | 否 | `ast.parse` + 白名单运算符，禁 `eval` |
| `search` | mock 搜索 | 否 | 预设语料 dict, 关键词命中返回 |
| `todo` | create / list / update | **是** | 写 `session.memory.todos`，**跨轮次核心载体**。id 由 `create` 自动生成（session 内自增整数，如 `1`/`2`，对外显示 `#1`） |

`todo` 工具签名示例（JSON Schema）：

```json
{
  "name": "todo",
  "description": "管理任务列表。用于规划任务、记录进度、跨轮次追踪。",
  "parameters": {
    "type": "object",
    "properties": {
      "action": {"type": "string", "enum": ["create", "list", "update"]},
      "title":  {"type": "string", "description": "create 时必填"},
      "id":     {"type": "string", "description": "update 时必填"},
      "status": {"type": "string", "enum": ["PLANNED", "IN_PROGRESS", "DONE"]}
    },
    "required": ["action"]
  }
}
```

---

## 5. Session / Memory 设计（笔试核心）

### 5.1 数据结构（session/models.py）

```python
@dataclass
class TodoItem:
    id: str
    title: str
    status: str        # PLANNED | IN_PROGRESS | DONE
    created_at: str

@dataclass
class Memory:
    todos: list[TodoItem]       # 跨轮次任务状态 (族 F 的状态载体)
    plan: list[str]             # Planner 产出的步骤 (族 D)；Replanner 修订后覆盖此处
    lessons: list[str]          # Reflexion 产出的教训 (族 E)
    workspace: dict             # ReWOO 变量绑定 (族 C′)：{"E1": result, "E2": result}，solver 读取

@dataclass
class Message:
    role: str                   # user | assistant | tool
    content: str
    tool_call_id: str | None = None

@dataclass
class Session:
    id: str
    messages: list[Message]
    memory: Memory
    fsm_state: str              # 族 F 当前主状态
    created_at: str
    updated_at: str
    step_count: int
```

### 5.2 持久化（session/store.py）

- 每个 `session_id` 一个文件：`sessions/{session_id}.json`
- `load()`：读 JSON → 反序列化 Session；不存在则新建。
- `save()`：序列化 → 写文件，`encoding="utf-8"`。
- `list()`：列目录返回所有 session 摘要。
- 并发：文件级写入用临时文件 + `os.replace` 原子替换。

### 5.3 Memory 召回时机与放置方式（README 重点章节）

> 笔试明确要求："memory 的召回时机与放置方式说明"。

**召回时机（when）**：
1. **每轮开始**：`Agent.chat` 入口 `store.load(session_id)` → 拿回完整 Session（messages + memory + fsm_state）。
2. **每次 Executor 构造请求**：`_build_messages` 从 `session.memory` 取当前 todos / plan / lessons，注入 system prompt。
3. **每轮结束**：`store.save(session)` → 工具写过的 memory、Reflexion 产出的 lesson、FSM 新状态全部落盘。
4. **跨进程重启**：因 JSON 持久化，进程重启后下轮仍能 load 回状态 → 跨轮次继续执行成立。

**放置方式（where/how）**：
- memory 作为**独立结构化字段**，**不混进 `messages` 历史**（避免 token 爆炸 + 避免污染对话上下文）。
- 按需注入到 **system prompt 顶部**，分三段：

```
[SYSTEM]
你是任务执行 Agent。可调用工具：calculator / search / todo。

【当前任务列表】     ← memory.todos（todo 工具维护）
- [#1] 写季度报告大纲  [IN_PROGRESS]
- [#2] 收集数据       [PLANNED]

【执行计划】         ← memory.plan（Planner 产出，若有）
1. ...
2. ...

【过往教训】         ← memory.lessons（Reflexion 产出，若有）
- 上次 calculator 收到非数字参数会报错, 先校验类型
```

- **为什么放 system prompt 而非 user message**：memory 是稳定上下文，不是用户当前指令；放 system 每轮自动生效，LLM 不会把它当一次性输入。

### 5.4 跨轮次继续执行场景（验收剧本）

| 轮 | 用户输入 | Router | 执行 | memory 变化 |
|----|---------|--------|------|------------|
| 1 | "帮我规划写季度报告" | PLAN_REQUIRED | Planner 出 3 步；Executor 逐步 → `todo.create` ×3 | todos=[3 条 PLANNED], plan=[3 步] |
| 2 | "第一步开始做了" | SIMPLE_TOOL | Executor → `todo.update(#1, IN_PROGRESS)` | todos[0].status=IN_PROGRESS |
| 3 | "进度怎么样了" | SIMPLE_TOOL | Executor → `todo.list` 读 memory.todos → 基于状态回答 | 无新增，读回现有 |
| 4 | "第二步报错了" | SIMPLE_TOOL | Executor 工具失败 → Reflexion 反思 → 存 lesson → 重试 | lessons += [教训] |

关键：轮 3 不当新问题处理，而是 load 回轮 2 的 todos 状态回答。

---

## 6. 异常处理与限制

| 场景 | 处理 |
|------|------|
| 工具抛异常 | Executor catch → 作为 `tool_result` 喂回 LLM（`"ERROR: ..."`）+ 触发 Reflexion → 存 lesson → 继续 loop |
| LLM API 异常（网络/限流/鉴权） | 上抛 → FastAPI 捕获 → HTTP 500 + 错误体；session 不 save（保持上一致状态） |
| JSON 文件损坏 | load 时 try/except，损坏则备份后重建空 session，trace 记录 |
| 达到 MAX_STEPS | Executor 截断，返回强制结束语，trace 标 `truncated=true` |
| Replan 超限（`MAX_REPLANS`） | 停止重规划，用旧 plan 剩余步继续，trace 标 `replan_capped`，最终走 RESPONDING |
| ReWOO solver 证据不足 | 升级 REPLANNING（族 D′）重规划；若 REPLANNING 也失败 → RESPONDING 兜底回答 |
| FSM 非法转移 | 抛 `InvalidTransition`，trace 记录，session 回 IDLE |
| Router 误判 | 允许 Executor 内 function-calling 兜底（LLM 仍可自主决定调工具） |
| 编码（Windows） | 所有文件 IO `encoding="utf-8"`；subprocess 传 `PYTHONIOENCODING=utf-8` |

`MAX_STEPS` 默认 10，`config.py` 可调。

---

## 7. Trace / 执行日志（trace/logger.py）

每 session 一个 `trace/{session_id}.jsonl`，每步一行：

```json
{"ts": "...", "type": "step", "step": 0}
{"ts": "...", "type": "llm_call", "step": 0, "tools_offered": ["calculator","search","todo"]}
{"ts": "...", "type": "tool_call", "step": 0, "name": "todo.create", "args": {"title":"写大纲"}}
{"ts": "...", "type": "tool_result", "step": 0, "result": "created #1"}
{"ts": "...", "type": "reflexion", "step": 1, "lesson": "..."}
{"ts": "...", "type": "replan", "count": 1, "reason": "retry_exhausted", "revised_steps": 2}
{"ts": "...", "type": "rewoo_dag", "step": 2, "nodes": ["E1","E2"], "edges": [["E1","E2"]]}
{"ts": "...", "type": "rewoo_solve", "step": 2, "vars": ["E1","E2"], "evidence_sufficient": true}
{"ts": "...", "type": "route", "value": "plan_required"}
{"ts": "...", "type": "fsm", "from": "ROUTING", "to": "PLANNING"}
{"ts": "...", "type": "fsm", "from": "EXECUTING", "to": "REPLANNING"}
{"ts": "...", "type": "truncated"}
```

- `GET /trace/{session_id}`：返回日志，前端可折叠展示。
- 作用：笔试要求"工具调用 trace 或执行日志"；调试用；演示用。

---

## 8. LLM 接入（llm/client.py）

- 协议：DeepSeek 官方 OpenAI 兼容接口。
- SDK：`openai` Python 包，`base_url=https://api.deepseek.com`，`api_key` 从环境变量。
- 模型：`deepseek-chat`（支持 function calling）。
- 关键方法：
  - `chat_with_tools(messages, tools) -> Resp(tool_calls, text)`
  - `respond(messages, input) -> str`（Router=direct 路径，不传 tools）
  - `synthesize(plan, results) -> str`（Planner 路径末尾合成）
- 配置集中在 `config.py`，不硬编码 key。

---

## 9. Web 接口（main.py + static/index.html）

### FastAPI 路由
| 方法 | 路径 | 功能 |
|------|------|------|
| POST | `/chat` | body: `{session_id, input}` → `{answer, session_id}` |
| GET | `/sessions` | 列所有 session 摘要 |
| GET | `/sessions/{id}` | 单 session 详情（messages + memory） |
| GET | `/trace/{id}` | trace 日志 |
| GET | `/` | 返回 `static/index.html` |

### 前端（vanilla JS，单文件）
- 输入框 + 发送按钮 + 对话气泡区。
- session_id 首次访问生成 UUID 存 localStorage，后续复用（支撑跨轮次）。
- 侧栏可展开 trace（折叠每步的 tool_call/result）。

---

## 10. 测试策略

| 层 | 测试 | 说明 |
|----|------|------|
| 工具 | `test_tools.py` | calculator 安全性（注入用例）、search 命中、todo CRUD |
| FSM | `test_fsm.py` | 合法/非法转移、todo 子状态、REPLANNING 4 条新转移 |
| Executor | `test_executor.py` | mock LLMClient：①无 tool_call 直接返 ②一轮 tool_call 后返 ③max_steps 截断 ④工具异常触发 reflexion ⑤返回 `needs_replan` 标志 |
| Replanner | `test_replanner.py` | mock：①retry_exhausted 触发 replan ②`MAX_REPLANS` 截断 ③修订后剩余步骤生效、已完成步不重跑 |
| ReWOO | `test_rewoo.py` | mock：①独立步骤打成 DAG ②变量绑定落 `memory.workspace` ③solver 判证据不足 → 触发 replan（C′→D′ 链路） |
| 跨轮次 | `test_session_cross_turn.py` | §5.4 剧本：轮1建 todo → 轮2读回，断言 memory 持久 |
| 集成 | 手动 + 录屏 | 真 DeepSeek API 跑 §5.4 全剧本 |

LLM 调用在单测里全部 mock，避免消耗 API + 保证确定性。

---

## 11. 提交物清单（对应笔试要求）

| 笔试要求 | 交付 |
|---------|------|
| 代码链接 | 本仓库 |
| 录屏 | `docs/demo/` 下终端/网页录屏（README 链接） |
| README：运行方式 | 安装、配置 API key、`uvicorn main:app`、访问 localhost |
| README：系统设计 | 贴本 spec 精简版 + 架构图 |
| README：memory 召回时机与放置 | 直接引用 §5.3 |
| AI Prompt 与问题解决记录 | `PROMPTS.md`（记录 brainstorm 过程的 prompt、踩坑、决策） |

---

## 12. 验收 Checklist

- [ ] 真实 DeepSeek API 跑通多轮对话
- [ ] Router 正确分流三种路径
- [ ] Executor function-calling loop 跑通（含 max_steps 截断）
- [ ] 工具异常触发 Reflexion 且 lesson 落 memory
- [ ] Reflexion 重试耗尽触发 REPLANNING，Replanner 修订剩余 plan 且 `MAX_REPLANS` 截断生效
- [ ] ReWOO：独立步骤簇打成 DAG，worker 绑变量并行执行，结果落 `memory.workspace`，solver 合成
- [ ] ReWOO solver 判证据不足时升级 REPLANNING（C′ → D′ 兜底链路通）
- [ ] FSM 非法转移被拒（含 REPLANNING 相关 4 条新转移）
- [ ] §5.4 四轮剧本：进程重启后轮 3/4 仍能读到前序 memory
- [ ] trace 完整记录每步，前端可展示
- [ ] README 含运行/设计/memory 三段
- [ ] PROMPTS.md 含 prompt 与问题记录
- [ ] 单测全绿（mock LLM）

---

## 13. 风险与对策

| 风险 | 对策 |
|------|------|
| DeepSeek function calling 偶发不遵守 schema | Executor 兜底：tool_call 参数缺字段 → 报错回喂 LLM；Router 也允许直入 Executor 让 LLM 自纠 |
| DeepSeek 国内访问 | base_url 可配；README 说明需自备可用 key/网络 |
| JSON 并发写冲突 | 原子替换 `os.replace`；单 session 串行（同 session_id 请求排队） |
| minimal 张力（分层重） | README 显式说明设计取舍；模块边界清晰便于面试讲解 |
| Windows 编码 | 统一 utf-8，见 §6 |

---

## 附录 A：配置项（config.py）

```python
DEEPSEEK_API_KEY = env("DEEPSEEK_API_KEY")
DEEPSEEK_BASE_URL = "https://api.deepseek.com"
MODEL = "deepseek-chat"
MAX_STEPS = 10                  # 单步 Executor 内 function-calling 循环上限
MAX_REPLANS = 2                 # 单轮 plan 重规划上限 (族 D′)，防爆循环
REWOO_PARALLEL_ENABLED = True   # ReWOO (族 C′) 独立步骤并行执行开关
SESSION_DIR = "sessions"
TRACE_DIR = "trace"
HOST = "127.0.0.1"
PORT = 8000
```

## 附录 B：依赖（pyproject.toml）

- `fastapi`, `uvicorn`
- `openai`（指向 DeepSeek）
- `pydantic`
- `httpx`（openai 依赖）
- dev: `pytest`, `pytest-asyncio`
