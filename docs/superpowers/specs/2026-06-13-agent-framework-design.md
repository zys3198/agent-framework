# 最小可用 Agent — 系统设计 Spec

- **日期**: 2026-06-13
- **项目**: agent-framework
- **状态**: Updated 2026-06-25（Phase 0-4 实现后回写；ReWOO/Replanner/FSM 已删，补 Memory System）
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
笔试题要求 "最小可用"，本设计采用**分层 runtime**（Router + Planner + Executor + Reflexion 四族），代码量高于纯最小实现。分层的好处：每一族职责单一、可独立测试、README 能讲清架构选型；代价是代码量约为纯最小实现的 2–3 倍。

> **实现演进说明**：初版设计过 5+2 族（含 ReWOO C′、Replanner D′、FSM 模块 F），在 Phase 0（commit `f4c9de9` 删 dead ReWOO/Replanner、`fdc9728` 删 dead FSM 模块）证实为 dead path 后删除。ReWOO 的并行/重规划假设在实际任务中未触发，且 function-calling loop 本身已覆盖等价能力；独立 FSM 模块降级为 `Session.fsm_state` 字符串常量（IDLE/PLANNING/EXECUTING/REFLECTING/WAITING），由 Agent 内联设置，不再强制合法转移。Phase 2-4 新增 Memory System（file-based memory / 异步召回 / AGENTS.md loader / 三层 compaction / per-session lock / Contract C），见 §5.5。

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
│   fsm_state 字段驱动状态记录，串联下面四层（FSM 模块已删） │
│                                                          │
│   ┌─────────┐   ┌──────────┐   ┌──────────┐   ┌───────┐	│
│   │ Router  │──▶│ Planner  │──▶│ Executor │──▶│Reply │ │
│   │  (I)    │   │   (D)    │   │   (C+E)  │   │      	│ │
│   └─────────┘   └──────────┘   └──────────┘   └───────┘	│
│        │              │              │                   │
│        │         生成 plan      function-calling loop    │
│        │        （D′/C′ 已删）   + Reflexion 自纠 (E)      │
│        │                      + step-0 异步 memory 召回  │
└────────┬──────────────┬──────────────┬──────────────────┘
         │              │              │
┌────────▼──────────────▼──────────────▼──────────────────┐
│  SessionStore (session/)   ToolRegistry (tools/)         │
│  TraceLogger (trace/)      LLMClient (llm/)              │
└─────────────────────────────────────────────────────────┘
```

### 1.2 族职责映射（现状 4 族；原 D′ Replanner / C′ ReWOO / F FSM 已删）

| 族 | 模块 | 职责 | 触发时机 |
|----|------|------|---------|
| **I — Router** | `runtime/router.py` | 第一道判断：DIRECT / SIMPLE_TOOL / PLAN_REQUIRED。一次轻量 LLM 调用（不带 tools，输出 enum 三选一）；把 todos/最近 lessons 摘要注入 router prompt（`_build_router_context`） | 每轮输入后最先执行 |
| **D — Planner** | `runtime/planner.py` | 复杂任务生成有序步骤列表（`Step` 仅 `prompt` 字段）；接收 `project_context`（AGENTS.md 三层聚合） | Router 判 `plan_required` |
| **C — Executor** | `runtime/executor.py` | Function-calling 循环：LLM ↔ 工具分发 ↔ 结果回填；内含 Reflexion（微观自纠）+ step-0 异步 memory 召回 | 每步执行；SIMPLE_TOOL 直入 |
| **E — Reflexion** | `runtime/reflexion.py` | 工具失败时让 LLM 产一句 lesson 存 `memory.lessons`；攒满 `_EXHAUSTION_THRESHOLD=3` 条判穷尽，返回 `Outcome(needs_replan=True)` | Executor 捕获工具异常时 |

**已删除（Phase 0，commit `f4c9de9`/`fdc9728`）**：
- ~~D′ — Replanner~~：宏观重规划。证实为 dead path（实际任务不触发）后删除。失败步不再重规划，改为交 `synthesize` 标注 `[STEP i FAILED]` 让模型知道该步失败。
- ~~C′ — ReWOO~~：微观并行 DAG（`E1/E2` 变量绑定 + solver 合成）。删除，function-calling 单步已支持并行 tool_call。
- ~~F — FSM 模块~~：独立状态机模块删除。降级为 `Session.fsm_state` 字符串常量，由 Agent 内联设置，不再强制合法转移、不抛 `InvalidTransition`。

> 备注：**function calling 是 DeepSeek（OpenAI 兼容）API 的接口协议，不是 Agent 框架**。loop、工具分发、session、memory、trace、状态字段全部自实现，满足"核心 runtime 自己实现"。

### 1.3 状态机（F）定义

> **现状**：独立 FSM 模块已删除（Phase 0 `fdc9728`）。下方为**逻辑语义**描述，实际实现是 `Session.fsm_state` 字符串常量（IDLE/PLANNING/EXECUTING/REFLECTING/WAITING），由 `Agent._chat_impl` 内联设置；不再强制合法转移，不抛 `InvalidTransition`。REFLECTING→REPLANNING 链路因 Replanner 删除不再成立（Reflexion 攒满 3 条 lesson 判穷尽 → 返回 `needs_replan`，由 synthesize 标 `[STEP i FAILED]`）。

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

合法转移（**逻辑语义**；独立 FSM 模块已删，`InvalidTransition` 不再抛，`fsm_state` 仅作记录）：

> 现状：下表中带 `REPLANNING` 的 4 条转移因 Replanner 删除已失效。实际路径是 Reflexion 攒满 3 条 lesson → 返回 `needs_replan` → `synthesize` 标 `[STEP i FAILED]`，不再进入 REPLANNING 状态。

| from | event | to |
|------|-------|-----|
| IDLE | input | ROUTING |
| ROUTING | route=direct | RESPONDING |
| ROUTING | route=simple | EXECUTING |
| ROUTING | route=plan | PLANNING |
| PLANNING | plan_ready | EXECUTING |
| EXECUTING | tool_error | REFLECTING |
| REFLECTING | retry | EXECUTING |
| ~~EXECUTING | replan_needed | REPLANNING~~ | 已失效（Replanner 删） |
| ~~REFLECTING | retry_exhausted | REPLANNING~~ | 已失效（Replanner 删） |
| ~~REPLANNING | plan_updated | EXECUTING~~ | 已失效（Replanner 删） |
| ~~REPLANNING | abort | RESPONDING~~ | 已失效（Replanner 删） |
| EXECUTING | done | RESPONDING |
| RESPONDING | replied | IDLE |

> **REFLECTING vs REPLANNING 区分（历史）**：REFLECTING = 微观自纠，同一步学 lesson 重试（族 E）；REPLANNING = 宏观自纠（族 D′，已删）。现状链路：`tool_error → REFLECTING → 重试 → reflexion_exhausted → 返回 needs_replan → synthesize 标 [STEP i FAILED]`。~~`MAX_REPLANS`（默认 2）防爆循环~~ 已删。

Todo 子状态（memory 内）：`NONE → PLANNED → IN_PROGRESS → DONE`，由 `todo` 工具驱动。

### 1.4 设计空间（已评估，部分纳入）

- **ToT / LLMCompiler**：推理搜索类，agent 场景杀鸡牛刀。不实现。
- **ReAct 文本解析**：function-calling 已覆盖等价能力且更稳，故不采用 prompt 解析路径。
- **ReWOO**：~~已纳入~~（族 C′，Phase B 曾实现）。微观并行子模式：Planner 检测独立步骤簇 → 打成 DAG（`E1=...`, `E2=...(依赖E1)`）→ worker 绑变量执行（不走 LLM）→ solver 一次合成。**Phase 0 已删除**（commit `f4c9de9`）：证实为 dead path + function-calling 单步已支持并行 tool_call。Replanner（族 D′）为其兜底的链路也一并删除。
- **ReWOO（已删除）**：~~已纳入~~。初版按上方设计实现过 C′/D′，Phase 0（`f4c9de9`）证实 ReWOO 的并行/重规划在实际任务不触发、且 function-calling 单步已支持并行 tool_call，遂删除 ReWOO + Replanner。保留此条记录设计空间评估历史。
- **Multi-Agent / CodeAct**：超出 minimal，不做。

---

## 2. 模块清单与职责

```
agent_framework/
├── main.py                 # FastAPI app + build_agent 装配 + 路由 + 全局异常处理
├── config.py               # env 读取: API_KEY/BASE_URL/MODEL/MAX_STEPS/SESSION_DIR/HOST/PORT
├── llm/
│   └── client.py           # DeepSeek (OpenAI 兼容) wrapper: chat_with_tools/respond/synthesize
├── ctx/                    # Phase 3 新增
│   └── compactor.py        # 三层 compaction: spill 大结果 / microcompact / auto_compact (LLM 摘要)
├── tools/
│   ├── base.py             # Tool 协议 + ToolRegistry (register/dispatch/schema 导出)
│   ├── calculator.py       # 安全表达式求值 (ast, 白名单运算符)
│   ├── search.py           # mock 搜索 (预设语料)
│   ├── todo.py             # 跨轮次核心: create/list/update, 写 session.memory.todos
│   └── memory.py           # Phase 2 新增: WriteMemory (门控写入) + ReadMemoryBody (懒读正文)
├── runtime/
│   ├── router.py           # Router.classify(input, memory) → Route (族 I)
│   ├── planner.py          # Planner.make_plan(input, memory, project_context) → list[Step] (族 D)
│   ├── executor.py         # Executor.run(session, prompt, trace, project_context) → Outcome (族 C+E, 含异步召回)
│   ├── reflexion.py        # Reflexion.reflect(call, error, memory) → Lesson(text, reflexion_exhausted) (族 E)
│   ├── recaller.py         # Phase 2 新增: 异步 memory 召回 + filter_tool_usage 工具规避
│   ├── agent_memory.py     # Phase 2 新增: load_project_context 三层 AGENTS.md 聚合
│   └── agent.py            # Agent 编排四族 + per-session lock + compactor + Contract C
│   # 已删 (Phase 0 commit f4c9de9/fdc9728): rewoo.py / replanner.py / fsm.py
├── session/
│   ├── models.py           # Session / Memory / TodoItem / Step / MemoryEntry / Message
│   └── store.py            # JSON 持久化 (原子 os.replace, 损坏备份重建, delete, list)
├── trace/
│   └── logger.py           # 每步落 trace/{sid}.jsonl
├── static/
│   └── index.html          # 暖色极简聊天 UI (vanilla JS fetch, 无构建)
├── tests/                  # 18 个测试文件 (见 §10)
├── tests/                  # 17 个测试文件 (见 §10)
├── README.md
└── PROMPTS.md              # AI Prompt 与问题解决记录
```

**单一职责边界**（每个模块应能独立理解 + 独立测试）：
- `router` 只决定走哪条路径，不执行。
- `planner` 只产出步骤列表，不执行。
- `executor` 只跑 function-calling loop，不决定路由；返回 `needs_replan` 标志，不自行重规划。
- `reflexion` 只产出 lesson，不重试（重试由 executor 控制）。
- ~~`replanner`~~ / ~~`rewoo`~~ / ~~`fsm`~~ 三模块已删（Phase 0）。`fsm_state` 字段由 agent 内联设置，不强制合法转移。
- `store` 只读写 JSON，不解析语义。

---

## 3. 核心循环（笔试要求的最小 loop，落在 Executor + Agent 编排）

### 3.1 Agent 顶层编排（runtime/agent.py）

```python
async def chat(self, session_id, user_input) -> str:
    async with (await self._lock_for(session_id)):     # per-session 锁串行化 load→modify→save
        return await self._chat_impl(session_id, user_input)

async def _chat_impl(self, session_id, user_input) -> str:
    session = self.store.load(session_id)              # 1. 召回 (load)
    session.fsm_state = IDLE
    if self._compactor is not None:                    # Phase 3: 路由前先 compact（阈值下为 no-op）
        if await self._compactor.compact(session):
            self.store.save(session)                   # compact 产生变更先落盘
    project_context = load_project_context(workspace_root, user_home)  # AGENTS.md 三层聚合（每轮 fresh）
    trace = TraceLogger(trace_dir / f"{session_id}.jsonl")

    route = await self.router.classify(user_input, session.memory)     # 族 I
    trace.log_route(route.value)

    if route == DIRECT:
        session.fsm_state = WAITING
        answer = await self.llm.respond([system+memory_index+history], user_input)
        # DIRECT: agent 自己写 user + assistant（工具路径由 executor 写，见 Contract C）
        session.messages.append(Message("user", user_input))
        session.messages.append(Message("assistant", answer))

    elif route == SIMPLE_TOOL:
        session.fsm_state = EXECUTING
        outcome = await self.executor.run(session, user_input, trace, project_context)  # 族 C+E+召回
        session.fsm_state = WAITING
        answer = outcome.text                           # needs_replan 仅记 trace（Phase 0 无 replanner）

    else:  # PLAN_REQUIRED
        session.fsm_state = PLANNING
        plan = await self.planner.make_plan(user_input, memory, project_context)  # 族 D
        session.memory.plan = plan
        session.fsm_state = EXECUTING
        results = {}
        for i in range(len(plan)):                    # Phase 0: 失败步不重规划
            outcome = await self.executor.run(session, plan[i].prompt, trace, project_context)
            results[i] = outcome                       # needs_replan 交 synthesize 标 [STEP i FAILED]
        answer = await self.llm.synthesize(plan, results, project_context)
        session.fsm_state = WAITING
        session.messages.append(Message("assistant", answer))  # 仅写最终合成答案

    session.fsm_state = IDLE
    session.updated_at = now()
    self.store.save(session)                          # 2. 持久化 (save)
    trace.close()
    return answer
```

### 3.2 Executor function-calling loop（runtime/executor.py）

```python
async def run(self, session, user_input, trace) -> Outcome:
    # Contract C: executor 是 messages 唯一写入者；先 persist user turn
    session.messages.append(Message("user", prompt))
    messages = [system_prompt(memory)] + [memory_index_msg(entries, project_context)] + history
    # step-0 异步 memory 召回（与首步 ReAct 并行，只看 name+description，见 §5.5）
    recall_task = create_task(recaller.recall(prompt, entries)) if entries else None

    for step in range(MAX_STEPS):
        trace.log_step(step)
        resp = await self.llm.chat_with_tools(messages, tools.schemas())
        # step 0 后注入召回结果（去重 + 工具规避：排除当前工具的 usage 词条，保留 caveat）
        if step == 0 and recall_task is not None:
            inject_recalled(await recall_task, first_tool=resp.tool_calls[0].name if resp.tool_calls else None)
        if not resp.tool_calls:
            session.messages.append(Message("assistant", resp.text))
            return Outcome(text=resp.text, needs_replan=False)

        session.messages.append(Message("assistant", resp.text, tool_calls=resp.tool_calls))
        for tc in resp.tool_calls:
            trace.log_tool_call(step, tc.name, tc.args)
            try:
                result = await registry.dispatch(tc, session)          # todo/memory 写 session.memory
            except Exception as e:
                result = f"ERROR: {e}"
                lesson = await reflexion.reflect(tc, result, memory)   # 族 E
                memory.lessons.append(lesson.text)
                trace.log_reflexion(step, lesson.text)
                if lesson.reflexion_exhausted:                          # 攒满 3 条判穷尽
                    flush_pending_tools(resp.tool_calls, tc.id, ...)   # 防 orphan tool message（Contract C）
                    return Outcome(text=result, needs_replan=True)
            trace.log_tool_result(step, result)
            # Contract C: tool message 在任何 early return 之前 append，防 reloaded session 出现孤儿 tool
            session.messages.append(Message("tool", result, tool_call_id=tc.id))

    trace.log_truncated()
    session.messages.append(Message("assistant", "(truncated)"))
    return Outcome(text="(truncated)", needs_replan=True)
```

> `Outcome = (text: str, needs_replan: bool)`。**Phase 0 后**：`needs_replan=True`（Reflexion 攒满 3 条 lesson 判穷尽 / max_steps 触顶）不再走 Replanner，而是记入 trace，由 `synthesize` 标注 `[STEP i FAILED]` 让模型知道该步失败。

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
| `write_memory` | 写一条 memory entry | **是** | Phase 2。门控写入：拒代码/路径/git 命令；feedback/project 强制 `Rule/Why/How to apply`；project 禁相对时间；同 (type,name) 覆盖更新。详见 §5.5 |
| `read_memory_body` | 按 id 读 entry 正文 | 否 | Phase 2。渐进披露：正文默认不加载，模型按需拉 |

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
class Step:                     # 族 D 产出的单步；from_dict 容忍 legacy is_rewoo_cluster 等多余键
    prompt: str

@dataclass
class TodoItem:
    id: str
    title: str
    status: str = "PLANNED"     # PLANNED | IN_PROGRESS | DONE
    created_at: str

@dataclass
class MemoryEntry:              # Phase 2: file-based memory 条目（见 §5.5）
    id: str
    type: str                   # user | feedback | project | reference
    name: str
    description: str
    keywords: list[str]
    content: str                # 正文懒加载，索引只放 name/description/keywords
    saved_at: str

@dataclass
class Memory:
    todos: list[TodoItem]       # 跨轮次任务状态
    plan: list[Step]            # Planner 产出的步骤 (族 D)
    lessons: list[str]          # Reflexion 产出的教训 (族 E)
    entries: list[MemoryEntry]  # Phase 2: file-based memory 条目
    workspace: dict             # 保留字段（ReWOO 已删，目前未用）

@dataclass
class Message:
    role: str                   # user | assistant | tool
    content: str
    tool_call_id: str | None = None
    tool_calls: list[dict] | None = None   # assistant 携带 OpenAI function-calling 调用（持久化保配对）

@dataclass
class Session:
    id: str
    messages: list[Message]
    memory: Memory
    fsm_state: str = "IDLE"     # 字符串常量（FSM 模块已删，见 §1.3）
    created_at: str
    updated_at: str
    # 注：原设计 step_count 字段已删
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
2. **每次 Executor 构造请求**：`build_system_prompt(memory)` 从 `session.memory` 取当前 todos / plan / lessons，注入 system prompt。
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

> **Phase 2-4 扩展**：除上述结构化 memory（todos/plan/lessons → system prompt）外，另有 file-based memory 条目系统（`MemoryEntry`）、异步召回（Recaller）、AGENTS.md 三层聚合、三层 compaction。这些是 spec 初版未覆盖的增量，详见 §5.5。

### 5.4 跨轮次继续执行场景（验收剧本）

| 轮 | 用户输入 | Router | 执行 | memory 变化 |
|----|---------|--------|------|------------|
| 1 | "帮我规划写季度报告" | PLAN_REQUIRED | Planner 出 3 步；Executor 逐步 → `todo.create` ×3 | todos=[3 条 PLANNED], plan=[3 步] |
| 2 | "第一步开始做了" | SIMPLE_TOOL | Executor → `todo.update(#1, IN_PROGRESS)` | todos[0].status=IN_PROGRESS |
| 3 | "进度怎么样了" | SIMPLE_TOOL | Executor → `todo.list` 读 memory.todos → 基于状态回答 | 无新增，读回现有 |
| 4 | "第二步报错了" | SIMPLE_TOOL | Executor 工具失败 → Reflexion 反思 → 存 lesson → 重试 | lessons += [教训] |

关键：轮 3 不当新问题处理，而是 load 回轮 2 的 todos 状态回答。

### 5.5 Memory System (Phase 2-4，初版 spec 未覆盖)

Claude Code 风格的 file-based memory：渐进披露 + 异步召回 + 分层 compaction，**非 vector RAG**。

#### File-based memory（索引常驻，正文懒加载）

Memory 条目存在 `Session.memory.entries`（非 embedding store）。每个 `MemoryEntry` = id/type/name/description/keywords/content/saved_at。每轮由 `build_memory_context_message()`（`runtime/agent.py`）构建完整索引，注入为 **user message**（软约束，非 system prompt）。索引只含 name/description/keywords/saved_at（约 100 token/条）；`content` 默认不加载，模型通过 `read_memory_body` 工具按需拉。

索引两道硬上限（先到为准）：`_MEMORY_INDEX_MAX_LINES = 200` 行，`_MEMORY_INDEX_MAX_BYTES = 25*1024` 字节。

四种 type（`MEMORY_ENTRY_TYPES`，`session/models.py`）：user / feedback / project / reference。feedback、project 的 content 必须含 `Rule`/`Why`/`How to apply` 三个标记；project 的相对时间词（yesterday/上周/2 days ago）被拒，必须绝对日期。

#### 写入门控（`write_memory`，`tools/memory.py`）

拒绝「查得到而非值得记」的条目。启发式 `_looks_like_code_or_path` 检查代码围栏、赋值/def/class/import、文件路径（带扩展名或盘符）、git 子命令。另强制：合法 type、非空 name/content、keywords 为字符串列表、feedback/project 的结构标记、project 绝对日期。

同 (type, name)（大小写不敏感）匹配则**原地更新**（latest wins）；否则 `_next_id` 追加新条目。

#### AGENTS.md loader（`runtime/agent_memory.py`）

`load_project_context(workspace_root, user_home)` 按序聚合三层永久上下文：
1. **Project AGENTS.md** — 沿 `workspace_root` 父链向上遍历，每层读 AGENTS.md（最内层最后出现，优先级最高）。
2. **Local** — `workspace_root/AGENTS.local.md`（git-ignored 个人覆盖）。
3. **User** — `~/.agents/AGENTS.md`。

每轮 `Agent.chat` 重读，经 `build_memory_context_message` / `Planner.make_plan` / `Executor.run` 的 `project_context` 透传。因每轮重读，**不**折进 compaction 摘要。

#### 异步召回（`runtime/recaller.py`，接入 `runtime/executor.py`）

`Recaller.recall(query, entries)` 用中等模型按 name+description 筛相关 id（渐进披露——筛选时模型看不到 content）。返回严格 JSON `{"ids":[...]}`，被指示「保守，仅明显相关」。

Executor 接入：
- **step 0** 若配置了 recaller 且有 entries，召回作为 `asyncio.Task` 与主模型首步 ReAct **并行**启动。
- step 0 后 await 该任务，应用两道本地过滤（无第二次 LLM round-trip）：①去重——已在索引里的 id 丢弃；②工具规避（`filter_tool_usage`）——LLM 选定工具名已知后，排除 description 含 `how to use`/`usage`/`使用说明`/`用法` 的条目，但**保留** caveat/bug 条目。
- 幸存 id 注入为 `Recalled from memory:` user message，每行标 saved_at + 「使用前核实」提醒。

#### 分层 compaction（`ctx/compactor.py`，demo 精简到 3 层）

`Compactor.compact(session)` 在每个 `Agent.chat` 起始、路由前运行。层在阈值下为 no-op，常见路径开销低。

- **Layer 1 — 大结果溢写**（`spill_large_results`）：tool result > `large_result_bytes`（默认 4096）写到 `spill_dir`（sha256 前缀），message 替换为 80 字预览 + `[spill:<digest>]` 标记（可 Read 找回）。
- **Layer 2 — microcompact**（`microcompact`）：只留最近 `microcompact_keep`（默认 5）个 tool result，丢弃更早的 tool message 及其前导 assistant(tool_calls)。状态信息（todos/plan/workspace）在 `session.memory`，本层**不碰**。
- **Layer 3 — Auto-Compact**（`auto_compact`）：估算 token（`_estimate_tokens`，约 4 字节/token，无 tokenizer 依赖）超 `auto_compact_tokens`（默认 8000）时，整段对话送 LLM 摘要器。输出替换 message 列表为 **3-segment 链**：
  1. boundaryMarker — `[COMPACT] session continuation...`，含压缩前 token 数与最后消息引用（让模型知道是交接而非新开始）。
  2. summary — 固定 9 节摘要（Primary Request and Intent / Key Technical Concepts / Files and Code Sections / Errors and fixes / Problem Solving / All user messages enumerated / Pending Tasks / Current Work at file+function granularity / Optional Next Step）。
  3. attachments — todos/plan/workspace **原样**恢复（状态信息的 externalized recall）。

#### 信息分流

compaction 按信息类型而非统一文本路由：
- **语义信息**（意图/决策/解题）→ 进 summary。
- **状态信息**（todos/plan/workspace）→ 进 attachments，原样恢复——绝不被摘要压平，绝不被 microcompact 丢弃。
- **永久上下文**（AGENTS.md）→ 每轮重读，不进摘要。
- **配置**（system prompt/工具表）→ 每次请求重建。

#### 安全网

- **熔断**：`auto_compact` 统计连续摘要失败；达 `circuit_breaker_limit`（默认 3）后 `circuit_tripped = True`，拒绝后续压缩而非在坏摘要器上空转。
- **递归守卫**：`_is_compaction_output` 查前 3 条消息标记（`[COMPACT]`/`session continuation`/`interrupted context`）；若对话已像压缩产物则不再压缩（防 infinite compact-the-compaction）。

#### 为什么选 compaction 而非其他

评估并否决的三种替代（`docs/PLAN.md`）：
- **滑动窗口**（超 N 丢最旧）：简单，但丢早期 system 指令与任务框定，随对话增长丢失原始意图。
- **纯摘要**（整段历史压成一坨）：丢细粒度、斩依赖——后步依赖的 tool result 被压成散文，破坏 OpenAI API 要求的 assistant(tool_calls)→tool(result) 配对。
- **Vector RAG 召回**（embed 历史，top-k 检索）：破坏时序——按相似度而非发生顺序返回，加检索噪声与 embedding store 依赖。

选分层 compaction 的理由：状态信息原样存活（todos/plan/workspace 进 attachments，模型要用的不会丢给摘要）、语义信息进结构化摘要（可压不失叙事）、无检索排序问题（message 链保序，不重排）。代价：layer-3 触发时的摘要器 LLM 成本——由熔断与 8k 阈值限定。

#### per-session lock 与 Contract C

- **per-session lock**（`Agent._session_locks`）：`chat` 用 `asyncio.Lock` 串行化整段 load→modify→save 事务，防同 session 并发请求丢更新。
- **Contract C**：SIMPLE_TOOL 与 PLAN_REQUIRED 每步的 `session.messages` 写入权归 Executor；DIRECT 由 Agent 自己写 user+assistant。四消息序列（user→assistant(tool_calls)→tool→assistant）原子化，防 reloaded session 出现孤儿 tool message（API 会 400）。reflexion 耗尽时 `_flush_pending_tools` 把同批剩余 tool 也补成 error 结果再 return。

---

## 6. 异常处理与限制

| 场景 | 处理 |
|------|------|
| 工具抛异常 | Executor catch → 作为 `tool_result` 喂回 LLM（`"ERROR: ..."`）+ 触发 Reflexion → 存 lesson → 继续 loop |
| LLM API 异常（网络/限流/鉴权） | 上抛 → FastAPI 捕获 → HTTP 500 + 错误体；session 不 save（保持上一致状态） |
| JSON 文件损坏 | load 时 try/except，损坏则备份后重建空 session，trace 记录 |
| 达到 MAX_STEPS | Executor 截断，返回强制结束语，trace 标 `truncated=true` |
| Reflexion 穷尽（攒满 3 条 lesson 仍失败） | 返回 `needs_replan=True`，`_flush_pending_tools` 补齐剩余 tool message 防孤儿，由 synthesize 标 `[STEP i FAILED]` |
| Auto-Compact 摘要器连续失败 | 熔断（`circuit_breaker_limit=3`）后 `circuit_tripped`，拒绝后续压缩（见 §5.5） |
| ~~Replan 超限（`MAX_REPLANS`）~~ | 已删（Phase 0）：无 Replanner / MAX_REPLANS |
| ~~ReWOO solver 证据不足~~ | 已删（Phase 0）：无 ReWOO |
| ~~FSM 非法转移 抛 `InvalidTransition`~~ | 已删（Phase 0）：FSM 模块移除，`fsm_state` 仅记录 |
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
{"ts": "...", "type": "route", "value": "plan_required"}
{"ts": "...", "type": "truncated"}
```

> 现状（Phase 0 后）trace/logger.py 实际只 emit 7 种 type：`step` / `llm_call` / `tool_call` / `tool_result` / `reflexion` / `route` / `truncated`。上方 `replan` / `rewoo_dag` / `rewoo_solve` / `fsm` 四类已随 ReWOO/Replanner/FSM 删除而移除。

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
  - `synthesize(plan, results, project_context) -> str`（Planner 路径末尾合成，第三参透传 AGENTS.md 上下文）
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

> 现状：17 个测试文件，155 passed。test_fsm / test_replanner / test_rewoo / test_session_cross_turn 已随对应代码删除。

| 层 | 测试 | 说明 |
|----|------|------|
| 工具 | `test_tools.py` | calculator 安全性（注入用例）、search 命中、todo CRUD |
| Executor | `test_executor.py` | mock LLMClient：①无 tool_call 直接返 ②一轮 tool_call 后返 ③max_steps 截断 ④工具异常触发 reflexion ⑤返回 `needs_replan` 标志 |
| Router | `test_router.py` | DIRECT/SIMPLE_TOOL/PLAN_REQUIRED 分类 |
| Planner | `test_planner.py` | JSON 解析、Step 构造 |
| Agent 编排 | `test_agent.py` | DIRECT/SIMPLE_TOOL/PLAN_REQUIRED 三路径 + Contract C |
| Memory System | `test_compactor.py` / `test_recaller.py` / `test_agent_memory.py` / `test_models.py` | 三层 compaction、异步召回、AGENTS.md loader、数据模型 |
| Memory 工具 | `test_store.py` / `test_llm_client.py` / `test_config.py` | 持久化、DeepSeek wrapper、env 配置 |
| Web 层 | `test_web.py` / `test_trace.py` | FastAPI 路由、trace 日志 |
| Reflexion | `test_reflexion.py` | lesson 产出、exhaustion 阈值 |
| 跨轮次 | `test_integration.py` | §5.4 剧本回归：Contract C 四消息序列、reload 不变量、replan 标注 |
| 集成 | 手动 + 录屏 | 真 DeepSeek API 跑 §5.4 全剧本 |
| E2E | `test_e2e_deepseek.py` | 真 DeepSeek API（无 key 时 skip）：DIRECT/SIMPLE_TOOL/cross-turn memory/compaction |

除 `test_e2e_deepseek.py`（真 API）外，LLM 调用全部 mock（FakeLLM / ScriptedExecutor），避免消耗 API + 保证确定性。

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

> 回写于 2026-06-25（Phase 0-4 完成后）。

- [x] 真实 DeepSeek API 跑通多轮对话
- [x] Router 正确分流三种路径
- [x] Executor function-calling loop 跑通（含 max_steps 截断）
- [x] 工具异常触发 Reflexion 且 lesson 落 memory
- [x] Reflexion 穷尽返回 `needs_replan`，交 synthesize 标 `[STEP i FAILED]`（Phase 0：原 Replanner / `MAX_REPLANS` 已删）
- [-] ~~ReWOO：DAG / workspace / solver 合成~~（已删，Phase 0）
- [-] ~~ReWOO solver 升级 REPLANNING（C′ → D′ 兜底）~~（已删，Phase 0）
- [-] ~~FSM 非法转移被拒~~（FSM 模块已删，降级字符串常量）
- [x] §5.4 四轮剧本：进程重启后轮 3/4 仍能读到前序 memory
- [x] trace 完整记录每步，前端可展示
- [x] README 含运行/设计/memory 三段
- [x] PROMPTS.md 含 prompt 与问题记录
- [x] 单测全绿（除 4 个 test_e2e_deepseek 跑真 API 外全 mock，**155 passed**，2026-06-25 实测）
- [x] Memory System（Phase 2-4）：file-based memory 索引/正文懒加载、`write_memory` 门控、异步召回、AGENTS.md 三层聚合、三层 compaction（熔断 + 递归守卫）—— 见 §5.5

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
DEEPSEEK_API_KEY = os.environ.get("DEEPSEEK_API_KEY", "")
DEEPSEEK_BASE_URL = os.environ.get("DEEPSEEK_BASE_URL", "https://api.deepseek.com")
MODEL = os.environ.get("MODEL", "deepseek-chat")
MAX_STEPS = 10                  # 单步 Executor function-calling 循环上限（_int_env 可覆盖）
SESSION_DIR = BASE_DIR / "sessions"
TRACE_DIR = BASE_DIR / "trace"
HOST = "127.0.0.1"
PORT = 8000
# Compactor 阈值（硬编码于 ctx/compactor.py，非 config，列出供参考）：
#   large_result_bytes=4096 / microcompact_keep=5 / auto_compact_tokens=8000 / circuit_breaker_limit=3
# 已删配置（Phase 0）：MAX_REPLANS、REWOO_PARALLEL_ENABLED
```

## 附录 B：依赖（pyproject.toml）

- `fastapi`, `uvicorn`
- `openai`（指向 DeepSeek）
- `pydantic`
- `httpx`（openai 依赖）
- dev: `pytest`, `pytest-asyncio`
- `python-dotenv`（config.py 加载 .env）
