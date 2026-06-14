# agent-framework

从零实现的最小可用 Agent (自实现 runtime, 不依赖 LangChain / OpenHands)。多轮对话 + 会话持久化、基本循环 (输入 -> 判断直接答 / 调工具 -> 执行 -> 读结果 -> 继续)、>=3 个工具、DeepSeek (OpenAI 兼容) API、RePLANNING 动态重规划、ReWOO 计划/执行解耦 DAG、FastAPI Web 层。

---

## 运行方式

Python 3.12。代码与 venv 都在 `agent_framework/` 下:

```bash
cd agent_framework
python -m venv .venv
.venv/Scripts/python -m pip install -e ".[dev]"

# 配置 DeepSeek API Key —— 方式 A（推荐，.env 加载）
cp .env.example .env
#   编辑 .env 填入 DEEPSEEK_API_KEY=sk-...
# 方式 B：环境变量
export DEEPSEEK_API_KEY=sk-...

# 启动（在 agent_framework/ 目录内）
.venv/Scripts/python -m uvicorn main:app
# 浏览器打开 http://127.0.0.1:8000
```

**无 API Key 时**: app 正常启动, `/sessions`、`/trace` 等只读路由可用; `/chat` 返回 503 (`DEEPSEEK_API_KEY not configured`)。`.env` 被 `.gitignore` 忽略，不会泄露密钥。

### 前端

暖色极简聊天 UI（`static/index.html`，单文件 vanilla JS，无构建步骤）：
- 侧栏会话列表（标题 = 首条消息派生；空会话显示「新会话」），hover 右侧 × 删除
- 消息气泡（user 橙底 / assistant 白卡 / tool 等宽虚线框），`· · ·` thinking 动画
- session_id 存 localStorage，刷新/重启不丢

### 提交前四件套 (全绿才提交)

```bash
cd agent_framework
.venv/Scripts/python -m ruff check .
.venv/Scripts/python -m ruff format --check .
.venv/Scripts/python -m mypy
.venv/Scripts/python -m pytest -q
```

---

## 系统设计

分层 runtime + 会话级 FSM + Plan-and-Execute (宏观), 内含 REPLANNING (动态重规划) 与 ReWOO (微观计划/执行解耦)。完整 spec: `docs/superpowers/specs/2026-06-13-agent-framework-design.md`。

### 5+2 族分层 runtime

| 族 | 模块 | 职责 |
|---|---|---|
| I — Router | `runtime/router.py` | 一次轻量 LLM 调用, 分类输入为 DIRECT / SIMPLE_TOOL / PLAN_REQUIRED |
| D — Planner | `runtime/planner.py` | 复杂任务分解为有序 step 列表 (JSON 解析) |
| C — Executor | `runtime/executor.py` | function-calling 循环: LLM <-> tool dispatch <-> result 回填 |
| E — Reflexion | `runtime/reflexion.py` | 工具失败时学 lesson, 微观自纠 (同一步重试) |
| F — 状态跟踪 | `runtime/agent.py`（内联常量） | session 级状态（IDLE/PLANNING/EXECUTING/REFLECTING/WAITING），fsm 模块已删除（简化） |
| D' — Replanner | `runtime/replanner.py` | 宏观自纠: 基于已执行结果修订剩余 plan |
| C' — ReWOO | `runtime/rewoo.py` | 微观并行: plan DAG -> worker (tool 并行) -> solver 合成 |

### Agent 主循环

```
Agent.chat(session_id, user_input)
  -> store.load(session_id)        # 每轮开始: 拿回完整 Session
  -> Router.classify()             # 族 I: 选路径
  -> {
       DIRECT:        LLM.respond() -> Agent 追加 user+assistant
       SIMPLE_TOOL:   Executor.run() (族 C, 持有 session.messages 写权)
       PLAN_REQUIRED: Planner.make_plan() (族 D)
                      -> while (plan steps):
                           Executor.run() 或 ReWOO.run() (族 C/C')
                           if needs_replan && replans < MAX_REPLANS:
                             Replanner.revise() (族 D') -> 覆写 plan
                           else if needs_replan:  # cap
                             续跑旧 plan 下一步
                      -> LLM.synthesize() 合成最终答案
     }
  -> store.save(session)           # 每轮结束: 全量落盘
```

### ReWOO 子模式

Planner 判断某步是"独立步骤簇"时标记 `is_rewoo_cluster=True`。Agent 执行到该步时调用 ReWOO 而非普通 Executor:

1. **plan_dag**: LLM 出 DAG (`{"nodes":[{"id":"E1","tool":"...","args":{...},"deps":[]}]}`)
2. **execute_dag**: worker 按依赖序跑工具, 绑 `${E1}` 变量引用上游结果, 结果落 `memory.workspace`
3. **solve**: LLM 合成最终答案; 证据不足则升级 REPLANNING (C' -> D')

当 plan 所有步都是 ReWOO cluster 时, solver 已合成答案, Agent 跳过顶层 synthesize。

### FSM 状态

`IDLE -> ROUTING -> {RESPONDING | EXECUTING | PLANNING} -> ... -> IDLE`

关键转移: `EXECUTING -> REFLECTING -> EXECUTING` (重试), `EXECUTING/REFLECTING -> REPLANNING -> EXECUTING` (重规划), `REPLANNING -> RESPONDING` (abort 兜底)。

---

## Memory 召回时机与放置 (spec section 5.3)

笔试明确要求说明 memory 的召回时机与放置方式。

### 时机

| 时机 | 动作 |
|---|---|
| 每轮开始 | `Agent.chat` 入口 `store.load(session_id)` -> 拿回完整 Session (messages + memory + fsm_state), 跨进程重启也不丢 |
| 每次 Executor 构造请求 | `build_system_prompt(memory)` 从 memory 取当前 todos / plan / lessons, 注入 system prompt |
| 每轮结束 | `store.save(session)` -> 工具写过的 memory、Reflexion 产出的 lesson、FSM 新状态全部落盘 |

### 放置方式

Memory 是结构化字段 (`Memory` dataclass: `todos` / `plan` / `lessons` / `workspace`), **不混进 messages**。通过 `build_system_prompt()` 注入 system prompt 的三个段落:

```
You are a helpful agent.
【Todos】             <- memory.todos
- [#1] buy milk [PLANNED]
【Plan】              <- memory.plan
- step A | step B
【Lessons】           <- memory.lessons (Reflexion 产出)
- 上次 calculator 收到非数字参数会报错, 先校验类型
```

**为什么放 system prompt 而非 user msg**: memory 是稳定上下文, 不是用户当前指令。放 system 每轮自动生效, LLM 不会把它当一次性输入。

---

## 验收 Checklist (spec section 12)

| 项目 | 状态 | 说明 |
|---|---|---|
| 真实 DeepSeek API 跑通多轮对话 | [x] | e2e 实跑通过 (DIRECT/SIMPLE_TOOL/cross-turn memory, 4 轮, 2026-06-14) |
| Router 正确分流三种路径 | [x] | mock 测试覆盖 (test_router + test_agent) |
| Executor function-calling loop 跑通 (含 max_steps 截断) | [x] | mock 测试 (test_executor) |
| 工具异常触发 Reflexion 且 lesson 落 memory | [x] | mock 测试 (test_executor) |
| Reflexion 重试耗尽触发 REPLANNING, MAX_REPLANS 截断生效 | [x] | mock 测试 (test_agent + test_integration) |
| ReWOO: DAG 并行执行, 结果落 workspace, solver 合成 | [x] | mock 测试 (test_rewoo + test_agent) |
| ReWOO solver 证据不足升级 REPLANNING | [x] | mock 测试 (test_rewoo) |
| 跨轮次继续执行（建 todo → 追问读回） | [x] | **真 API e2e 实跑通过** |
| trace 完整记录每步, 前端可展示 | [x] | mock 测试 (test_trace + test_web) |
| ≥3 工具 + 最大步数 + 异常处理 | [x] | calculator/search/todo；MAX_STEPS=10；工具错误回喂 LLM 不崩 |
| 删除会话 | [x] | `DELETE /sessions/{id}` + 前端 × 按钮 |
| README 含运行/设计/memory 三段 | [x] | 本文件 |
| PROMPTS.md 含 prompt 与问题记录 | [x] | `PROMPTS.md` |
| 单测全绿 (mock LLM) | [x] | 104 passed |

**Status**: S1-S6 代码全部完成, mock-LLM 测试全绿 (104 passed)。真实 DeepSeek API e2e 实跑通过 (4 轮 DIRECT/SIMPLE_TOOL/cross-turn memory, 2026-06-14)。

---

## 测试

```bash
cd agent_framework
.venv/Scripts/python -m pytest -q    # 104 passed
.venv/Scripts/python -m ruff check .  # All checks passed
.venv/Scripts/python -m mypy          # Success: no issues found in 22 source files (strict)
```

- **108 个测试** 全绿 (含 S6 新增 7 个回归测试 `test_integration.py`: Contract C 四消息序列 + id 配对、跨路径 user 互斥、build_system_prompt 纯函数、reload 不变量、replan 覆写、MAX_REPLANS cap)。
- 所有 LLM 调用均 mock (FakeLLM / ScriptedExecutor / ScriptedReplanner), 不依赖真实 API。
- mypy strict 模式, 23 个源文件零错误。

---

## 项目结构

```
agent_framework/
  config.py            # 配置 (env 读取)
  main.py              # FastAPI app + Agent 装配
  llm/client.py        # DeepSeek (OpenAI 兼容) wrapper
  runtime/
    agent.py           # 顶层编排 (Router -> {DIRECT|SIMPLE_TOOL|PLAN_REQUIRED})
    router.py          # 族 I: 分类
    planner.py         # 族 D: 分解
    executor.py        # 族 C: function-calling 循环
    reflexion.py       # 族 E: 微观自纠
    replanner.py       # 族 D': 宏观自纠
    rewoo.py           # 族 C': 微观并行 DAG
    fsm.py             # 族 F: 状态机
  session/
    models.py          # Session / Memory / Message / Step / TodoItem
    store.py           # JSON 文件持久化 (原子替换)
  tools/
    base.py            # ToolRegistry + Tool 协议
    calculator.py      # 计算器
    search.py          # 搜索 (mock)
    todo.py            # 待办管理
  trace/logger.py      # JSONL 执行日志
  tests/               # 14 个测试文件, 104 passed
  static/index.html    # 前端 UI
docs/superpowers/      # spec + 实现计划
PROMPTS.md             # AI 辅助开发记录
```

实现计划与代码风格: `docs/superpowers/plans/` 与 `docs/superpowers/STYLE.md`。
