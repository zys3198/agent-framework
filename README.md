# agent-framework

从零实现的最小可用 Agent (自实现 runtime, 不依赖 LangChain / OpenHands)。多轮对话 + 会话持久化、基本循环 (输入 -> 判断直接答 / 调工具 -> 执行 -> 读结果 -> 继续)、>=3 个工具、DeepSeek (OpenAI 兼容) API、Reflexion 微观自纠、file-based memory 系统 (索引/正文懒加载/异步召回/三层 compaction)、FastAPI Web 层。

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

分层 runtime (Router/Planner/Executor/Reflexion 四族) + Plan-and-Execute + file-based memory 系统 (Phase 2-4 新增)。完整 spec: `docs/superpowers/specs/2026-06-13-agent-framework-design.md`（§5.5 详述 memory system）。

### 族分层 runtime（现状 4 族；原 ReWOO/Replanner/FSM 在 Phase 0 删除）

| 族 | 模块 | 职责 |
|---|---|---|
| I — Router | `runtime/router.py` | 一次轻量 LLM 调用, 分类输入为 DIRECT / SIMPLE_TOOL / PLAN_REQUIRED |
| D — Planner | `runtime/planner.py` | 复杂任务分解为有序 step 列表 (JSON 解析) |
| C — Executor | `runtime/executor.py` | function-calling 循环: LLM <-> tool dispatch <-> result 回填 |
| E — Reflexion | `runtime/reflexion.py` | 工具失败时学 lesson, 微观自纠 (同一步重试) |

> **状态字段（非族）**：`Session.fsm_state` 字符串常量（IDLE/PLANNING/EXECUTING/REFLECTING/WAITING），由 `runtime/agent.py` 内联设置，不再算独立族。
>
> **已删（Phase 0，commit `f4c9de9`/`fdc9728`）**：~~D' Replanner~~（宏观重规划，dead path）、~~C' ReWOO~~（并行 DAG，function-calling 单步已支持并行 tool_call）、~~独立 FSM 模块~~（降级为字符串常量）。失败步不再重规划，改交 `synthesize` 标 `[STEP i FAILED]`。

### Agent 主循环

```
Agent.chat(session_id, user_input)
  -> per-session asyncio.Lock       # 串行化 load->modify->save 事务
  -> store.load(session_id)         # 每轮开始: 拿回完整 Session
  -> Compactor.compact()            # Phase 3: 路由前先 compact (阈值下 no-op)
  -> load_project_context()         # Phase 2: AGENTS.md 三层聚合 (每轮 fresh)
  -> Router.classify()              # 族 I: 选路径
  -> {
       DIRECT:        LLM.respond() -> Agent 追加 user+assistant
       SIMPLE_TOOL:   Executor.run() (族 C, 持有 session.messages 写权 / Contract C)
       PLAN_REQUIRED: Planner.make_plan() (族 D)
                      -> for step in plan:        # Phase 0: 失败步不重规划
                           Executor.run() (族 C+E, 含异步 memory 召回)
                           needs_replan 仅记 trace
                      -> LLM.synthesize() 合成最终答案 (失败步标 [STEP i FAILED])
     }
  -> store.save(session)           # 每轮结束: 全量落盘
```

### ~~ReWOO 子模式~~（已删除，Phase 0）

> 初版实现过的微观并行子模式，Phase 0 证实为 dead path（实际任务不触发 + function-calling 单步已支持并行 tool_call）后删除。原设计如下，仅留作历史记录：

1. **plan_dag**: LLM 出 DAG（worker 按依赖序跑工具，绑 `${E1}` 变量）
2. **solve**: LLM 合成最终答案

**现状**：`Step` 仅 `prompt` 字段；`memory.workspace` 字段保留但未用；失败步由 `synthesize` 标注。

### 状态字段（FSM 模块已删，降级字符串常量）

`IDLE -> ROUTING -> {RESPONDING | EXECUTING | PLANNING} -> ... -> IDLE`

> 独立 FSM 模块已删除（Phase 0 `fdc9728`）。`fsm_state` 是 `Session` 的字符串字段，由 Agent 内联设置，不再强制合法转移。Reflexion 攒满 3 条 lesson 判穷尽 → 返回 `needs_replan`（不再走 REPLANNING 链路，因 Replanner 已删）。

关键转移: `EXECUTING -> REFLECTING -> EXECUTING` (Reflexion 微观重试)。原 `-> REPLANNING -> EXECUTING` 链路因 Replanner 删除已废弃。

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

## Memory System

This framework implements a Claude Code-style memory system: **file-based memory with progressive disclosure, async recall, and layered compaction** — not vector RAG. The overhaul is tracked in `docs/PLAN.md` (Phases 0-3 complete; this section is the Phase 4 doc deliverable).

### File-based memory (index always loaded, content lazy)

Memory entries live on the `Session` (`session/models.py`), not in an embedding store. Each `MemoryEntry` has an `id`, `type`, `name`, `description`, `keywords`, `content`, and `saved_at`. The full index is built every turn by `build_memory_context_message()` (`runtime/agent.py`) and injected as a **user message** (soft constraint, not system prompt). Only `name`/`description`/`keywords`/`saved_at` are in the index — roughly 100 tokens per entry. The full `content` is **not** loaded by default; the model pulls it on demand via the `read_memory_body` tool (`tools/memory.py`).

Two hard caps on the index (whichever hits first): `_MEMORY_INDEX_MAX_LINES = 200` lines, `_MEMORY_INDEX_MAX_BYTES = 25 * 1024` bytes.

### Four memory types

`MEMORY_ENTRY_TYPES = ("user", "feedback", "project", "reference")` (`session/models.py`):
- **user** — user preferences, motivations, environment.
- **feedback** — corrections/lessons; `content` must contain the markers `Rule`, `Why`, `How to apply`.
- **project** — project facts/decisions; same `Rule`/`Why`/`How to apply` structure, and relative time phrases must be normalized to absolute dates.
- **reference** — lookup material.

### Write gates (`write_memory`, `tools/memory.py`)

`write_memory` rejects entries that are "findable" rather than worth remembering. The heuristic gate (`_looks_like_code_or_path`) checks for fenced code blocks, assignment/`def`/`class`/`import` statements, file paths (with extensions or drive letters), and git subcommands (`git add|commit|push|...`). It also enforces: valid `type`, non-empty `name`/`content`, `keywords` as a list of strings, the `Rule`/`Why`/`How to apply` structure for `feedback`/`project`, and absolute dates for `project` (relative-time markers like "yesterday"/"上周"/"2 days ago" are rejected).

### Dedup + conflict resolution on write

`_find_existing_entry` matches on `(type, name)` (case-insensitive). If a match exists, the entry is **updated in place** (name/description/keywords/content/saved_at overwritten, same `id`); otherwise a new entry with `_next_id` is appended. This gives same-name-same-type overwrite (latest wins) and cross-name coexistence.

### AGENTS.md loader (`runtime/agent_memory.py`)

`load_project_context(workspace_root, user_home)` assembles permanent context from up to three layers, in this order:
1. **Project AGENTS.md** — traverses `workspace_root` up its parent chain, reading `AGENTS.md` at each level (innermost wins by appearing last in the joined output).
2. **Local AGENTS.local.md** — `workspace_root/AGENTS.local.md` (git-ignored personal overrides).
3. **User AGENTS.md** — `~/.agents/AGENTS.md`.

The merged text is loaded at the start of each `Agent.chat` and threaded through `build_memory_context_message` / `Planner.make_plan` / `Executor.run` as the `project_context` argument. Because it is re-read every turn, it is **not** folded into compaction summaries — it is rebuilt fresh (see "Information channeling" below).

### Async memory recall (`runtime/recaller.py`, wired in `runtime/executor.py`)

`Recaller.recall(query, entries, current_tool)` uses a medium LLM to select relevant entry ids by `name`+`description` only (progressive disclosure — the model never sees `content` during filtering). The LLM returns strict JSON `{"ids": [...]}` parsed by `_parse_ids`; it is instructed to be conservative ("only clearly relevant").

Integration in `Executor.run`:
- At **step 0**, if a recaller is configured and entries exist, recall is kicked off as an `asyncio.Task` so it runs **in parallel** with the main model's first ReAct step.
- After step 0, the task is awaited and two local filters are applied (no second LLM round-trip):
  1. **Dedup** — ids already present in the injected memory index are dropped.
  2. **Tool-avoidance** (`filter_tool_usage`) — once the LLM-chosen tool name is known, exclude entries whose `description` contains usage keywords (`how to use`, `usage`, `使用说明`, `用法`), while **keeping** caveat/bug entries. The rationale: don't distract the model with a tool's manual while it is actively calling that tool, but do keep known gotchas.
- Surviving ids are injected as a `Recalled from memory:` user message, each line annotated with `saved_at` and a staleness reminder.

### Layered compaction (`ctx/compactor.py`, demo-trimmed to 3 layers)

`Compactor.compact(session)` runs at the start of every `Agent.chat`, before routing. Layers are no-ops below their thresholds, so the common path is cheap.

- **Layer 1 — large-result spillover** (`spill_large_results`): tool results larger than `large_result_bytes` (default 4096) are written to disk under `spill_dir` (sha256 prefix), and the message is replaced with an 80-char preview + a `[spill:<digest>]` marker reclaimable via Read.
- **Layer 2 — microcompact** (`microcompact`): keeps only the most recent `microcompact_keep` (default 5) tool results, dropping older `tool` messages and their preceding `assistant(tool_calls)` messages. State info (`todos`/`plan`/`workspace`) lives in `session.memory` and is **never touched** by this layer.
- **Layer 3 — Auto-Compact** (`auto_compact`): when estimated tokens (`_estimate_tokens`, ~4 bytes/token, no tokenizer dependency) exceed `auto_compact_tokens` (default 8000), the whole conversation is sent to the LLM summarizer. The output replaces the message list with a **3-segment chain**:
  1. `boundaryMarker` — `[COMPACT] session continuation...` with pre-compaction token count and last message ref, so the model knows this is a handoff, not a fresh start.
  2. `summary` — a fixed 9-section summary (Primary Request and Intent; Key Technical Concepts; Files and Code Sections; Errors and fixes; Problem Solving; All user messages enumerated; Pending Tasks; Current Work at file+function granularity; Optional Next Step).
  3. `attachments` — `todos`/`plan`/`workspace` restored **verbatim** (the "externalized recall" of state info).

### Information channeling

Compaction routes information by kind rather than treating all text uniformly:
- **Semantic info** (user intent, decisions, problem-solving) → goes into the **summary**.
- **State info** (todos/plan/workspace) → goes into **attachments**, restored verbatim — never summarized, never dropped by microcompact.
- **Permanent context** (AGENTS.md) → **reloaded fresh** each turn from disk, not stored in the summary.
- **Config** (system prompt, tool list) → rebuilt every request.

### Safety net

- **Circuit breaker**: `auto_compact` tracks consecutive summary failures; after `circuit_breaker_limit` (default 3) it trips `circuit_tripped = True` and refuses further compaction rather than looping on a broken summarizer.
- **Recursion guard**: `_is_compaction_output` inspects the first 3 messages for markers (`[COMPACT]`, `session continuation`, `interrupted context`); if the conversation already looks like a compaction output, it is not re-compacted (prevents infinite compact-the-compaction loops).

### Why compaction over alternatives

Three industry alternatives were considered and rejected (`docs/PLAN.md`, Phase 4 deliverable):

- **Sliding window** (drop oldest messages past N): simple, but it drops early system instructions and task framing, losing the original intent as the conversation grows.
- **Pure summarization** (summarize the whole history into one blob): loses fine-grained detail and chops dependencies — e.g. a `tool` result that a later step depends on gets flattened into prose, breaking the assistant(tool_calls) → tool(result) pairing the OpenAI API requires.
- **Vector RAG recall** (embed history, retrieve top-k): breaks temporal ordering — retrieval returns by similarity, not by when things happened — and adds retrieval noise (irrelevant-but-similar chunks) and an embedding-store dependency.

**Why layered compaction was chosen**: it preserves state info verbatim (todos/plan/workspace survive in attachments, so nothing the model needs to act on is lost to a summary), keeps semantic info in a structured summary (compressible without losing the narrative), and has no retrieval-ordering problem (the message chain stays sequential; nothing is re-ranked). The tradeoff accepted is summarizer LLM cost on layer-3 triggers — bounded by the circuit breaker and the 8k-token threshold.

---

## 验收 Checklist (spec section 12)

| 项目 | 状态 | 说明 |
|---|---|---|
| 真实 DeepSeek API 跑通多轮对话 | [x] | e2e 实跑通过 (DIRECT/SIMPLE_TOOL/cross-turn memory, 4 轮, 2026-06-14) |
| Router 正确分流三种路径 | [x] | mock 测试覆盖 (test_router + test_agent) |
| Executor function-calling loop 跑通 (含 max_steps 截断) | [x] | mock 测试 (test_executor) |
| 工具异常触发 Reflexion 且 lesson 落 memory | [x] | mock 测试 (test_executor) |
| Reflexion 穷尽返回 needs_replan，交 synthesize 标 [STEP i FAILED] | [x] | mock 测试 (test_executor + test_integration)；Phase 0：Replanner/MAX_REPLANS 已删 |
| ~~ReWOO: DAG 并行执行~~ / ~~solver 升级 REPLANNING~~ | [-] | 已删 (Phase 0)，mock 测试 (test_rewoo) 已随代码移除 |
| 跨轮次继续执行（建 todo → 追问读回） | [x] | **真 API e2e 实跑通过** |
| trace 完整记录每步, 前端可展示 | [x] | mock 测试 (test_trace + test_web) |
| ≥3 工具 + 最大步数 + 异常处理 | [x] | calculator/search/todo；MAX_STEPS=10；工具错误回喂 LLM 不崩 |
| 删除会话 | [x] | `DELETE /sessions/{id}` + 前端 × 按钮 |
| README 含运行/设计/memory 三段 | [x] | 本文件 |
| PROMPTS.md 含 prompt 与问题记录 | [x] | `PROMPTS.md` |
| 单测全绿 (mock LLM) | [x] | 155 passed |

**Status**: S1-S6 + Phase 2-4 (Memory System) 代码全部完成, mock-LLM 测试全绿 (155 passed)。真实 DeepSeek API e2e 实跑通过 (4 轮 DIRECT/SIMPLE_TOOL/cross-turn memory, 2026-06-14)。Phase 0 已删 ReWOO/Replanner/FSM。

---

## 测试

```bash
cd agent_framework
.venv/Scripts/python -m pytest -q    # 155 passed
.venv/Scripts/python -m ruff check .  # All checks passed
.venv/Scripts/python -m mypy          # Success: no issues found in 22 source files (strict)
```

- **155 个测试** 全绿 (含 `test_integration.py` 回归: Contract C 四消息序列 + id 配对、跨路径 user 互斥、build_system_prompt 纯函数、reload 不变量)。
- 所有 LLM 调用均 mock (FakeLLM / ScriptedExecutor), 不依赖真实 API。
- mypy strict 模式, 22 个源文件零错误。

---

## 项目结构

```
agent_framework/
  config.py            # 配置 (env 读取)
  main.py              # FastAPI app + Agent 装配
  llm/client.py        # DeepSeek (OpenAI 兼容) wrapper
  ctx/compactor.py     # Phase 3: 三层 compaction (spill/microcompact/auto_compact)
  runtime/
    agent.py           # 顶层编排 (Router -> {DIRECT|SIMPLE_TOOL|PLAN_REQUIRED}) + per-session lock
    router.py          # 族 I: 分类
    planner.py         # 族 D: 分解
    executor.py        # 族 C: function-calling 循环 (含 Contract C + 异步召回)
    reflexion.py       # 族 E: 微观自纠
    recaller.py        # Phase 2: 异步 memory 召回 + 工具规避
    agent_memory.py    # Phase 2: load_project_context 三层 AGENTS.md 聚合
    # 已删 (Phase 0): rewoo.py / replanner.py / fsm.py
  session/
    models.py          # Session / Memory / Message / Step / TodoItem / MemoryEntry
    store.py           # JSON 文件持久化 (原子替换)
  tools/
    base.py            # ToolRegistry + Tool 协议
    calculator.py      # 计算器
    search.py          # 搜索 (mock)
    todo.py            # 待办管理
    memory.py          # Phase 2: WriteMemory (门控) + ReadMemoryBody (懒读正文)
  trace/logger.py      # JSONL 执行日志
  tests/               # 17 个测试文件, 155 passed
  static/index.html    # 前端 UI
docs/superpowers/      # spec + 实现计划
PROMPTS.md             # AI 辅助开发记录
```

实现计划与代码风格: `docs/superpowers/plans/` 与 `docs/superpowers/STYLE.md`。
