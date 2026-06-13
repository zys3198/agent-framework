# Agent Framework — 实现计划

- **日期**: 2026-06-13
- **对应 spec**: `docs/superpowers/specs/2026-06-13-agent-framework-design.md`
- **状态**: Draft（待 review）
- **项目状态**: greenfield（仅 docs/，无代码）

---

## 0. 总策略

- **TDD 纪律**：每模块先写失败测试 → 实现 → 测试绿。LLM 调用全 mock，集成测试才打真 API。
- **分片顺序**：1 → 2 → 3 → 4 串行（runtime 依赖链）。5 可与 3/4 并行（web stub）。6 收尾。
- **每片交付**：代码 + 测试 + `ctx_shell pytest` 绿 + 该片 README 段。
- **commit 粒度**：一片 1-3 commit，conventional commits（`feat:`/`test:`/`docs:`）。
- **Phase A（REPLANNING）先于 Phase B（ReWOO）**：A 低成本高价值，B 依赖 A 兜底。

---

## 1. 切片清单

| # | 切片 | 模块 | 依赖 | 验收 |
|---|------|------|------|------|
| S1 | 基础设施 | config / llm / tools / session / store / trace / fsm 基线 | 无 | 工具单测 + fsm 合法转移 + store 读写 |
| S2 | runtime 核心 | router / planner / executor(Outcome) / reflexion / agent(DIRECT+SIMPLE_TOOL) | S1 | mock LLM 跑两条路径 |
| S3 | REPLANNING (Phase A) | fsm 加 REPLANNING / replanner / agent while-loop / reflexion retry_exhausted | S2 | replan 触发 + MAX_REPLANS 截断 |
| S4 | ReWOO (Phase B) | rewoo / memory.workspace / planner 簇检测 / agent rewoo 分支 | S3 | DAG 并行 + 变量绑定 + 证据不足升级 replan |
| S5 | Web | main.py FastAPI / static/index.html / trace endpoint | S2 | curl /chat 返答 + 前端跑通 |
| S6 | 集成 + 文档 | 跨轮次测试 / README / PROMPTS.md / demo | S3,S4,S5 | §5.4 四轮剧本真 API 跑通 |

---

## S1 — 基础设施

**文件**：
```
agent_framework/
├── config.py              # env 读取，所有常量
├── llm/client.py          # DeepSeek 封装 (openai SDK + base_url)
├── tools/base.py          # Tool Protocol + ToolRegistry
├── tools/calculator.py    # ast 白名单求值
├── tools/search.py        # mock 语料
├── tools/todo.py          # CRUD 写 session.memory.todos
├── session/models.py      # TodoItem/Memory/Message/Session dataclass
├── session/store.py       # JSON 持久化 (os.replace 原子写)
├── trace/logger.py        # jsonl 追加日志
└── runtime/fsm.py         # SessionFSM 基线状态 (含 REPLANNING 占位转移定义，S3 启用)
```

**关键契约**：
- `Tool.run(args, session) -> str`，`ToolRegistry.schemas() -> list[dict]`，`ToolRegistry.dispatch(call, session) -> str`
- `LLMClient.chat_with_tools(messages, tools) -> Resp(tool_calls, text)`，`respond()`，`synthesize()`
- `SessionFSM.transition(state)`：非法抛 `InvalidTransition`
- `Store.load(sid)/save(session)/list()`，utf-8 编码
- `Session` 含 `memory.workspace: dict`（S4 用，先建空字段）

**测试（test-first）**：
- `test_tools.py`：calculator 拒注入（`__import__`/`eval`）/ search 命中空 / todo create-list-update
- `test_fsm.py`：合法转移过 / 非法抛 InvalidTransition
- `test_store.py`：save→load 往返一致 / 损坏 JSON 备份重建 / list 返摘要

**验证命令**：
```bash
ctx_shell "cd agent_framework && python -m pytest tests/test_tools.py tests/test_fsm.py tests/test_store.py -q"
```

---

## S2 — runtime 核心

**文件**：
```
runtime/router.py      # Router.classify(input, memory) -> Route(Enum: DIRECT/SIMPLE_TOOL/PLAN_REQUIRED)
runtime/planner.py     # Planner.make_plan(input, memory) -> list[Step]
runtime/executor.py    # Executor.run(session, input, trace) -> Outcome(text, needs_replan)
runtime/reflexion.py   # Reflexion.reflect(call, error, memory) -> Lesson(text, reflexion_exhausted)
runtime/agent.py       # Agent.chat(session_id, input) -> str (DIRECT + SIMPLE_TOOL 两路径先通)
```

**关键契约**：
- `Outcome = dataclass(text: str, needs_replan: bool)`
- `Lesson = dataclass(text: str, reflexion_exhausted: bool)`（S3 用 exhaustion 标志）
- Agent 注入 memory 到 system prompt（spec §5.3 三段：todos / plan / lessons）
- Executor function-calling loop：无 tool_call 返 / max_steps 截断 / 异常触发 reflexion

**测试**：
- `test_router.py`：mock LLM 返 enum → 三路由分流
- `test_executor.py`：①无 tool_call 直接返 ②一轮 tool_call 后返 ③max_steps 截断 ④工具异常触发 reflexion
- `test_agent.py`：DIRECT 路径不传 tools / SIMPLE_TOOL 路径入 Executor

**验证命令**：
```bash
ctx_shell "cd agent_framework && python -m pytest tests/test_router.py tests/test_executor.py tests/test_agent.py -q"
```

---

## S3 — REPLANNING (Phase A)

**文件**：
```
runtime/fsm.py         # 启用 REPLANNING 状态 + 4 条新转移
runtime/replanner.py   # Replanner.revise(plan, results, memory) -> list[Step] (剩余修订)
runtime/agent.py       # PLAN_REQUIRED 分支: for -> while + replan
runtime/reflexion.py   # retry 计数达阈值 触发 reflexion_exhausted=True
config.py              # MAX_REPLANS=2
trace/logger.py        # log_replan(count), log_fsm(REPLANNING)
```

**关键契约**：
- 转移：`EXECUTING→REPLANNING`（replan_needed）/ `REFLECTING→REPLANNING`（retry_exhausted）/ `REPLANNING→EXECUTING`（plan_updated）/ `REPLANNING→RESPONDING`（abort）
- `Replanner.revise(plan, results, memory)`：已完成步保留，修订剩余；prompt 注入「原 plan + 已执行结果 + 失败信号」
- Agent while-loop：`outcome.needs_replan and replans < MAX_REPLANS` → transition REPLANNING → revise → continue
- 超限：用旧 plan 剩余步续跑，trace 标 `replan_capped`

**测试**：
- `test_fsm.py` 补：4 条 REPLANNING 转移合法 / 反向非法
- `test_replanner.py`：①retry_exhausted 触发 ②MAX_REPLANS 截断 ③修订后已完成步不重跑
- `test_agent.py` 补：replan 链路端到端（mock executor 先 needs_replan=True 后 False）

**验证命令**：
```bash
ctx_shell "cd agent_framework && python -m pytest tests/test_fsm.py tests/test_replanner.py tests/test_agent.py -q"
```

---

## S4 — ReWOO (Phase B)

**文件**：
```
runtime/rewoo.py       # ReWOO.run(session, step, trace) -> Outcome
                       #   plan_dag() -> {E1:tool_call, E2:...(dep E1)}
                       #   worker()   -> 绑变量入 memory.workspace
                       #   solver()   -> 合成 + evidence_sufficient 判定
runtime/planner.py     # make_plan 内检测独立簇，标 step.is_rewoo_cluster=True
trace/logger.py        # log_rewoo_dag(nodes,edges), log_rewoo_solve(vars,sufficient)
```

**关键契约**：
- DAG 节点 `E1/E2/...`，依赖用占位符引用前序输出
- worker 不走 LLM（直接 dispatch tool，绑变量），省 round-trip
- solver 一次 LLM 合成；判 `evidence_sufficient`，不足返 `Outcome(needs_replan=True)`（C′→D′ 兜底）
- 变量落 `memory.workspace`，跨步可读
- `REWOO_PARALLEL_ENABLED` 控独立节点并行（asyncio.gather）

**测试**：
- `test_rewoo.py`：①独立步打成 DAG ②变量绑定落 workspace ③solver 证据不足 → needs_replan=True
- `test_planner.py` 补：独立簇检测标 is_rewoo_cluster

**验证命令**：
```bash
ctx_shell "cd agent_framework && python -m pytest tests/test_rewoo.py tests/test_planner.py -q"
```

---

## S5 — Web

**文件**：
```
main.py                # FastAPI app: POST /chat, GET /sessions, GET /sessions/{id}, GET /trace/{id}, GET /
static/index.html      # vanilla JS: 输入框 + 对话区 + trace 折叠侧栏；session_id 存 localStorage
```

**关键契约**：
- `/chat` body `{session_id, input}` → `{answer, session_id}`
- session_id 首访 UUID 生成存 localStorage，复用（支撑跨轮次）
- trace endpoint 返 jsonl，前端折叠展示每步 tool_call/result/replan/rewoo

**验证**：
```bash
ctx_shell "cd agent_framework && uvicorn main:app --port 8000 &"
ctx_shell "curl -s -X POST localhost:8000/chat -d '{\"session_id\":\"t1\",\"input\":\"hi\"}'"
```
浏览器手测 DIRECT + SIMPLE_TOOL 两路径。

---

## S6 — 集成 + 文档

**任务**：
- `test_session_cross_turn.py`：spec §5.4 四轮剧本（轮1建 todo → 轮2 update → 轮3 list 读回 → 轮4 工具失败触发 reflexion+replan），断言 memory 跨轮持久
- 真 DeepSeek API 跑 §5.4 全剧本 + REPLANNING + ReWOO 触发场景，录屏
- `README.md`：运行方式 / 系统设计精简版 / memory 召回时机与放置（引 spec §5.3）/ REPLANNING+ReWOO 取舍说明
- `PROMPTS.md`：brainstorm + 本轮 REPLANNING/ReWOO 决策 prompt + 踩坑记录

**验证**：
```bash
ctx_shell "cd agent_framework && python -m pytest -q"              # 全绿
ctx_shell "cd agent_framework && DEEPSEEK_API_KEY=$KEY python -m pytest tests/test_session_cross_turn.py --real-api -q"
```

---

## 2. 风险与对策

| 风险 | 对策 |
|------|------|
| DeepSeek function calling 不守 schema | Executor 兜底：缺字段报错回喂 LLM；Router 误判时 LLM 仍可自主调工具 |
| REPLANNING 死循环 | `MAX_REPLANS=2` 硬截断 + trace `replan_capped` |
| ReWOO planner 押宝失败 | solver `evidence_sufficient=False` → 升级 REPLANNING |
| Windows 编码 | 全 IO `encoding=utf-8`，subprocess 加 `PYTHONIOENCODING=utf-8` |
| JSON 并发写 | `os.replace` 原子替换；同 session_id 请求排队（S5 加锁） |

---

## 3. 并行/串行决策

- S1→S2→S3→S4 **串行**：runtime 依赖链，不可乱序。
- S5 可与 S3/S4 **并行**（web 不依赖 replan/rewoo 内部，只要 S2 agent.chat 在）。
- S6 **最后**：依赖全部。
- 单 reviewer agent 跟 S2/S3（功能改动，中风险）；S4 加 adversarial 三 agent（ReWOO 逻辑复杂，高风险）。

---

## 4. 完成定义（DoD）

- [ ] S1-S6 全部切片测试绿（mock LLM）
- [ ] §5.4 四轮剧本真 API 跑通（含进程重启续跑）
- [ ] REPLANNING 触发 + MAX_REPLANS 截断演示
- [ ] ReWOO DAG 并行 + 证据不足升级 replan 演示
- [ ] trace 前端可展示（含 replan/rewoo 事件）
- [ ] README 三段（运行/设计/memory）+ PROMPTS.md 完整
- [ ] spec §12 checklist 全勾
