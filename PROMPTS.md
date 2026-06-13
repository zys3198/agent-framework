# PROMPTS.md

本项目是面试笔试题"从零实现一个最小可用的 Agent"。本文记录 AI 辅助开发全过程: 设计阶段的 brainstorm prompt、关键技术决策、踩坑记录、开发流程。项目分 6 个 slice (S1-S6) 增量交付, 每个 slice 由子智能体独立实现 + 两阶段 review。

---

## 1. 设计阶段 prompt

### 1.1 初始 brainstorm

给 AI 的核心问题是: "实现一个最小可用的 Agent runtime, 不依赖 LangChain/OpenHands, 要有多轮对话、工具调用、会话持久化。请先做架构设计。"

AI 提出了两种路线:

- **minimal 路线**: 一个 while 循环 + 一个 LLM call + if-else 判断工具。轻量但难扩展, REPLANNING 和 ReWOO 往上叠的时候会改得面目全非。
- **分层路线**: 把 Agent 职责拆成若干"族" (function family), 每族一个模块, 通过组合拼出完整循环。重一点但每个模块边界清晰, 便于面试讲解和后续扩展。

### 1.2 关键设计问题: minimal vs 分层

这是整个设计阶段最大的决策点。AI 列出了分层的 5+2 族:

| 族 | 模块 | 职责 |
|---|---|---|
| I — Router | `router.py` | 分类输入: DIRECT / SIMPLE_TOOL / PLAN_REQUIRED |
| D — Planner | `planner.py` | 复杂任务分解为有序 step 列表 |
| C — Executor | `executor.py` | function-calling 循环 (LLM <-> tool <-> result) |
| E — Reflexion | `reflexion.py` | 工具失败时学 lesson, 微观自纠 |
| F — FSM | `fsm.py` | session 级状态机, 校验合法转移 |
| D' — Replanner | `replanner.py` | 宏观自纠: 修订剩余 plan |
| C' — ReWOO | `rewoo.py` | 微观并行: plan DAG -> worker -> solver |

### 1.3 决策: 用户选全烙分层

用户明确选择了分层方案。理由: 面试项目要能讲清楚每个模块为什么存在, 分层虽然代码量多一些但每个模块职责单一, REPLANNING (S3) 和 ReWOO (S4) 可以作为独立族平滑加入, 不用改已有代码。spec 文档在 `docs/superpowers/specs/2026-06-13-agent-framework-design.md`。

---

## 2. 关键技术决策

### 2.1 Contract C: executor 是 tool 路径上 session.messages 的唯一写者

这是 S2 实现时发现的最关键的 bug。最初 Agent 和 Executor 都往 `session.messages` 写消息, 导致跨轮次持久化后出现"孤儿 tool 消息" -- 一个 `role=tool` 的消息前面没有对应的 `assistant(tool_calls)` 消息。DeepSeek/OpenAI API 会直接返回 400 错误拒绝这种序列。

**决策**: 在 SIMPLE_TOOL 和 PLAN_REQUIRED 的每一步, Executor 是 `session.messages` 的唯一写者。它先把 user prompt 追加进去, 再构建 LLM message list, 把 `assistant(tool_calls)` 和 `tool(result)` 作为原子对一起写入。DIRECT 路径由 Agent 自己追加 user + assistant。PLAN_REQUIRED 的 Agent 只追加最终合成的 assistant 答案, 每步的中间消息由 executor 写。

这条契约直接消除了孤儿 tool 消息的可能性。`Message` dataclass 专门加了 `tool_calls` 和 `tool_call_id` 字段, 保证 reload 后配对关系不丢。

### 2.2 ReWOO 作为微观并行嵌在 PAE 内

ReWOO (Reasoning WithOut Observation) 不是独立的顶层路径, 而是 PLAN_REQUIRED 的一步。Planner 如果判断某步是"独立步骤簇", 会标记 `is_rewoo_cluster=True`, Agent 在执行到这一步时调用 ReWOO 而非普通 Executor。

ReWOO 内部三段: `_plan_dag` (LLM 出 DAG) -> `_execute_dag` (worker 并行跑工具, 绑 `${E1}` 变量) -> `_solve` (LLM 合成)。solver 判证据不足时升级 REPLANNING (C' -> D')。

### 2.3 REPLANNING cap 行为: 续跑旧 plan

spec §6 明确规定: Replan 超过 `MAX_REPLANS` (默认 2) 时, **停止重规划, 用旧 plan 的剩余步继续跑**, 而不是报错或终止。这保证了即使重规划无法收敛, Agent 也能给出一个兜底答案而不是卡死。

实现上, Agent 的 while 循环里 `if outcome.needs_replan and replans < self._max_replans` 才触发 replanner; 条件不满足时直接 `i += 1` 继续旧 plan 的下一步。

### 2.4 Memory 召回放 system prompt, 不混进 messages

spec §5.3: memory (todos / plan / lessons) 是稳定上下文, 不是用户当前指令。放在 system prompt 里每轮自动生效, LLM 不会把它当一次性输入。如果塞进 messages 作为 user 消息, 会污染对话历史且 LLM 可能把它当作需要回应的内容。

`build_system_prompt(memory)` 是纯函数, 把 todos / plan / lessons 拼成三段注入 base system prompt。Executor 每次构建 LLM 请求时调用它。

---

## 3. 踩坑记录

### 3.1 `.gitignore` 的 `trace/` 误伤源码包

S1 初始 `.gitignore` 写了 `trace/`, 意图忽略 trace 日志目录。但 Python 源码包也叫 `trace/` (`agent_framework/trace/`), 结果整个 trace 模块被 git 忽略了。修复: 把规则锚定到具体路径 `agent_framework/sessions/` 和 trace 日志的实际输出目录, 不用裸 `trace/`。

### 3.2 `pip install -e` 多顶层包问题

pyproject.toml 同时把 `agent_framework/` 和根目录 `docs/` 等暴露为包路径, 导致 `pip install -e .` 把多个顶层目录当 package 处理。修复: `py-modules=[]`, 明确只安装 `agent_framework` 一个包, 其余目录不参与安装。

### 3.3 S2 executor 多轮持久化 Critical bug (孤儿 tool)

详见 §2.1 Contract C。最初实现里 executor 在 tool error 早返回时没把 tool 消息写进 session.messages, reload 后 DeepSeek API 400。根因: 消息写入和早返回的顺序错了。修复: 在 `except` 块里、return 之前就把 tool message append 进去, 保证任何返回路径都不留孤儿。

### 3.4 spec §3.1 vs §5.1 `plan: list[str]` 类型矛盾

spec §3.1 写 `plan: list[str]` (plan 是字符串列表), §5.1 的 ReWOO 又需要 `is_rewoo_cluster` 标记。纯字符串没法带这个标记。S3 引入 `Step` dataclass (`prompt: str, is_rewoo_cluster: bool, done: bool`), `Memory.plan` 从 `list[str]` 改为 `list[Step]`。`Step.from_dict` 兼容旧 session 文件里的裸字符串 (自动 wrap)。

### 3.5 spec §3.1 `log_replan(replans)` 1-参 vs 实现 3-参

spec §3.1 写 `log_replan(replans)` 单参数, 但实际实现需要更多上下文: `log_replan(replans, reason, new_step_count)` 三参数。原因是 trace 日志要记录"为什么 replan"和"新 plan 多长", 单靠计数不够调试。实现按三参走, spec 没同步更新 -- 这是 spec 和代码漂移的一个典型例子。

### 3.6 ReWOO solver 已合成 -> agent 跳过顶层 synthesize

当 plan 的所有步都是 ReWOO cluster 时, 最后一步的 solver 已经合成了最终答案。如果 Agent 还去调顶层 `llm.synthesize`, 就等于二次合成, 既浪费 API 调用又可能产生不一致的答案。修复: Agent 检查 `all(s.is_rewoo_cluster for s in plan)`, 为真则直接取最后一步 outcome 的 text 作为答案, 跳过 synthesize。

---

## 4. 开发流程

### 4.1 subagent-driven: 每 task 一个子智能体

每个 slice (S1-S6) 拆成若干 task, 每个 task 由一个独立的子智能体 (subagent) 实现。子智能体拿到的是自包含的 task 描述 (文件路径、API 契约、验收标准), 不依赖主对话的上下文。这样:

- 每个 task 可以并行 (如果无依赖)。
- 主智能体负责协调和 review, 不写底层代码。
- 子智能体失败不影响其他 task。

### 4.2 两阶段 review

每个 task 实现后走两阶段 review:

1. **Standards review**: 代码是否遵循项目规范 (类型注解、编码风格、文件组织)。
2. **Spec review**: 实现是否匹配 spec / task 描述的验收标准。

两个 review 都通过才 merge。高风险改动 (auth / DB schema / 架构) 走三 agent adversarial review (找问题 / 求证 / 反驳, 2/3 通过)。

### 4.3 按风险 verify (CLAUDE.md §3)

不是所有改动都需要同等验证:

- 低风险 (机械重命名 / 格式) -> 不 verify, tsc + lint + test 过即可。
- 中风险 (功能改动) -> 单 reviewer agent。
- 高风险 (auth / DB / 架构 / 安全) -> 三 agent adversarial。

### 4.4 TDD

Bug 修复先写失败测试重现, 再改实现。新功能至少列验收 checklist 贴结果。S6 的回归测试 (`test_integration.py`) 正是这个流程的产物 -- 补上 unit test 漏掉的跨路径持久化、Contract C 端到端、reload 不变量等行为。

---

## 5. 工具栈

- **Python 3.12**, venv 隔离。
- **pytest + pytest-asyncio** (auto mode): 异步测试 bare `async def test_*`。
- **ruff**: lint + format (替代 black + flake8 + isort)。
- **mypy strict**: 23 个源文件零类型错误。
- **FastAPI + uvicorn**: Web 层。
- **openai SDK** (指向 DeepSeek base_url): LLM 调用。
- 不用 LangChain / OpenHands / AutoGen -- 这是面试要求。
