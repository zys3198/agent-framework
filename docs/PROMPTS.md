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


---

## 6. Memory 系统改造的 prompt 设计 (Phase 1-3)

### 6.1 write_memory 启发式闸门

`tools/memory.py` 的 `WriteMemory.run` 在写入前用三道纯启发式规则拦截"能查到的不该存"，不做 LLM 二次校验（PLAN.md Phase 1 决策：demo 不做重）。`_looks_like_code_or_path` 用 `_CODE_PATTERNS` 正则组匹配代码特征：```围栏块、`def/class/import/return` 语句、`for/while/if:` 控制流、`=>` 箭头、文件扩展名（`.py/.js/.ts/...`）、绝对路径（`[A-Za-z]:\` 或 `/`）、`git add/commit/push/...` 命令；命中任一即返回 `"content looks like code, file path, or git command"`。第二道闸门是结构校验：`feedback`/`project` 类型必须同时含 `Rule`/`Why`/`How to apply` 三标记（`_has_required_structure`）。第三道闸门是时间规范化：`project` 类型若命中 `_RELATIVE_TIME_MARKERS`（"今天/昨天/上周/next week..."）或 `_RELATIVE_TIME_PATTERNS`（"N days ago"/"in N days"/"next Monday"）即拒，强制绝对日期。

### 6.2 read_memory_body 渐进披露

`tools/memory.py` 的 `ReadMemoryBody` 是配套的"第二阶段懒加载"工具。索引常驻只暴露 `name`/`description`/`keywords`，模型判断某条相关后才调 `read_memory_body(id)` 按 `entry.id` 取回完整 `content`。`description` 明确为 `"Read the content of a memory entry by id."`，参数仅 `id`。这样避免所有条目正文常驻上下文，把 content 的 token 开销推迟到真正需要时（PLAN.md Phase 1：content 不常驻，模型匹配到才 read_memory_body 读）。

### 6.3 memory 索引注入

`runtime/agent.py` 的 `build_memory_context_message` 把 memory 索引拼成 **user message**（非 system prompt，软约束），放在 system 消息之后。`_memory_index_lines` 逐条格式化为 `- id=... type=... name=... description=... keywords=... saved_at=...`（约 100 token/条）。双闸截断：`_MEMORY_INDEX_MAX_LINES = 200` 行 与 `_MEMORY_INDEX_MAX_BYTES = 25 * 1024`（25KB），逐条累加字节，哪个先到就 `break`。返回值 `role="user"`，与 `claude_context`（若有）用空行拼接后整体作为一个 user 消息注入。状态信息（todos/plan/lessons）不进索引，而是 `build_system_prompt` 里作为 system prompt 硬约束拼入。

### 6.4 CLAUDE.md 四层加载

`runtime/claude_memory.py` 的 `load_claude_context(workspace_root, user_home)` 向上遍历目录树收集项目级 `CLAUDE.md`。逻辑：`[*reversed(resolved_root.parents), resolved_root]` 即从文件系统根逐级下到 workspace_root，每层读 `CLAUDE.md`，命中即 `_section("Project CLAUDE: {file}", text)` 入列（外层先、内层后，越靠近 workspace 越靠后覆盖）。之后再追加 `workspace_root/CLAUDE.local.md`（本地个人覆盖）与 `~/.claude/CLAUDE.md`（用户全局）。最终 `"

".join(sections)`。全部文件用 `encoding="utf-8"` 读取并 `.strip()`。四层 = 项目树多层 CLAUDE.md + CLAUDE.local.md + 用户全局 CLAUDE.md。

### 6.5 召回 prompt

`runtime/recaller.py` 的 `Recaller.recall(query, entries, current_tool)` 先把候选清单拼成 `id=... name=... description=...`（仅 name+description，不喂 content），再构造召回 prompt：

> Memory entries:
{candidates}

Query: {query}

Return strict JSON {"ids": [...]} with ids of relevant entries. Be conservative — only clearly relevant.

system 角色为 `"You are a memory recall filter."`，用中型模型（PLAN.md：留接口可换，先用 deepseek-chat）筛选。`_parse_ids` 用 `re.search(r"\{.*\}", text, re.DOTALL)` + `json.loads` 提取，解析失败返回空（宁缺毋滥）。随后 `filter_tool_usage` 纯本地过滤：当 `current_tool` 已知时，按 `_USAGE_KEYWORDS = ["用法", "how to use", "usage", "使用说明"]` 在 description 中匹配，剔除"用法/文档"类条目，保留 caveat/bug 类——即工具执行阶段不再召回泛用说明，只留陷阱警告。

### 6.6 压缩摘要 prompt

`ctx/compactor.py` 的 `_build_summary_prompt` 生成固定 9 章节摘要。`_SUMMARY_SECTIONS` 为硬编码列表：`Primary Request and Intent` -> `Key Technical Concepts` -> `Files and Code Sections` -> `Errors and fixes` -> `Problem Solving` -> `All user messages (enumerate, do not summarize)` -> `Pending Tasks` -> `Current Work (finest granularity: file + function name)` -> `Optional Next Step`，各章节拼成 `### {section}`。prompt 前言声明"本会话是从之前一次因上下文耗尽而中断的对话延续过来的"，附完整对话历史 + 用户消息枚举。输出三段链（`auto_compact`）：首段 `boundaryMarker`（`[COMPACT] session continuation...`，含压缩前 token 估算与 last message ref），中段 LLM 摘要，末段 `_build_attachments` 原样贴状态信息（Todos/Plan/Workspace）——语义内容走 LLM 摘要通道，状态信息走附件原样恢复通道，两者分道。安全网：`_RECURSION_MARKERS` 防重复压缩 + 每 session 熔断计数（默认 3 次失败即 trip）。
