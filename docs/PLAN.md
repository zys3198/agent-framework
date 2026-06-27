# Agent Framework Memory 改造计划

四来源合并:面试官建议 + Claude Code 记忆系统文章 + Claude Code 上下文管理文章 + 子代理审查发现。经完整性审查(Hubble)与合理性审查修正。

## 目标

改进本框架 memory 召回(面试官作业),借鉴 Claude Code 真实机制(不做 RAG),顺带修地基缺陷、删死代码、补生产化债务。

## 2026-06-27 架构收深计划

本轮不新增功能，目标是收深已有 seam：
- `Executor.run()` 返回 `ExecutionResult`，让 tool-turn 结果和 message ownership 显式化。
- `runtime/memory_projector.py` 集中 memory -> prompt/index/attachments 的投影策略。
- `Store.with_session()` 提供同步 load->mutate->save 事务 seam；async chat 仍由 Agent 的 per-session asyncio lock 串行化。

不做 provider-neutral LLM 抽象；只有一个 DeepSeek/OpenAI-compatible Adapter，提前抽象收益不足。

## 阻塞性决策(已定)

- 路线:复刻 Claude Code 真实机制(文件式 memory + 渐进披露 + 中型模型筛选 + 分层压缩)。异步并行召回是核心上的增量,不是并列方案。
- 持续 ReAct:选方案 A -- 删 Replanner,保留 Planner。Planner 做初始分解(有价值),执行中失败靠 Executor 内部 ReAct + Reflexion 滚动,不靠 plan 后重规划。
- ReWOO:删(死路径,rewoo_cluster 永不触发)。
- Reflexion:保留机制(工具失败学 lesson = 执行中 ReAct 的一部分),去掉"plan 后自纠"定位。
- memory 内容来源:自动沉淀 + 手动写入两者都支持。
- 召回模型:留接口可换更小模型,先用 deepseek-chat。
- demo 规模裁剪:Phase 3 压缩砍到 3 层(大结果存盘 + microcompact 预处理 + Auto-Compact),Snip/Context Collapse/60 分钟衰减不做(单请求 demo 无意义)。
- Router 去留:保留 Router 做三分类,不废。面试官意见 2 的"小模型路由挪去召回"指不要用大模型自评难度路由。本框架保留一次轻量 LLM 做 DIRECT/SIMPLE_TOOL/PLAN_REQUIRED 硬分类(非"自评能否完成"),召回器(Phase 2b)独立筛 memory 条目。职责分开:Router 定路径,Recaller 选记忆。不做规则/关键词替换 Router(demo 收益小,Planner 还用同类 LLM,Router 一次调用 token 成本可接受)。Phase 0 只修 Router 死参数(注入 memory),不重构。

---

## Phase 0 -- 删死代码 + 修 P0 地基

一个 commit: fix: drop dead code + inject system prompt on all paths + schema上移

### 删死代码(决策 A)
- runtime/rewoo.py 整文件删(死路径 + 假并行 + 不写 messages 三问题随整体删除覆盖)
- runtime/replanner.py 整文件删(决策 A)
- runtime/planner.py:删 rewoo_cluster 检测(L70-71)
- session/models.py:删 Step.is_rewoo_cluster 字段
- runtime/agent.py:删 ReWOO/Replanner import + 构造参数 + PLAN_REQUIRED 分支里的 rewoo/replan 循环。保留 Planner.make_plan 注入 memory。简化为:plan 后逐步 Executor.run,失败不重规划,直接 synthesize。
- main.py:删 ReWOO/Replanner import + 装配
- config.py:删 MAX_REPLANS
- trace/logger.py:删 log_rewoo_dag / log_rewoo_solve / log_replan
- 测试:删 test_rewoo.py / test_replanner.py;test_agent.py 删 rewoo/replan 用例;test_models.py 删 is_rewoo_cluster 断言;test_planner.py 删 rewoo_cluster 用例;test_trace.py 删 rewoo 断言;test_integration.py 删 replan 用例

### 删 Replanner 的兜底(合理性审查第 2/8 点)
- plan 某步彻底失败时,该步 outcome(含 ERROR)记进 synthesize 输入,让模型知道"这步失败了",不静默糊弄。synthesize prompt 增加"标注失败步"字段。

### 修 P0 地基
- build_system_prompt 接到 SIMPLE_TOOL / PLAN_REQUIRED 路径(Plato 发现的断层) -- executor.run 开头拼 system
- 修多 tool_call 孤儿 bug:批内 exhausted 时补齐剩余 tool 消息再 return
- SIMPLE_TOOL 检查 needs_replan:outcome.needs_replan 时至少标注/告警(删了 Replanner 后不再有重规划路径,但不能静默吞)
- Router.classify + Planner.make_plan 注入 memory(死参数修复)

### function calling schema 加载上移(来源 1-1)
- 把 chat_with_tools 里 `{"type":"function","function":t}` 拼装逻辑挪到 ToolRegistry.schemas(),client 层接收标准格式不再拼装
- 几行改动,顺带做

---

## Phase 1 -- memory 数据结构 + 渐进披露

一个 commit: feat: progressive disclosure memory (write gates + index常驻 + content懒加载)

### 数据结构(session/models.py)
- 新增 MemoryEntry(id, type, name, description, keywords, content, saved_at)
- type 闭合四类型:user / feedback / project / reference
- feedback/project 强制结构:content 必含 Rule + Why + How to apply
- project 相对时间必须换算绝对日期

### 写入闸门(tools/memory.py 新增)
- write_memory 工具:校验 type 属于四类型;校验 feedback/project 结构;拒绝"能查到的不该存"(代码现状/目录/git log/bug 修法)
- 不该存校验用启发式(检查含文件路径/git 命令/能 grep 到的代码片段),不做 LLM 二次校验(合理性审查第 3 点:demo 不做重)
- read_memory_body(id):模型按需读 content(渐进披露第二阶段)
- 手动写入:开发者/用户可直接写文件(两者都支持)
- TDD 时序:write_memory 的"不该存"启发式(含文件路径/git 命令/grep 得到的代码片段就拒)是规则密集代码,先写失败测试断言各类拒绝场景(写代码片段被拒/写 feedback 无 Why 被拒/相对时间未换算被拒),再写实现。走 SP test-driven-development 流程。read_memory_body 懒加载简单,补单测即可。

### 写入时去重剪枝(即时去重,不做后台 Auto Dream 进程)
> 注:记忆系统文章的 Auto Dream 是后台周期进程(扫历史 JSONL 找信号再整理),本框架只在写入路径即时去重,机制不同,不沿用 Auto Dream 命名以免歧义。
- 合并重复(name/description 语义去重)
- 解决矛盾(同 type 同主题冲突时保留最新 + 标注)
- 相对时间到绝对时间

### 注入(runtime/agent.py build_system_prompt 重写)
- MEMORY.md 索引常驻:name + description + keywords(约 100 token/条)
- 200 行 / 25KB 双闸(哪个先到截断)
- 注入为 user message(system 之后),非系统提示词(软约束)
- content 不常驻,模型匹配到才 read_memory_body 读
- todo/plan 等状态信息走附件通道原样恢复 = 外化到代码层能召回(来源 1-4 措辞明确)

### AGENTS.md 四层级(按 session/project 分层)
- Project 级(./AGENTS.md) + Local 级(./AGENTS.local.md) + User 级(~/.agents/AGENTS.md)三层(demo 不做 Managed 组织级)
- 启动加载,向上遍历目录树拼接

### lessons 加 maxlen 滚动窗口
- memory.lessons 加 maxlen(如 20),超限丢最旧
- build_system_prompt 只注最近 N 条

---

## Phase 2 -- 异步化 + memory 召回

拆两步提交(合理性审查第 4 点:异步化工作量被低估):

### Phase 2a -- LLM client 异步化
一个 commit: refactor: AsyncOpenAI for true async
- client.py 三个方法(respond/chat_with_tools/synthesize)改 async,用 AsyncOpenAI
- 所有调用点 `asyncio.to_thread(self._llm.X)` 改 `await self._llm.X`
- 测试 mock 改 AsyncMock / 异步 fake
- 留召回模型接口(可换更小模型),先用 deepseek-chat
- 这一步不涉及新功能,纯重构,为 2b 的 gather 铺路

### Phase 2b -- memory 召回器
一个 commit: feat: async memory recall (中型模型筛选 + asyncio.gather)

按 Claude Code 记忆系统文章的协议:
1. 收集候选:读所有 MemoryEntry 的 name + description
2. 构造候选清单提示 + 用户 query
3. 中型模型结构化输出(严格 JSON,返回相关条目 id 列表,宁缺毋滥)
4. 两道过滤:本轮已注入不重复;工具避雷时机(正在调某工具时不选其"用法说明",保留"踩坑"类)

### 并行召回集成(runtime/executor.py)
- 第一轮 ReAct 正常跑
- asyncio.gather 起召回(与主模型第一轮并行)
- 第二轮 ReAct 前注入召回结果
- TDD 时序:两道过滤(去重 + 工具避雷)是规则密集代码,先写失败测试(同条不重复注入 / 调用 calculator 时其用法说明不选但已知缺陷保留),再写实现。走 SP test-driven-development。
- 注:当前 Executor.run 是 for step in range(max_steps) 闭环,跑到结束才 return。要支持"第一轮后注入召回再继续",需把 run 重构为可单步推进(拆 step() + 暴露中途注入点)。这是本 Phase 被低估的工作量,与 2a 同级。
- 调试预案:Executor 循环骨架重构易卡(碰消息序列原子性 = Contract C)。若 2-3 轮重构不成,按 AGENTS.md §4 止血,走 SP systematic-debugging(4 阶段根因:复现单步推进 bug / 检查 messages 序列断裂 / 对照 Contract C 不变量 / 定位注入点时机),不"再试试"硬写。

### 时效性标记
- 注入时附 saved_at + "This memory was saved N days ago. Verify before acting."
- 类型决定信任度:动机类不过期 / 位置类必须核对

---

## Phase 3 -- 分层压缩 + 生产化债务

一个 commit: feat: layered compaction (信息分通道,3层裁剪版) + prod hardening

### 分层压缩(ctx/compactor.py 新增) -- demo 裁剪到 3 层
- 第 1 层 大结果存盘:单工具结果超阈值(经验值 4KB)写磁盘,消息只留预览;原内容可再 Read 取回
- 第 2 层 microcompact 预处理:摘要前先清"可重新获取"的 tool result,只留最近 N 个(状态信息 todos/plan 走附件不裁)
- 第 3 层 Auto-Compact 全量摘要:绝对阈值(上限减缓冲,demo 用经验值 8k)触发,整段对话送摘要器重写
- 砍掉:Snip(单请求无远古探索)/ Context Collapse(实现复杂收益低)/ 60 分钟时间衰减(单请求无意义)
- TDD 时序:三个阈值触发(4KB 大结果存盘 / microcompact 最近 N / 8k Auto-Compact)是规则密集代码,先写失败测试(3.9KB 不触发 / 4.1KB 触发存盘 / 状态信息不进 microcompact 裁剪),再写实现。走 SP test-driven-development。

### 信息分通道(总纲)
- 语义信息(用户意图/方案/决策)到摘要
- 状态信息(todos/plan/workspace)到附件,原样恢复(= 面试官说的"外化召回")
- 永久上下文(AGENTS.md)缓存清理重载(不进摘要)
- 配置(system prompt/工具列表)每次重建
### 压缩后消息链组装(四段式裁剪版)
- boundaryMarker(边界标记):记录"自动触发 + 压前 token 数 + 末条消息 ID",配合摘要开头"本会话是从之前一次上下文耗尽而中断的对话延续过来的",让模型知道是接力不是重头
- summaryMessages(摘要):9 部分清单的压缩结果
- attachments(附件):todos/plan/workspace 原样恢复 = 外化召回落地
- hookResults:demo 无 hook,省略
- 注:原文四段(boundaryMarker + summary + attachments + hookResults),demo 砍 hookResults 留三段

### 摘要 prompt(9 部分固定章节)
1. Primary Request and Intent
2. Key Technical Concepts
3. Files and Code Sections
4. Errors and fixes
5. Problem Solving
6. All user messages(枚举不概括)
7. Pending Tasks
8. Current Work(最细粒度,到文件+函数名)
9. Optional Next Step
- 同模型做摘要(保质量 + 复用 prompt cache)
- 摘要开头:"本会话是从之前一次因上下文耗尽而中断的对话延续过来的"

### 安全网
- circuit breaker:连续 3 次摘要失败熔断
- 递归守卫:compact/session_memory 来源不触发压缩防死循环

### 文件恢复量化
- 最多重载 5 文件 / 每文件 5k token / 总预算 50k(按 deepseek-chat 窗口调)

### 生产化债务
- LLM client 容错(来源 4 缺失项):显式 timeout + max_retries + 指数退避(tenacity 或手写)
- Store 并发 lost update:per-session 锁或版本号
- config.py 非数字启动崩:try/except + 默认值
- token 用量记录(读 resp.usage) + 延迟日志(perf_counter 包 LLM 调用)
- 全局异常处理(main.py 加 exception_handler,openai 异常映射 HTTP 状态码)

---

## Phase 4 -- 验收

一个 commit: test: memory recall + compaction coverage + e2e + docs

- Phase 1/2/3 的 mock 测试(FakeLLM / AsyncMock)
- e2e 真实 DeepSeek API 回归(长对话场景,验证压缩 + 召回)
- README memory 段 + PROMPTS.md 更新
- README 补业界三方案缺点论证(来源 3-10):滑动窗口砍开局指令/摘要切碎依赖/向量召回乱时序,论证为何选压缩
- ruff + mypy strict + pytest 全绿
- SP requesting-code-review:对 Phase 2a 异步化 / 2b Executor 单步改造 / 3 压缩器(碰并发、消息序列原子性、memory 当 user message 注入)做一次中高风险审查。项目 plans/ 显示用过 SP,路径熟
- codex-security:security-diff-scan(可选):只扫新增 tools/memory.py(write_memory 落盘 + read_memory_body 读文件的路径,prompt injection 可能让模型写恶意路径/超大内容)。面试 demo 可省,答辩时"已考虑 injection 并做了路径校验"是加分项

---

## 执行顺序

Phase 0 到 1 到 2a 到 2b 到 3 到 4。每个 commit 提交前展示 git diff --cached --stat(1.3 人工确认线)。

Phase 0 是前置(工具路径不注入 system prompt,改 memory 白做)。Phase 2b 召回依赖 2a 异步化(gather 需要 AsyncOpenAI)。Phase 3 压缩依赖 Phase 2(召回结果也要被压缩管理)。
- 卡点回退:任一 Phase 连续 2-3 轮同方向打转,走 AGENTS.md §4(停 → git diff 定位 → restore/revert/reset → 根因重做),不硬补丁。Executor 循环改造(Phase 2b)最易卡,优先走 SP systematic-debugging。
