# Claude Code 上下文管理机制（据文章整理）

> 来源：附件文章《面试官皱眉：你知道 Claude Code 的上下文窗口管理吗？》作者小林。
> 说明：以下源码常量、p99.99 数据、模型选择等均为**文章对源码的解读**，未亲自核对 Claude Code 仓库，仅供参考。拿去面试/写设计文档前建议 grep 核对关键符号。

## 一、核心哲学

上下文管理不是「省 token」，是「保信息结构」。不同信息半衰期不同 → **分通道管理**：语义信息进摘要、状态信息走附件、永久信息靠缓存重载、配置每次重建。

## 二、为什么需要主动管理（背景）

Agent 费窗口的三处叠加：

- 开局固定开支 5–10k token（system prompt + 工具描述 + CLAUDE.md）
- 工具调用**双倍记账**：`tool_use` + `tool_result` 都算 token
- 大文件 Read：单个源文件几千上万 token

扩窗口治标不治本：

- 费钱（长上下文推理贵）
- Attention 平方复杂度 → TTFT（首 token 延迟）升高
- **Lost in the Middle**：首尾记得清、中间模糊，窗口再大也无法解决

## 三、5 层压缩金字塔（从轻到重）

原则：**能不压就不压，必须压从最轻开始**。前 4 层有 3 层零 API 开销，极端情况才走到第 5 层。

| 层 | 机制 | 触发条件 | 成本 |
|---|---|---|---|
| 1 | 大结果存磁盘 | 单工具结果 > 50KB | 零 API |
| 2 | Snip 砍远古消息 | 对话头部无用 | 零 API（本地删 + 边界标记） |
| 3 | Micro-Compact 时间衰减 | 距上次 API 调用 > 60min | 零 API（清旧 tool_result，留最近 5 个） |
| 4 | Context Collapse 读时投影 | 90% 触发、95% 升级 | 零 API（写时不动、读时投影） |
| 5 | **Auto-Compact 全量摘要** | ~93%（上限 − 13k） | 一次 API（全文重写） |

Snip 释放的 token 数会传给第 5 层，避免重复压缩。

## 四、Auto-Compact（第 5 层）深度拆解

### 1. 触发时机 — 绝对阈值

```ts
export const AUTOCOMPACT_BUFFER_TOKENS = 13_000
// threshold = effectiveContextWindow - 13_000
```

- 不按轮数、不按百分比，按「距窗口上限固定缓冲」
- **据本文**：13k 基于摘要任务 **p99.99** 输出长度统计（实测约 17.3k + 冗余）
- 优势：窗口扩到 1M 也不膨胀，可预测

### 2. 手动 vs 自动

| | 手动 `/compact` | 自动 |
|---|---|---|
| customInstructions | 接受（指定保留重点） | 不接受 |
| suppressFollowUpQuestions | 关 | **开**（避免摘要里塞新问题打断任务） |
| circuit breaker | — | 连续失败 3 次停止重试（据本文：源于曾 1000+ 会话反复重试烧钱） |

### 3. 取舍逻辑 — 全量重写 + 四段式（核心反直觉点）

**所有**历史消息（含最近的）全送摘要器重写，不保留最近 N 条。理由：反正 Lost in the Middle 模型也看不清中间，不如全压成结构化精华。

压缩后消息链（`buildPostCompactMessages`）四段式：

```ts
export function buildPostCompactMessages(result: CompactionResult): Message[] {
  return [
    result.boundaryMarker,      // 边界标记：自动/手动 + 压前 token + 末条消息 ID
    ...result.summaryMessages,  // 摘要（前 200 轮全压这里）
    ...result.attachments,      // 附件：最近文件/计划/技能/异步任务状态 ← 状态恢复通道
    ...result.hookResults,      // hook 执行结果
  ]
}
```

**信息分类**：

- 语义信息（意图/方案/决策）→ 走摘要
- 状态信息（`a.py:42 有 bug`/子任务输出）→ 走附件，**原样恢复**

### 4. 预处理 microcompact

跑摘要前先把大工具结果（Read/Bash/Grep/Glob/WebFetch/WebSearch/Edit/Write）内容清空、留元数据占位符 → 对话瘦身 → 摘要器负担小、质量高。

它本身也是第 3 层独立机制，按 60min 衰减单独触发，不只服务于 Auto-Compact。

### 5. 文件恢复 — 三个常量

```ts
export const POST_COMPACT_MAX_TOKENS_PER_FILE  = 5_000
export const POST_COMPACT_TOKEN_BUDGET         = 50_000
export const POST_COMPACT_MAX_FILES_TO_RESTORE = 5
```

- 最多重载 5 个文件；每文件最多 5k；总预算 50k
- 按「最近活跃度（最近 Read 过）」排序挑选

### 6. 三类特殊信息不进摘要

| 信息 | 处理方式 |
|---|---|
| **CLAUDE.md** | 不注入；清空 `getUserContext` 缓存 → 下一轮自动从磁盘重载（永久存活，每轮重载） |
| **system prompt** | 不参与压缩；用 `buildEffectiveSystemPrompt` 重建，刷最新工具/权限/MCP 列表 |
| **异步任务状态** | 作为附件重新注入（保住子 agent 进度） |

### 7. 摘要 prompt 设计

- **防呆**：「只返回文本，禁止任何工具调用」前后各喊一遍（前后包夹，源于早期模型无视单次警告）
- **输出**：`<analysis>` 草稿区（最终剥离）+ `<summary>` 9 部分清单

**9 部分清单**：

1. Primary Request and Intent（主要请求和意图）
2. Key Technical Concepts（关键技术概念）
3. Files and Code Sections（涉及的文件和代码段）
4. Errors and fixes（错误和修复）
5. Problem Solving（解决的问题）
6. **All user messages（所有用户消息，枚举不落）**
7. Pending Tasks（待办）
8. **Current Work（当前工作，最细颗粒度，精确到文件+函数）**
9. Optional Next Step（下一步建议）

第 6 项（枚举所有用户消息，捕捉需求变更）和第 8 项（细粒度当前进度，让接续流畅）是设计灵魂。

- **模型选择**：据本文用**同一模型**（非便宜小模型），保质量 + 复用 Prompt Cache

### 8. 压完接续 — 五步流水线

1. 全消息送摘要器生成摘要
2. 清空缓存（`readFileState` / `loadedNestedMemoryPaths` / `getUserContext`）
3. **并发**生成附件（最近文件/异步任务状态/技能）
4. `buildPostCompactMessages` 组装新消息链
5. 新链替换旧链，用户只看到「Compacted」

接续细节：

- 摘要开头包装一句「本会话是从之前一次因上下文耗尽而中断的对话延续过来的」→ 让模型知道是**接力不是重头**
- 末尾带 transcript 文件路径 → 兜底查细节通道（开 Kairos 模式旧消息才落盘，否则一次性破坏性删除）
- 自动模式 `suppressFollowUpQuestions` 开，避免摘要里塞「请确认 A 还是 B」打断任务

## 五、常见方案的硬伤（对比铺垫）

| 方案 | 硬伤 |
|---|---|
| 滑动窗口 | 砍头部 → 丢全局指令，agent 失控 |
| 每 N 轮摘要 | 触发时机死板、粒度粗 |
| 向量召回历史 | 破坏时序、切断 tool_use/tool_result 成对、top-k 必漏关键决策 |

**Claude Code 不走「保留 + 召回」，走「重写整段对话」。**

## 六、一句话总结

**5 层金字塔（4 轻零开销兜底 1 重全量摘要）+ 绝对阈值（上限 − 13k）+ 全量重写四段式 + 信息分通道（语义进摘要 / 状态走附件 / 永久靠重载 / 配置每次重建）**。
