# agent-framework

从零实现的最小可用 Agent（自实现 runtime，不依赖 LangChain/OpenHands）。

多轮对话 + 会话维护、基本循环（输入 → 判断直接答 / 调工具 → 执行 → 读结果 → 继续）、≥3 个工具、DeepSeek（OpenAI 兼容）API。

## 状态

- [x] **S1 基础设施**（config / llm / tools / session / store / trace / fsm）
- [ ] S2 runtime 核心（router / planner / executor / reflexion / agent）
- [ ] S3 REPLANNING（动态重规划）
- [ ] S4 ReWOO（计划 / 执行解耦 DAG）
- [ ] S5 Web（FastAPI + 前端）
- [ ] S6 集成 + 文档

## 开发

Python 3.12。代码与 venv 都在 `agent_framework/`：

```bash
cd agent_framework
python -m venv .venv
.venv/Scripts/python -m pip install -e ".[dev]"

# 提交前四件套（全绿才提交）
.venv/Scripts/python -m ruff check .
.venv/Scripts/python -m ruff format --check .
.venv/Scripts/python -m mypy
.venv/Scripts/python -m pytest -q
```

## 架构

分层 runtime + 会话级 FSM + Plan-and-Execute（宏观），内含 REPLANNING（动态重规划）与 ReWOO（微观计划/执行解耦）。详见 `docs/superpowers/specs/2026-06-13-agent-framework-design.md`。

实现计划与代码风格：`docs/superpowers/plans/` 与 `docs/superpowers/STYLE.md`。
