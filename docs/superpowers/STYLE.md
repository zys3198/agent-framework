# Code Style — agent-framework

权威。与 spec / plan 冲突时，实现以本文件为准。

## 工具链

- **Lint + Format**：ruff（代 black + flake8 + isort）。line-length=88。
- **类型检查**：mypy `strict`，CI gate（src 不过不让合）。tests 目录降级为 `warn_return_any` + 不强制 `disallow_untyped_defs`（mock 难严格）。
- **Docstring**：极简。仅模块级 + public class/函数 一行中文。内部函数不写。
- **依赖**：`ruff`、`mypy` 进 `[project.optional-dependencies].dev`。

## ruff 配置（写入 pyproject.toml）

```toml
[tool.ruff]
line-length = 88
target-version = "py312"

[tool.ruff.lint]
select = ["E", "F", "W", "I", "UP", "B", "SIM", "RUF"]
ignore = ["E501"]  # line-length 由 format 管

[tool.ruff.lint.isort]
known-first-party = ["config", "llm", "tools", "session", "trace", "runtime"]

[tool.ruff.format]
quote-style = "double"
```

## mypy 配置（写入 pyproject.toml）

```toml
[tool.mypy]
python_version = "3.12"
strict = true
files = ["config.py", "llm", "tools", "session", "trace", "runtime"]

[[tool.mypy.overrides]]
module = "tests.*"
strict = false
disallow_untyped_defs = false
warn_return_any = false
```

## 规则

### 类型（mypy strict 落地）
- 所有函数参数 + 返回值必须有类型注解。无 bare `def f(x):`。
- 禁 `Any`。确需类型擦除用显式 `Any` + 注释说明，或定义 Protocol。
- `session` 参数一律 `session: "Session"`（tools.run 签名）。
- `LLMClient.__init__(self, client: OpenAI, model: str)`——用真实 SDK 类型，注入测试用子类/协议。
- `from __future__ import annotations` 置文件首（延迟求值，避循环 import）。

### 命名
- snake_case 函数/变量；PascalCase 类；UPPER_SNAKE 常量。
- 私有前缀 `_`（`_eval`、`_emit`、`_now`）。

### async
- tools `run` / ToolRegistry.dispatch / agent `chat` / executor `run`：`async`。
- LLMClient 方法（respond/chat_with_tools/synthesize）：同步（openai 同步 client 简单）。agent 内 `await asyncio.to_thread(...)` 包同步 LLM 调用，免阻塞事件循环。

### 错误处理
- 工具内部异常 → 返回 `"ERROR: ..."` 字符串（喂回 LLM），不 raise 出 tool 边界。
- 系统级（LLM API 网络/鉴权、FS 权限）→ raise，由上层（agent/FastAPI）捕获。
- store 损坏 JSON → 备份 + 重建空 session，不 raise。

### 注释（继承 AGENTS.md）
- 默认不写注释。
- 只写 WHY：隐藏约束、不变量、特定 bug workaround、反直觉行为。
- 不写 WHAT（代码自解释）、不引用当前 task / 调用方。

### 编码（继承 AGENTS.md §9）
- 全文件 IO `encoding="utf-8"`。
- subprocess 传 `env={"PYTHONIOENCODING": "utf-8"}`。
- 路径用 `pathlib.Path`。

### dataclass vs pydantic
- 本层（models/store/config）优先 dataclass。
- FastAPI 请求/响应模型（S5）用 pydantic。

## 提交前检查

```bash
.venv/Scripts/python -m ruff check .
.venv/Scripts/python -m ruff format --check .
.venv/Scripts/python -m mypy
.venv/Scripts/python -m pytest -q
```
四条全过才 commit。
