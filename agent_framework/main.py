"""FastAPI web layer: chat route, session/trace readers, static UI."""

from __future__ import annotations

import json
import logging
import uuid
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

import config
from ctx.compactor import Compactor
from llm.client import LLMClient
from runtime.agent import Agent
from runtime.executor import Executor
from runtime.planner import Planner
from runtime.reflexion import Reflexion
from runtime.router import Router
from session.store import Store
from tools.base import ToolRegistry
from tools.calculator import Calculator
from tools.memory import ReadMemoryBody, WriteMemory
from tools.search import Search
from tools.todo import Todo

log = logging.getLogger(__name__)


class ChatRequest(BaseModel):
    session_id: str | None = None
    input: str


class ChatResponse(BaseModel):
    answer: str
    session_id: str


def build_agent() -> Agent:
    """Wire the full Agent from env. Raises RuntimeError if no API key."""
    registry = ToolRegistry()
    registry.register(Calculator())  # type: ignore[arg-type]
    registry.register(Search())  # type: ignore[arg-type]
    registry.register(Todo())  # type: ignore[arg-type]
    registry.register(WriteMemory())  # type: ignore[arg-type]
    registry.register(ReadMemoryBody())  # type: ignore[arg-type]
    llm = LLMClient.from_env(
        api_key=config.DEEPSEEK_API_KEY,
        base_url=config.DEEPSEEK_BASE_URL,
        model=config.MODEL,
    )
    store = Store(config.SESSION_DIR)
    spill_dir = config.SESSION_DIR.parent / "spill"
    compactor = Compactor(llm=llm, spill_dir=spill_dir)
    return Agent(
        store=store,
        router=Router(llm=llm),
        executor=Executor(
            llm=llm,
            registry=registry,
            reflexion=Reflexion(llm=llm),
            max_steps=config.MAX_STEPS,
        ),
        llm=llm,
        trace_dir=config.TRACE_DIR,
        planner=Planner(llm=llm),
        workspace_root=Path.cwd(),
        compactor=compactor,
    )


def _trace_path(trace_dir: Path, session_id: str) -> Path:
    """Resolve the jsonl trace file, guarding against path traversal.

    Only the basename of session_id is used; a .jsonl suffix is always appended.
    """
    safe_name = Path(session_id).name
    return trace_dir / f"{safe_name}.jsonl"


def _read_trace(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    out: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        try:
            out.append(json.loads(stripped))
        except json.JSONDecodeError:
            continue
    return out


def create_app(agent: Agent | None, store: Store, trace_dir: Path) -> FastAPI:
    """Build a FastAPI app wired to the given agent/store/trace_dir.

    If agent is None the /chat route returns 503 (no API key configured);
    read-only routes still work.
    """
    app = FastAPI(title="agent-framework")

    @app.exception_handler(Exception)
    async def global_exception_handler(request: Any, exc: Exception) -> Any:
        log.exception("unhandled exception: %s %s", request.method, request.url.path)
        try:
            from openai import APITimeoutError, OpenAIError
        except ImportError:
            APITimeoutError = None  # type: ignore[assignment,misc]
            OpenAIError = None  # type: ignore[assignment,misc]
        if APITimeoutError is not None and isinstance(exc, APITimeoutError):
            from fastapi.responses import JSONResponse
            return JSONResponse(status_code=503, content={"detail": "LLM request timed out"})
        if OpenAIError is not None and isinstance(exc, OpenAIError):
            from fastapi.responses import JSONResponse
            return JSONResponse(status_code=502, content={"detail": "LLM API error"})
        from fastapi.responses import JSONResponse
        return JSONResponse(status_code=500, content={"detail": "internal error"})


    static_dir = Path(__file__).resolve().parent / "static"
    if static_dir.is_dir():
        app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")

    @app.post("/chat", response_model=ChatResponse)
    async def chat(req: ChatRequest) -> ChatResponse:
        if agent is None:
            raise HTTPException(
                status_code=503, detail="DEEPSEEK_API_KEY not configured"
            )
        sid = req.session_id or _new_session_id()
        answer = await agent.chat(sid, req.input)
        return ChatResponse(answer=answer, session_id=sid)

    @app.get("/sessions")
    def list_sessions() -> list[dict[str, Any]]:
        return store.list()

    @app.get("/sessions/{session_id}")
    def get_session(session_id: str) -> dict[str, Any]:
        return store.load(session_id).to_dict()

    @app.get("/trace/{session_id}")
    def get_trace(session_id: str) -> list[dict[str, Any]]:
        return _read_trace(_trace_path(trace_dir, session_id))

    @app.delete("/sessions/{session_id}")
    def delete_session(session_id: str) -> dict[str, Any]:
        session_deleted = store.delete(session_id)
        trace_path = _trace_path(trace_dir, session_id)
        # Atomic: unlink raises FileNotFoundError if absent (no is_file TOCTOU),
        # PermissionError if TraceLogger still holds the handle for an in-flight
        # /chat (esp. Windows). Both are OSError subclasses.
        trace_deleted = False
        try:
            trace_path.unlink()
            trace_deleted = True
        except FileNotFoundError:
            pass
        except OSError as e:
            log.warning("trace delete failed for %s: %s", session_id, e)
        if not session_deleted and not trace_deleted:
            raise HTTPException(status_code=404, detail="session not found")
        return {"deleted": True, "session": session_deleted, "trace": trace_deleted}

    @app.get("/")
    def index() -> FileResponse:
        return FileResponse(str(static_dir / "index.html"))

    return app


def _new_session_id() -> str:
    return uuid.uuid4().hex


def _build_default_app() -> FastAPI:
    store = Store(config.SESSION_DIR)
    try:
        agent = build_agent()
    except RuntimeError:
        agent = None
    return create_app(agent, store, config.TRACE_DIR)


app = _build_default_app()


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("main:app", host=config.HOST, port=config.PORT, reload=False)
