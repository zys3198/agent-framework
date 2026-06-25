from __future__ import annotations

import logging
import os
from pathlib import Path

from dotenv import load_dotenv

BASE_DIR = Path(__file__).resolve().parent
# Load .env (gitignored) next to config.py before reading env vars.
load_dotenv(BASE_DIR / ".env")

log = logging.getLogger(__name__)


def _int_env(name: str, default: int) -> int:
    raw = os.environ.get(name)
    if raw is None:
        return default
    try:
        return int(raw)
    except (TypeError, ValueError):
        log.warning("env %r=%r not an int, falling back to %d", name, raw, default)
        return default

DEEPSEEK_API_KEY = os.environ.get("DEEPSEEK_API_KEY", "")
DEEPSEEK_BASE_URL = os.environ.get("DEEPSEEK_BASE_URL", "https://api.deepseek.com")
MODEL = os.environ.get("MODEL", "deepseek-chat")

MAX_STEPS = _int_env("MAX_STEPS", 10)

SESSION_DIR = BASE_DIR / os.environ.get("SESSION_DIR", "sessions")
TRACE_DIR = BASE_DIR / os.environ.get("TRACE_DIR", "trace")

HOST = os.environ.get("HOST", "127.0.0.1")
PORT = _int_env("PORT", 8000)
