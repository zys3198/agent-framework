from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

BASE_DIR = Path(__file__).resolve().parent
# Load .env (gitignored) next to config.py before reading env vars.
load_dotenv(BASE_DIR / ".env")

DEEPSEEK_API_KEY = os.environ.get("DEEPSEEK_API_KEY", "")
DEEPSEEK_BASE_URL = os.environ.get("DEEPSEEK_BASE_URL", "https://api.deepseek.com")
MODEL = os.environ.get("MODEL", "deepseek-chat")

MAX_STEPS = int(os.environ.get("MAX_STEPS", "10"))

SESSION_DIR = BASE_DIR / os.environ.get("SESSION_DIR", "sessions")
TRACE_DIR = BASE_DIR / os.environ.get("TRACE_DIR", "trace")

HOST = os.environ.get("HOST", "127.0.0.1")
PORT = int(os.environ.get("PORT", "8000"))
