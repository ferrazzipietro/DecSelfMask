from __future__ import annotations

import os
from pathlib import Path


def _read_dotenv_file(path: Path) -> dict[str, str]:
    values: dict[str, str] = {}
    if not path.exists():
        return values
    for raw_line in path.read_text().splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        values[key.strip()] = value.strip().strip('"').strip("'")
    return values


def get_hf_token(explicit_token: str | None = None) -> str | None:
    if explicit_token:
        return explicit_token
    env_token = os.getenv("HF_TOKEN")
    if env_token:
        return env_token
    env_path = Path(".env")
    if env_path.exists():
        values = _read_dotenv_file(env_path)
        return values.get("HF_TOKEN")
    return None


def login_if_available(token: str | None = None) -> None:
    resolved = get_hf_token(token)
    if not resolved:
        return
    try:
        from huggingface_hub import login

        login(token=resolved)
    except Exception:
        return
