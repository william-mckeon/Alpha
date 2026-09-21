"""Allowlisted local credentials and recursive redaction for retained evidence."""
from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Any


CREDENTIAL_NAMES = ("OPENROUTER_API_KEY", "TAVILY_API_KEY", "ARCUS_PROXY_TOKEN")
SENSITIVE_FIELD = re.compile(r"authorization|api[_-]?key|tavilyapikey|access[_-]?token|proxy[_-]?token|cookie|password|secret", re.I)
KEY_PATTERN = re.compile(r"(?:tvly-[A-Za-z0-9_-]+|sk-or-v1-[A-Za-z0-9_-]+)")


def load_credentials(path: Path) -> None:
    """Load only known names, never override the process or evaluate shell syntax."""
    if not path.is_file():
        return
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        name, value = line.split("=", 1)
        name = name.strip()
        if name in CREDENTIAL_NAMES and not os.environ.get(name):
            value = value.strip().strip('"').strip("'")
            if value:
                os.environ[name] = value


def redact(value: Any, *, extra_secrets: tuple[str, ...] = ()) -> Any:
    """Redact both named fields and credentials embedded in free text/URLs."""
    if isinstance(value, dict):
        return {str(key): "[REDACTED]" if SENSITIVE_FIELD.search(str(key)) else redact(item, extra_secrets=extra_secrets) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [redact(item, extra_secrets=extra_secrets) for item in value]
    if not isinstance(value, str):
        return value
    for secret in (*extra_secrets, *(os.environ.get(name, "") for name in CREDENTIAL_NAMES)):
        if secret:
            value = value.replace(secret, "[REDACTED]")
    value = KEY_PATTERN.sub("[REDACTED]", value)
    value = re.sub(r"(?i)(bearer\s+)[^\s\"',;]+", r"\1[REDACTED]", value)
    return re.sub(r"(?i)((?:api[_-]?key|tavilyapikey|access_token)=)[^&#\s]+", r"\1[REDACTED]", value)
