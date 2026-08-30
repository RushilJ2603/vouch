#!/usr/bin/env python3
"""Start the complete judge-facing Vouch experience on one local port.

The page and offline evidence need no keys. Provider calls happen only after
the presenter clicks Run. A local, gitignored .env is loaded without printing
or exposing any value to the browser.
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))


def load_local_env(path: Path) -> set[str]:
    """Load simple KEY=VALUE entries without overriding the caller's env."""
    loaded: set[str] = set()
    if not path.exists():
        return loaded
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        name, value = line.split("=", 1)
        name = name.strip()
        if not name or not name.replace("_", "").isalnum():
            continue
        value = value.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in "\"'":
            value = value[1:-1]
        if name not in os.environ:
            os.environ[name] = value
            loaded.add(name)
    return loaded


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run the Vouch judge surface")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8501)
    args = parser.parse_args(argv)

    load_local_env(ROOT / ".env")
    readiness = {
        "DeepSeek": bool(os.environ.get("DEEPSEEK_API_KEY")),
        "GLM": bool(os.environ.get("ZAI_API_KEY")),
    }
    provider_line = " · ".join(
        f"{name} {'ready' if ready else 'key missing'}"
        for name, ready in readiness.items()
    )
    print(f"Vouch judge surface: http://{args.host}:{args.port}")
    print(provider_line)
    print("Provider calls begin only when Run is clicked in the browser.")

    os.chdir(ROOT)
    import uvicorn

    uvicorn.run("dashboard.server:app", host=args.host, port=args.port, log_level="warning")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
