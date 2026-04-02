"""
Точка входа: запускайте из корня репозитория.

    python main.py

"""

from __future__ import annotations

import os

import uvicorn


def main() -> None:
    host = os.environ.get("HOST", "0.0.0.0")
    port = int(os.environ.get("PORT", "8000"))
    reload_env = os.environ.get("UVICORN_RELOAD", "1").lower()
    reload = reload_env not in ("0", "false", "no")
    uvicorn.run(
        "src.api:app",
        host=host,
        port=port,
        reload=reload,
    )


if __name__ == "__main__":
    main()
