"""Load private prompts from secrets/materials or AGENT_ROOT after apply."""

from __future__ import annotations

import importlib.util
import os
import sys
from functools import lru_cache
from pathlib import Path

EXPORTS = (
    "ANSWER_PLAIN_TAIL",
    "ANSWER_STRUCTURED_TAIL",
    "CONVERSATION_STYLE",
    "JSON_MODE_NOTE",
    "SCHOOL21_CONTEXT",
    "build_answer_plain_prompt",
    "build_answer_system_prompt",
    "build_decompose_prompt",
    "build_expand_prompt",
    "build_resolve_query_prompt",
)


def agent_root() -> Path:
    return Path(os.getenv("AGENT_ROOT", Path(__file__).resolve().parents[2])).resolve()


def materials_dir() -> Path:
    raw = os.getenv("MATERIALS_AGENT_DIR")
    if raw:
        return Path(raw).resolve()
    return (agent_root().parent / "secrets" / "materials" / "agent").resolve()


def prompts_system_path() -> Path:
    candidates = (
        materials_dir() / "src" / "prompts" / "system.py",
        agent_root() / "src" / "prompts" / "system.py",
    )
    for path in candidates:
        if path.is_file():
            return path.resolve()
    raise ImportError(
        "Prompts not found. Run `bash scripts/apply_agent_materials.sh` or place "
        f"system.py at {candidates[1]}. Override via AGENT_ROOT / MATERIALS_AGENT_DIR."
    )


@lru_cache(maxsize=1)
def load_system_module():
    path = prompts_system_path()
    name = "src.prompts._system_impl"
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load prompts from {path}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod
