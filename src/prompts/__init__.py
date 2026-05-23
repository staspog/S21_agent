"""Public API for agent prompts (implementation in secrets/materials/agent)."""

from __future__ import annotations

from src.prompts._load import EXPORTS, load_system_module
from src.temporal import temporal_context_block

_system = load_system_module()


def _with_temporal(text: str) -> str:
    block = temporal_context_block()
    if block in text:
        return text
    return f"{block}\n\n{text}"


def build_answer_system_prompt(*, rag_only: bool, json_mode: bool = False, rc_unavailable: bool = False) -> str:
    return _with_temporal(
        _system.build_answer_system_prompt(
            rag_only=rag_only,
            json_mode=json_mode,
            rc_unavailable=rc_unavailable,
        )
    )


def build_answer_plain_prompt(*, rag_only: bool, rc_unavailable: bool = False) -> str:
    return _with_temporal(
        _system.build_answer_plain_prompt(
            rag_only=rag_only,
            rc_unavailable=rc_unavailable,
        )
    )


def build_decompose_prompt() -> str:
    return _with_temporal(_system.build_decompose_prompt())


def build_resolve_query_prompt() -> str:
    return _with_temporal(_system.build_resolve_query_prompt())


def build_expand_prompt() -> str:
    return _with_temporal(_system.build_expand_prompt())


for _name in EXPORTS:
    if _name.startswith("build_"):
        continue
    globals()[_name] = getattr(_system, _name)


def school21_context_with_temporal() -> str:
    return _with_temporal(_system.SCHOOL21_CONTEXT)


__all__ = list(EXPORTS) + ["school21_context_with_temporal"]
