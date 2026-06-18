"""Декомпозиция пользовательского запроса в 1..N независимых подвопросов.

Используется structured output GigaChat: возвращает Pydantic-модель `SubqueryPlan`.
Если LLM вернул пусто — fallback: один подвопрос == исходный запрос.
"""

from __future__ import annotations

import logging

from langchain_gigachat.chat_models import GigaChat

from src.llm_gate import invoke_with_retry
from src.pipeline.rag_faiss import rag_context_text
from src.prompts import build_decompose_prompt
from src.rc.schemas import RagChunk, SubqueryPlan

log = logging.getLogger("s21.pipeline.decompose")


_COMPOUND_MARKERS = (
    " и ",
    " а также ",
    "?,",
    ";",
    "? и ",
    "? а ",
)


def _is_simple_question(question: str) -> bool:
    """Один intent — skip LLM decompose."""
    q = (question or "").strip()
    if not q or len(q) > 120:
        return False
    lower = q.lower()
    if any(m in lower for m in _COMPOUND_MARKERS):
        return False
    if q.count("?") > 1:
        return False
    return True


def decompose_question(
    *,
    llm: GigaChat,
    question: str,
    max_subqueries: int = 5,
    rag_chunks: list[RagChunk] | None = None,
) -> list[str]:
    """LLM → 1..N подвопросов. Никогда не возвращает пустой список."""
    q = (question or "").strip()
    if not q:
        return []

    if _is_simple_question(q):
        log.info("decompose: simple question fast-path → 1 subquery")
        return [q]

    rag_text = rag_context_text(rag_chunks)

    structured = llm.with_structured_output(SubqueryPlan)
    try:
        result: SubqueryPlan = invoke_with_retry(
            structured.invoke,
            [
                ("system", build_decompose_prompt()),
                (
                    "human",
                    (
                        f"Вопрос пользователя:\n{q}\n\n"
                        + (
                            "RAG_CONTEXT (справка по теме вопроса; используй как фон, "
                            "чтобы точнее называть сущности/термины/процессы Школы 21; "
                            "не добавляй новых требований):\n"
                            f"{rag_text}\n\n"
                            if rag_text
                            else ""
                        )
                    ),
                ),
            ],
            label="decompose",
        )
        items = list(result.subqueries or [])
    except Exception:
        log.exception("decompose: structured_output failed, fallback to single subquery")
        items = []

    if not items:
        items = [q]

    items = items[: max(1, max_subqueries)]
    log.info(
        "decompose: %s подвопросов: %s",
        len(items),
        " | ".join(items),
    )
    return items
