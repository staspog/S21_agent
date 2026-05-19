"""Декомпозиция пользовательского запроса в 1..N независимых подвопросов.

Используется structured output GigaChat: возвращает Pydantic-модель `SubqueryPlan`.
Если LLM вернул пусто — fallback: один подвопрос == исходный запрос.
"""

from __future__ import annotations

import logging

from langchain_gigachat.chat_models import GigaChat

from src.llm_gate import invoke_with_retry
from src.rc.schemas import SubqueryPlan
from src.pipeline.rag_faiss import rag_context_text

log = logging.getLogger("s21.pipeline.decompose")


_SYSTEM_PROMPT = (
    "Ты — планировщик информационного поиска для чат-бота Школы 21 (Сбер).\n"
    "Получив вопрос пользователя, разложи его на 1–5 независимых ПОДВОПРОСОВ.\n"
    "Каждый подвопрос:\n"
    "  • короткий (1 предложение), на русском языке;\n"
    "  • описывает один информационный intent (одно событие/факт/ссылку/дату);\n"
    "  • не дублирует другие подвопросы;\n"
    "  • сохраняет смысл оригинала (без додумывания).\n"
    "Если запрос простой и про одну вещь — верни один элемент.\n"
    "Если запрос сложный (\"что было ... и что будет ... и где найти ...\") — раздели "
    "его на смысловые части.\n"
    "ОТВЕЧАЙ только согласно схеме SubqueryPlan."
)


def decompose_question(
    *,
    llm: GigaChat,
    question: str,
    max_subqueries: int = 5,
    rag_chunks: list[dict] | None = None,
) -> list[str]:
    """LLM → 1..N подвопросов. Никогда не возвращает пустой список."""
    q = (question or "").strip()
    if not q:
        return []

    rag_text = rag_context_text(rag_chunks)

    structured = llm.with_structured_output(SubqueryPlan)
    try:
        result: SubqueryPlan = invoke_with_retry(
            structured.invoke,
            [
                ("system", _SYSTEM_PROMPT),
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
