"""Сборка search intent из истории диалога.

Rocket.Chat chat.search и RAG должны искать по смыслу всего разговора,
а не только по последней реплике.
"""

from __future__ import annotations

import logging
import re

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage
from langchain_gigachat.chat_models import GigaChat

from src.campus_intent import infer_search_scope
from src.llm_gate import invoke_with_retry
from src.pipeline.rag_faiss import rag_context_text
from src.prompts import build_resolve_query_prompt
from src.rc.schemas import SearchIntent

log = logging.getLogger("s21.pipeline.resolve_query")

_META_PAT = re.compile(
    r"(?:"
    r"поищи\s+(?:в\s+)?(?:рокете?|rocketchat|чат(?:ах|е)?)"
    r"|поиск\s+(?:в\s+)?(?:рокете?|чат(?:ах|е)?)"
    r"|(?:ок(?:ей)?|ладно|давай|хорошо)[,.\s]+(?:поищи|найди|search)"
    r"|search\s+in\s+(?:rocket|rocketchat|chat)"
    r")",
    re.IGNORECASE,
)


def _human_messages(messages: list[BaseMessage]) -> list[str]:
    out: list[str] = []
    for msg in messages:
        if isinstance(msg, HumanMessage):
            c = msg.content
            text = c if isinstance(c, str) else str(c)
            text = text.strip()
            if text:
                out.append(text)
    return out


def _format_dialog(messages: list[BaseMessage]) -> str:
    lines: list[str] = []
    for msg in messages:
        if isinstance(msg, HumanMessage):
            c = msg.content
            text = c if isinstance(c, str) else str(c)
            lines.append(f"Пользователь: {text.strip()}")
        elif isinstance(msg, AIMessage):
            c = msg.content
            text = c if isinstance(c, str) else str(c)
            snippet = text.strip().replace("\n", " ")
            if len(snippet) > 400:
                snippet = snippet[:397] + "…"
            lines.append(f"Ассистент: {snippet}")
    return "\n".join(lines)


def is_meta_search_command(text: str) -> bool:
    return bool(_META_PAT.search((text or "").strip()))


def _invoke_search_intent(
    *,
    llm: GigaChat,
    messages: list[BaseMessage],
    rag_chunks: list[dict] | None,
) -> SearchIntent | None:
    rag_text = rag_context_text(rag_chunks)
    dialog = _format_dialog(messages)
    structured = llm.with_structured_output(SearchIntent)
    try:
        return invoke_with_retry(
            structured.invoke,
            [
                ("system", build_resolve_query_prompt()),
                (
                    "human",
                    (
                        f"Диалог:\n{dialog}\n\n"
                        + (
                            "RAG_CONTEXT (фон по теме; не добавляй новых требований):\n"
                            f"{rag_text}\n\n"
                            if rag_text
                            else ""
                        )
                        + "Сформулируй search_query и scope."
                    ),
                ),
            ],
            label="resolve_query",
        )
    except Exception:
        log.exception("resolve_query: LLM failed")
        return None


def resolve_search_intent(
    *,
    llm: GigaChat,
    messages: list[BaseMessage],
    rag_chunks: list[dict] | None = None,
) -> tuple[str, str]:
    """История диалога → (search_query, scope) для RAG и Rocket.Chat."""
    humans = _human_messages(messages)
    if not humans:
        return "", "general"

    last = humans[-1]
    result = _invoke_search_intent(llm=llm, messages=messages, rag_chunks=rag_chunks)
    if result is not None:
        q = (result.search_query or "").strip()
        scope = result.scope or "general"
        if q:
            log.info(
                "resolve_query: turns=%s scope=%s meta_last=%s → %r",
                len(humans),
                scope,
                is_meta_search_command(last),
                q,
            )
            return q, scope

    parts = [h for h in humans if not is_meta_search_command(h)]
    if not parts:
        parts = humans[:-1] if is_meta_search_command(last) and len(humans) > 1 else humans
    fallback = " ".join(parts[-3:]).strip() or last
    scope = infer_search_scope(fallback)
    log.info("resolve_query: fallback scope=%s → %r", scope, fallback)
    return fallback, scope


def resolve_search_query(
    *,
    llm: GigaChat,
    messages: list[BaseMessage],
    rag_chunks: list[dict] | None = None,
) -> str:
    """История диалога → один search_query (scope отбрасывается)."""
    query, _scope = resolve_search_intent(
        llm=llm,
        messages=messages,
        rag_chunks=rag_chunks,
    )
    return query
