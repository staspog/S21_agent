"""Лемматизация и генерация коротких поисковых строк для chat.search.

Rocket.Chat `chat.search` понимает текстовый поиск и команды (`on:DD/MM/YYYY`).
Длинный вопрос пользователя как searchText бесполезен — LLM генерирует 2–4
коротких варианта по смыслу подвопроса; при scope=today добавляется только
механический `on:дата`.
"""

from __future__ import annotations

import logging
import re
from functools import lru_cache

from langchain_gigachat.chat_models import GigaChat

from src.campus_intent import (
    TODAY_DIGEST_SEARCH_ANCHORS,
    is_broad_campus_today,
)
from src.llm_gate import invoke_with_retry
from src.pipeline.rag_faiss import rag_context_text
from src.prompts import build_expand_prompt
from src.rc.schemas import SearchQueries
from src.temporal import rc_date_anchor

log = logging.getLogger("s21.pipeline.expand_query")


_WORD_PAT = re.compile(r"[А-Яа-яЁё]+|[A-Za-z]{2,}")


@lru_cache(maxsize=1)
def _morph():
    import pymorphy3

    return pymorphy3.MorphAnalyzer()


def lemmatize_keywords(text: str, max_keywords: int = 8) -> list[str]:
    """Достаём содержательные слова из текста и приводим к нормальной форме."""
    words = _WORD_PAT.findall((text or "").lower())
    morph = _morph()
    seen: set[str] = set()
    out: list[str] = []
    for w in words:
        if len(w) < 3:
            continue
        try:
            lemma = morph.parse(w)[0].normal_form
        except Exception:
            lemma = w
        if lemma in seen:
            continue
        seen.add(lemma)
        out.append(lemma)
        if len(out) >= max_keywords:
            break
    return out


def _merge_queries(primary: list[str], extra: list[str], *, limit: int) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for q in primary + extra:
        s = (q or "").strip()
        if not s:
            continue
        key = s.lower()
        if key in seen:
            continue
        seen.add(key)
        out.append(s)
        if len(out) >= limit:
            break
    return out


def expand_query(
    *,
    llm: GigaChat,
    subquery: str,
    target: int = 3,
    rag_chunks: list[dict] | None = None,
    intent_scope: str | None = None,
) -> list[str]:
    lemmas = lemmatize_keywords(subquery)
    limit = max(target, 2)
    rag_text = rag_context_text(rag_chunks)

    prompt = (
        f"Подвопрос: {subquery}\n"
        f"Леммы-подсказки: {', '.join(lemmas) if lemmas else '—'}"
    )
    if rag_text:
        prompt += (
            "\n\nRAG_CONTEXT (термины и синонимы Школы 21 — для queries, "
            "не добавляй новых фактов):\n"
            f"{rag_text}"
        )
    prompt += (
        f"\n\nСгенерируй scope и {min(max(target, 2), 4)} поисковых строк для Rocket.Chat."
    )
    structured = llm.with_structured_output(SearchQueries)
    scope = "general"
    items: list[str] = []
    try:
        sq: SearchQueries = invoke_with_retry(
            structured.invoke,
            [("system", build_expand_prompt()), ("human", prompt)],
            label="expand_query",
        )
        scope = sq.scope or "general"
        items = list(sq.queries or [])
    except Exception:
        log.exception("expand_query: structured_output failed, fallback к леммам")

    if intent_scope == "today":
        scope = "today"

    if not items:
        items = lemmas[:2] or [subquery]

    if scope == "today":
        anchors = [rc_date_anchor()]
        if is_broad_campus_today(subquery):
            anchors.extend(TODAY_DIGEST_SEARCH_ANCHORS)
        limit_today = min(8, max(limit, 4) + len(anchors))
        items = _merge_queries(anchors, items, limit=limit_today)
    else:
        items = items[:limit]

    log.info("expand_query: subquery=%r scope=%s → %s", subquery, scope, items)
    return items
