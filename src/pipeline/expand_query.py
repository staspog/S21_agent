"""Лемматизация и генерация коротких поисковых строк для chat.search.

`chat.search` Rocket.Chat работает по подстроке, поэтому пользовательский запрос
вида «А где состоится большой карьерный день и во сколько начало?» как searchText
бесполезен. Поэтому делаем два шага:

1) `pymorphy3` — приводит русские слова к нормальной форме (для подсказки LLM).
2) LLM (structured output) — генерит 2–4 коротких searchText на 1–3 слова,
   нацеленных на подстроку, которая, скорее всего, встречается в сообщении.
"""

from __future__ import annotations

import logging
import re
from functools import lru_cache

from langchain_gigachat.chat_models import GigaChat

from src.llm_gate import invoke_with_retry
from src.rc.schemas import SearchQueries

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


_SYSTEM_PROMPT = (
    "Ты составляешь поисковые строки для Rocket.Chat (поиск по подстроке).\n"
    "Тебе дан подвопрос и список лемм-подсказок.\n"
    "Верни 2–4 разных поисковых строки длиной 1–3 слова — таких, чтобы их "
    "подстрочное вхождение с высокой вероятностью попало в сообщение об "
    "интересующем событии/теме.\n"
    "Правила:\n"
    "  • без знаков препинания и кавычек;\n"
    "  • используй имена собственные/ключевые термины (карьерный день, GitVerse, "
    "продакт-менеджмент и т.п.);\n"
    "  • НЕ повторяй вариации одного слова в разных формах — пусть варианты "
    "перекрывают разные аспекты;\n"
    "  • избегай служебных слов (про, для, как, что, когда);\n"
    "  • разные варианты — разные ракурсы (название, дата/период, площадка/чат).\n"
    "Отвечай только согласно схеме SearchQueries."
)


def expand_query(
    *,
    llm: GigaChat,
    subquery: str,
    target: int = 3,
) -> list[str]:
    lemmas = lemmatize_keywords(subquery)
    prompt = (
        f"Подвопрос: {subquery}\n"
        f"Леммы-подсказки: {', '.join(lemmas) if lemmas else '—'}\n"
        f"Сгенерируй ровно {min(max(target, 2), 4)} поисковых строк."
    )
    structured = llm.with_structured_output(SearchQueries)
    try:
        sq: SearchQueries = invoke_with_retry(
            structured.invoke,
            [("system", _SYSTEM_PROMPT), ("human", prompt)],
            label="expand_query",
        )
        items = list(sq.queries or [])
    except Exception:
        log.exception("expand_query: structured_output failed, fallback к леммам")
        items = []

    if not items:
        # fallback: первые 1–2 значимых леммы
        items = lemmas[:2] or [subquery]

    items = items[: max(target, 2)]
    log.info("expand_query: subquery=%r → %s", subquery, items)
    return items
