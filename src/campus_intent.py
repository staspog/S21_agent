"""Intent-хелперы для оперативных вопросов про кампус (дайджест ADM в RC)."""

from __future__ import annotations

import re

_BROAD_CAMPUS_TODAY = re.compile(
    r"(?:"
    r"что\s+(?:сегодня\s+)?(?:в\s+)?(?:кампусе|школе)"
    r"|что\s+сегодня\b"
    r"|сегодня\s+в\s+кампусе"
    r"|кампус\s+сегодня"
    r")",
    re.IGNORECASE,
)

_OPERATIONAL_CAMPUS = re.compile(
    r"(?:"
    r"\bсегодня\b"
    r"|кластер"
    r"|логистик"
    r"|что\s+(?:сегодня\s+)?(?:в\s+)?(?:кампусе|школе)"
    r"|где\s+поработать"
    r"|мероприят(?:ие|ия)\s+(?:сегодня|в\s+кампусе)"
    r"|кампус\s+сегодня"
    r")",
    re.IGNORECASE,
)

TODAY_DIGEST_SEARCH_ANCHORS: tuple[str, ...] = (
    "Что сегодня в кампусе",
    "Кампусная логистика",
    "Мероприятия в кампусе",
    "Клубная жизнь",
)


def is_broad_campus_today(text: str) -> bool:
    return bool(_BROAD_CAMPUS_TODAY.search(text or ""))


def is_operational_campus_question(text: str) -> bool:
    return bool(_OPERATIONAL_CAMPUS.search(text or ""))


def infer_search_scope(text: str) -> str:
    """Эвристика scope=today без LLM (fallback)."""
    if is_operational_campus_question(text) and re.search(
        r"\bсегодня\b", text or "", re.IGNORECASE
    ):
        return "today"
    if is_broad_campus_today(text):
        return "today"
    return "general"
