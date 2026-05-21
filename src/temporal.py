"""Текущая дата/время для промптов и date-aware поиска в Rocket.Chat."""

from __future__ import annotations

import os
from datetime import datetime
from functools import lru_cache
from zoneinfo import ZoneInfo

_WEEKDAYS_RU = (
    "понедельник",
    "вторник",
    "среда",
    "четверг",
    "пятница",
    "суббота",
    "воскресенье",
)

_MONTHS_RU = (
    "",
    "января",
    "февраля",
    "марта",
    "апреля",
    "мая",
    "июня",
    "июля",
    "августа",
    "сентября",
    "октября",
    "ноября",
    "декабря",
)


def agent_timezone() -> ZoneInfo:
    name = (os.getenv("S21_AGENT_TZ") or "Europe/Moscow").strip()
    try:
        return ZoneInfo(name)
    except Exception:
        return ZoneInfo("Europe/Moscow")


@lru_cache(maxsize=8)
def _now_cached(tz_name: str) -> datetime:
    return datetime.now(ZoneInfo(tz_name))


def now() -> datetime:
    tz = agent_timezone()
    return _now_cached(tz.key)


def rc_date_anchor() -> str:
    """Единственный механический якорь RC: фильтр по дате (не content-hardcode)."""
    return f"on:{now().strftime('%d/%m/%Y')}"


def temporal_context_block() -> str:
    """Блок для system prompt: «сегодня» для агента и пользователя."""
    dt = now()
    wd = _WEEKDAYS_RU[dt.weekday()]
    month = _MONTHS_RU[dt.month]
    iso = dt.strftime("%Y-%m-%d")
    dmy_dot = dt.strftime("%d.%m.%Y")
    dmy_slash = dt.strftime("%d/%m/%Y")
    dmy_short = dt.strftime("%d.%m")
    return (
        "## Сегодня (контекст сессии)\n"
        f"• Сейчас: **{wd}, {dt.day} {month} {dt.year}** ({iso}), часовой пояс {dt.tzinfo}\n"
        f"• Для Rocket.Chat: даты в постах — `{dmy_dot}`, `{dmy_short}`, `{dmy_slash}`, "
        f"`{dt.day} {month}`, команда `on:{dmy_slash}`\n"
        "• Оперативные вопросы про кампус **без другой даты** → по умолчанию **сегодня** "
        f"({dt.day} {month}); не выдумывай расписание вне EVIDENCE"
    )
