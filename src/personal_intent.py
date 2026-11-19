"""Conservative routing and local-day bounds for personal data requests."""

from __future__ import annotations

import re
from datetime import datetime, time, timedelta
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

_CAMPUS = re.compile(r"\b(?:кампус\w*|школ\w*)\b", re.IGNORECASE)
_PERSONAL = re.compile(r"\b(?:мой|мои|мо[яеюи]|мне|меня|моих|моему|мной)\b", re.IGNORECASE)
_CALENDAR = re.compile(r"\b(?:календар\w*|расписан\w*|провер\w*|защит\w*|слот\w*|событи\w*|встреч\w*)\b", re.IGNORECASE)
_PROJECT = re.compile(r"\bпроект\w*\b", re.IGNORECASE)
_PROFILE = re.compile(r"\b(?:уров\w*|опыт\w*|xp|балл\w*|прп|крп|prp|crp|коин\w*|монет\w*|профил\w*)\b", re.IGNORECASE)


def personal_kind(question: str) -> str | None:
    # A short balance/level question often omits "my". If the model fails to
    # select a read, fail closed instead of answering from generic RAG data.
    if re.search(r"\b(?:какой|каков|сколько)\b.{0,30}\b(?:уров\w*|балл\w*|опыт\w*|xp|prp|crp|монет\w*|коин\w*)\b", question, re.IGNORECASE):
        return "profile"
    if not _PERSONAL.search(question):
        return None
    if _PROFILE.search(question) and not (_CALENDAR.search(question) or _PROJECT.search(question)):
        return "profile"
    if _CAMPUS.search(question) and not _CALENDAR.search(question) and not _PROJECT.search(question):
        return None
    if _CALENDAR.search(question):
        return "calendar"
    if _PROJECT.search(question):
        return "projects"
    if re.search(r"\bчто\s+у\s+меня\b", question, re.IGNORECASE):
        return "calendar"
    return None


def day_bounds(tz_name: str, now: datetime, days_from_today: int = 0) -> tuple[str, str]:
    if now.utcoffset() is None:
        raise ValueError("now must be timezone-aware")
    try:
        zone = ZoneInfo(tz_name)
    except (ZoneInfoNotFoundError, ValueError):
        raise ValueError("invalid timezone") from None
    day = now.astimezone(zone).date() + timedelta(days=days_from_today)
    start = datetime.combine(day, time.min, tzinfo=zone)
    end = datetime.combine(day + timedelta(days=1), time.min, tzinfo=zone)
    return start.isoformat(), end.isoformat()


def calendar_bounds(question: str, tz_name: str, now: datetime) -> tuple[str, str]:
    """Resolve supported periods; never silently treat an unknown period as today."""
    if now.utcoffset() is None:
        raise ValueError("now must be timezone-aware")
    try:
        zone = ZoneInfo(tz_name)
    except (ZoneInfoNotFoundError, ValueError):
        raise ValueError("invalid timezone") from None
    local_day = now.astimezone(zone).date()
    if re.search(r"\bнедел\w*", question, re.IGNORECASE):
        monday = local_day - timedelta(days=local_day.weekday())
        if re.search(r"\bследующ\w*\s+недел", question, re.IGNORECASE):
            monday += timedelta(days=7)
        elif re.search(r"\bпрошл\w*\s+недел", question, re.IGNORECASE):
            monday -= timedelta(days=7)
        start = datetime.combine(monday, time.min, tzinfo=zone)
        end = datetime.combine(monday + timedelta(days=7), time.min, tzinfo=zone)
        return start.isoformat(), end.isoformat()
    if re.search(r"\b(?:послезавтра|вчера|месяц\w*|понедельник\w*|вторник\w*|сред[ауеы]|четверг\w*|пятниц\w*|суббот\w*|воскресень\w*|через|январ\w*|феврал\w*|март\w*|апрел\w*|ма[йяе]|июн\w*|июл\w*|август\w*|сентябр\w*|октябр\w*|ноябр\w*|декабр\w*)\b|\b\d{1,4}[./-]\d{1,2}", question, re.IGNORECASE):
        raise ValueError("unsupported calendar period")
    tomorrow = bool(re.search(r"\bзавтра\b", question, re.IGNORECASE))
    return day_bounds(tz_name, now, int(tomorrow))


def agenda_bounds(question: str, tz_name: str, now: datetime) -> tuple[str, str]:
    """Use the platform's 30-day default when no period is specified."""
    bounds = calendar_bounds(question, tz_name, now)  # also rejects unsupported periods
    if re.search(r"\b(?:сегодня|завтра|недел\w*)\b", question, re.IGNORECASE):
        return bounds
    zone = ZoneInfo(tz_name)
    day = now.astimezone(zone).date()
    return (datetime.combine(day, time.min, tzinfo=zone).isoformat(),
            datetime.combine(day + timedelta(days=30), time.min, tzinfo=zone).isoformat())
