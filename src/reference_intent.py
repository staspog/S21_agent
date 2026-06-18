"""Intent для справочных каталогов ADM (клубы, benefits).

Определяет, когда ответ строится из RAG без поиска в Rocket.Chat.
"""

from __future__ import annotations

import re
from typing import Literal

from src.rc.schemas import RagChunk

ReferenceTopic = Literal["clubs", "benefits", "guests"]

TOPIC_SLUGS: dict[ReferenceTopic, str] = {
    "clubs": "education/club_chats",
    "benefits": "benefits",
    "guests": "guests",
}

CANONICAL_URLS: dict[ReferenceTopic, str] = {
    "clubs": "https://applicant.21-school.ru/education/clubs",
    "guests": "https://applicant.21-school.ru/guests",
    "benefits": "https://applicant.21-school.ru/education/bonuses",
}

# Страницы adm_info без строки ``> Оригинал:`` — не публикуются на applicant.
UNPUBLISHED_APPLICANT_URLS: frozenset[str] = frozenset()

# Заголовки ## в club_chats/index.md (порядок важен для slice)
CLUB_CAMPUS_HEADERS: tuple[str, ...] = (
    "Москва",
    "Казань",
    "Новосибирск",
    "Сургут",
    "Великий Новгород",
    "Якутск",
    "Ярославль",
    "Магас",
    "Белгород",
    "Южно-Сахалинск",
    "Челябинск",
    "Нижний Новгород",
    "Липецк",
    "Уфа",
    "Омск",
    "Волгоград",
    "Пермь",
    "Ставрополь",
    "Сеченовский университет",
)

# Заголовки ## в benefits/index.md
BENEFITS_CITY_HEADERS: tuple[str, ...] = (
    "Москва",
    "Казань",
    "Новосибирск",
    "Уфа",
)

# Заголовки ## городов в guests/index.md (порядок важен для slice).
# Примечание: в самом md заголовок Магадана с опечаткой — «Магдан»; держим
# точную строку из файла, чтобы _select_section_slug_chunks матчился корректно.
GUEST_CAMPUS_HEADERS: tuple[str, ...] = (
    "Москва",
    "Казань",
    "Новосибирск",
    "Сургут",
    "Великий Новгород",
    "Якутск",
    "Магас",
    "Ярославль",
    "Белгород",
    "Южно-Сахалинск",
    "Челябинск",
    "Магдан",
    "Нижний Новгород",
    "Липецк",
    "Уфа",
    "Омск",
    "Ставрополь",
)

_CAMPUS_PATTERNS: tuple[tuple[re.Pattern[str], str], ...] = (
    (re.compile(r"\b(?:в\s+)?(?:г\.?\s*)?москв(?:е|у|ы)\b", re.I), "Москва"),
    (re.compile(r"\b(?:в\s+)?(?:г\.?\s*)?казан(?:и|ь|ью)\b", re.I), "Казань"),
    (re.compile(r"\b(?:в\s+)?(?:г\.?\s*)?новосибирск(?:е|а|у)?\b", re.I), "Новосибирск"),
    (re.compile(r"\b(?:в\s+)?(?:г\.?\s*)?сургут(?:е|а|у)?\b", re.I), "Сургут"),
    (re.compile(r"\b(?:в\s+)?(?:г\.?\s*)?велик(?:ом|ий)\s+новгород(?:е|у)?\b", re.I), "Великий Новгород"),
    (re.compile(r"\b(?:в\s+)?(?:г\.?\s*)?якутск(?:е|а|у)?\b", re.I), "Якутск"),
    (re.compile(r"\b(?:в\s+)?(?:г\.?\s*)?ярославл(?:е|ь|ю)\b", re.I), "Ярославль"),
    (re.compile(r"\b(?:в\s+)?(?:г\.?\s*)?магас(?:е|а|у)?\b", re.I), "Магас"),
    (re.compile(r"\b(?:в\s+)?(?:г\.?\s*)?белгород(?:е|а|у)?\b", re.I), "Белгород"),
    (re.compile(r"\b(?:в\s+)?(?:г\.?\s*)?(?:южно[- ]?)?сахалинск(?:е|а|у)?\b", re.I), "Южно-Сахалинск"),
    (re.compile(r"\b(?:в\s+)?(?:г\.?\s*)?челябинск(?:е|а|у)?\b", re.I), "Челябинск"),
    (re.compile(r"\b(?:в\s+)?(?:г\.?\s*)?нижн(?:ем|ий|его)\s+новгород(?:е|у)?\b", re.I), "Нижний Новгород"),
    (re.compile(r"\b(?:в\s+)?(?:г\.?\s*)?липецк(?:е|а|у)?\b", re.I), "Липецк"),
    (re.compile(r"\b(?:в\s+)?(?:г\.?\s*)?уф(?:е|а|у)?\b", re.I), "Уфа"),
    (re.compile(r"\b(?:в\s+)?(?:г\.?\s*)?омск(?:е|а|у)?\b", re.I), "Омск"),
    (re.compile(r"\b(?:в\s+)?(?:г\.?\s*)?волгоград(?:е|а|у)?\b", re.I), "Волгоград"),
    (re.compile(r"\b(?:в\s+)?(?:г\.?\s*)?перм(?:и|ь|ью)?\b", re.I), "Пермь"),
    (re.compile(r"\b(?:в\s+)?(?:г\.?\s*)?ставропол(?:е|я|ю)?\b", re.I), "Ставрополь"),
    (re.compile(r"\b(?:в\s+)?(?:г\.?\s*)?с(?:анкт[- ]?)?петербург(?:е|а|у)?\b", re.I), "Великий Новгород"),
    (re.compile(r"\b(?:в\s+)?(?:г\.?\s*)?(?:спб|питер(?:е|а|у)?)\b", re.I), "Великий Новгород"),
)

_CLUBS_RE = re.compile(
    r"(?:"
    r"\bклуб(?:ы|а|ов|е|у)?\b"
    r"|\bстуд(?:енческ)?(?:ие|их|ческ)?\s+клуб"
    r"|\bclub(?:s)?\b"
    r"|\brocketchat\b.*\bклуб"
    r"|\bчаты?\s+клуб"
    r")",
    re.I,
)

_BENEFITS_RE = re.compile(
    r"(?:"
    r"\bскидк(?:а|и|у|ой|ам|ами)?\b"
    r"|\bбонус(?:ы|а|ов|ам)?\b"
    r"|\bпромокод"
    r"|\bпартн[ёе]р"
    r"|\bbenefits\b"
    r")",
    re.I,
)

_GUESTS_RE = re.compile(
    r"(?:"
    r"\bгост(?:ь|я|ю|ём|ем|е|и|ей|ям|ями|ях)\b"
    r"|\bгостев(?:ая|ой|ую|ые|ым|ыми|ого)?\s*форм"
    r"|\bguest(?:s)?\b"
    r")",
    re.I,
)


def detect_reference_topic(question: str) -> ReferenceTopic | None:
    q = (question or "").strip()
    if not q:
        return None
    # Несколько топиков могут совпасть в одном вопросе — берём тот, чьё
    # ключевое слово встречается раньше (обобщает прежнюю clubs-vs-benefits
    # tie-логику по позиции «клуб»/«скид»).
    matches: list[tuple[int, ReferenceTopic]] = []
    for pat, topic in (
        (_CLUBS_RE, "clubs"),
        (_BENEFITS_RE, "benefits"),
        (_GUESTS_RE, "guests"),
    ):
        m = pat.search(q)
        if m:
            matches.append((m.start(), topic))
    if not matches:
        return None
    matches.sort(key=lambda t: t[0])
    return matches[0][1]


def detect_campus(question: str) -> str | None:
    q = question or ""
    for pat, name in _CAMPUS_PATTERNS:
        if pat.search(q):
            return name
    return None


def topic_slug(topic: ReferenceTopic | None) -> str | None:
    if topic is None:
        return None
    return TOPIC_SLUGS[topic]


def rag_covers_topic(chunks: list[RagChunk] | None, topic: ReferenceTopic | None) -> bool:
    if not topic or not chunks:
        return False
    slug = TOPIC_SLUGS[topic]
    for ch in chunks:
        if ch.metadata.slug == slug:
            return True
    return False


def should_skip_rocket_search(
    question: str,
    rag_chunks: list[RagChunk] | None,
    rocket_search: bool,
) -> bool:
    if not rocket_search:
        return False
    topic = detect_reference_topic(question)
    if topic is None:
        return False
    return rag_covers_topic(rag_chunks, topic)


def resolve_reference_context(
    question: str,
    rag_chunks: list[RagChunk] | None,
    rocket_search: bool,
) -> tuple[ReferenceTopic | None, str | None, bool]:
    """(topic, campus, rag_first)."""
    topic = detect_reference_topic(question)
    campus = detect_campus(question) if topic else None
    rag_first = should_skip_rocket_search(question, rag_chunks, rocket_search)
    return topic, campus, rag_first


__all__ = [
    "ReferenceTopic",
    "TOPIC_SLUGS",
    "CANONICAL_URLS",
    "UNPUBLISHED_APPLICANT_URLS",
    "CLUB_CAMPUS_HEADERS",
    "BENEFITS_CITY_HEADERS",
    "GUEST_CAMPUS_HEADERS",
    "detect_reference_topic",
    "detect_campus",
    "topic_slug",
    "rag_covers_topic",
    "should_skip_rocket_search",
    "resolve_reference_context",
]
