"""Разбор коротких согласий («давай», «ок») на проактивные предложения ассистента."""

from __future__ import annotations

import re

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage

_PROACTIVE_MARKER = "**Могу ещё подсказать:**"

_AFFIRMATIVE_FOLLOWUP_RE = re.compile(
    r"^(?:"
    r"давай(?:те)?"
    r"|ок(?:ей)?"
    r"|да"
    r"|ага"
    r"|угу"
    r"|хорошо"
    r"|ладно"
    r"|расскаж(?:и|ите)?"
    r"|интересно"
    r"|подробнее"
    r"|продолж(?:ай|ите)?"
    r"|yes"
    r"|please"
    r")(?:[!?.…,\s]+)?$",
    re.IGNORECASE,
)


def is_affirmative_followup(text: str) -> bool:
    return bool(_AFFIRMATIVE_FOLLOWUP_RE.match((text or "").strip()))


def extract_proactive_suggestions(text: str) -> list[str]:
    """Темы из хвоста «Могу ещё подсказать: …» последнего ответа."""
    raw = text or ""
    idx = raw.find(_PROACTIVE_MARKER)
    if idx < 0:
        return []
    tail = raw[idx + len(_PROACTIVE_MARKER) :].strip()
    if not tail:
        return []
    # Обрезаем всё после блока «Источники», если модель его вставила.
    for marker in ("\n## Источники", "\n##источники"):
        cut = tail.find(marker)
        if cut >= 0:
            tail = tail[:cut].strip()
    parts = re.split(r",\s*|\s+;\s+|\s+и\s+", tail)
    out: list[str] = []
    for part in parts:
        topic = part.strip().strip("-•* ").rstrip(".")
        if topic:
            out.append(topic)
    return out


_PROACTIVE_STOPWORDS = frozenset(
    {
        "как", "что", "где", "когда", "кто", "чем", "чём", "для", "про", "при",
        "или", "это", "там", "тут", "так", "все", "всё", "нужно", "надо",
        "можно", "если", "есть", "мне", "меня", "тебя", "его", "также",
        "the", "and", "for", "you", "can", "how", "what", "where",
        "на", "по", "до", "из", "от", "об", "во", "со", "за", "ну", "да", "нет",
    }
)

_WORD_RE = re.compile(r"[a-zа-яё0-9]+", re.IGNORECASE)


def _proactive_stems(text: str) -> set[str]:
    """Содержательные основы слов (стоп-слова отброшены, основа ≤4 символов)."""
    out: set[str] = set()
    for w in _WORD_RE.findall((text or "").lower()):
        if len(w) < 3 or w in _PROACTIVE_STOPWORDS:
            continue
        out.add(w[:4] if len(w) >= 4 else w)
    return out


def _suggestion_matches_question(suggestion: str, user_question: str) -> bool:
    """Слишком ли близка тема предложения к текущему вопросу пользователя."""
    sug = _proactive_stems(suggestion)
    if not sug:
        return False
    q = _proactive_stems(user_question)
    if not q:
        return False
    shared = sug & q
    if not shared:
        return False
    # Сильный сигнал: общая содержательная лемма (≥4 символов, напр. «гост», «клуб»).
    if any(len(s) >= 4 for s in shared):
        return True
    return len(shared) / len(sug) >= 0.6


def prune_proactive_suggestions(answer: str, user_question: str) -> str:
    """Убирает из «Могу ещё подсказать:» темы-самоповторы текущего вопроса.

    Если после фильтра тем не осталось — удаляет весь блок целиком. Возможный
    хвост «## Источники» (если модель его уже вписала) сохраняется.
    """
    raw = answer or ""
    idx = raw.find(_PROACTIVE_MARKER)
    if idx < 0:
        return answer
    head = raw[:idx].rstrip()
    after = raw[idx + len(_PROACTIVE_MARKER) :]

    tail_suffix = ""
    lower = after.lower()
    for marker in ("\n## источники", "\n##источники"):
        cut = lower.find(marker)
        if cut >= 0:
            tail_suffix = after[cut:].strip()
            break

    suggestions = extract_proactive_suggestions(raw)
    kept = [s for s in suggestions if not _suggestion_matches_question(s, user_question)]

    if kept:
        rebuilt = f"{head}\n\n{_PROACTIVE_MARKER} " + ", ".join(kept)
    else:
        rebuilt = head
    if tail_suffix:
        rebuilt = f"{rebuilt.rstrip()}\n\n{tail_suffix}"
    return rebuilt


def _message_text(msg: BaseMessage) -> str:
    c = msg.content
    return c if isinstance(c, str) else str(c)


def _last_assistant_before_last_human(messages: list[BaseMessage]) -> str | None:
    if not messages or not isinstance(messages[-1], HumanMessage):
        return None
    for msg in reversed(messages[:-1]):
        if isinstance(msg, AIMessage):
            text = _message_text(msg).strip()
            return text or None
    return None


def _prior_substantive_human(messages: list[BaseMessage]) -> str | None:
    """Последний содержательный вопрос пользователя до короткой реплики."""
    seen_last = False
    for msg in reversed(messages):
        if isinstance(msg, HumanMessage):
            text = _message_text(msg).strip()
            if not text:
                continue
            if not seen_last:
                seen_last = True
                continue
            if is_affirmative_followup(text):
                continue
            return text
    return None


_SHORT_CONTEXTUAL_RE = re.compile(
    r"^(?:"
    r"(?:что|как|где|когда|какой|какая|какие|сколько|можно\s+ли|нужно\s+ли)"
    r"|а\s+(?:что|как|где|когда)"
    r")",
    re.IGNORECASE,
)


def is_short_contextual_followup(text: str) -> bool:
    """Короткое уточнение в диалоге (не согласие «давай/ок»)."""
    t = (text or "").strip()
    if not t or is_affirmative_followup(t):
        return False
    if len(t.split()) > 8:
        return False
    if t.endswith("?"):
        return True
    return bool(_SHORT_CONTEXTUAL_RE.match(t))


def resolve_contextual_followup(messages: list[BaseMessage]) -> str | None:
    """Связывает короткий вопрос с предыдущим содержательным вопросом пользователя."""
    if not messages or not isinstance(messages[-1], HumanMessage):
        return None
    last_human = _message_text(messages[-1]).strip()
    if not is_short_contextual_followup(last_human):
        return None
    prior = _prior_substantive_human(messages)
    if not prior:
        return None
    return f"{prior} — уточнение: {last_human}"


def resolve_followup_topic(messages: list[BaseMessage]) -> str | None:
    """Тема для search/ответа при согласии или коротком уточнении в диалоге."""
    if not messages or not isinstance(messages[-1], HumanMessage):
        return None
    last_human = _message_text(messages[-1]).strip()

    contextual = resolve_contextual_followup(messages)
    if contextual:
        return contextual

    if not is_affirmative_followup(last_human):
        return None

    assistant = _last_assistant_before_last_human(messages)
    if assistant:
        suggestions = extract_proactive_suggestions(assistant)
        if suggestions:
            return suggestions[0]

    return _prior_substantive_human(messages)


def expand_followup_question(messages: list[BaseMessage], user_question: str) -> str:
    """Явная формулировка вопроса для генерации ответа."""
    topic = resolve_followup_topic(messages)
    if not topic:
        return user_question
    if " — уточнение: " in topic:
        return f"Ответь на уточнение в контексте предыдущего вопроса: {topic}"
    return f"Расскажи подробнее: {topic}"


__all__ = [
    "expand_followup_question",
    "extract_proactive_suggestions",
    "is_affirmative_followup",
    "is_short_contextual_followup",
    "prune_proactive_suggestions",
    "resolve_contextual_followup",
    "resolve_followup_topic",
]
