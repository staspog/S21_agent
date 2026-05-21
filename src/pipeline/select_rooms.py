"""Выбор Rocket.Chat-комнат для одного подвопроса.

В этом режиме LLM НЕ используется: мы всегда ищем по фиксированному набору
комнат (топ-25 по usersCount), который задаётся в `RocketChatConfig`.
`seed_rooms` уже содержит эти комнаты (через `RoomCatalog.seed_rooms()`).
"""

from __future__ import annotations

import logging

from langchain_gigachat.chat_models import GigaChat

from src.rc.schemas import Room

log = logging.getLogger("s21.pipeline.select_rooms")


def select_rooms(
    *,
    llm: GigaChat,
    subquery: str,
    catalog: list[Room],
    seed_rooms: list[Room],
    target: int = 4,
) -> list[Room]:
    """Возвращает фиксированный набор комнат для поиска (топ-25).

    Сигнатура сохранена для совместимости с графом, но `llm/subquery/target`
    не влияют на выбор комнат.
    """
    cap = max(1, int(target))
    if seed_rooms:
        chosen = list(seed_rooms[:cap])
        log.info("select_rooms: fixed top-rooms mode → %s rooms (cap=%s)", len(chosen), cap)
        return chosen

    chosen = list(catalog[:cap]) if catalog else []
    log.info("select_rooms: fixed mode fallback to catalog → %s rooms", len(chosen))
    return chosen
