"""Параллельный поиск по комбинации (комната × поисковая строка).

Возвращает список ранжированных списков `list[list[Hit]]` (по одному списку
на каждую (room, query)-пару) — далее их склеивает RRF-фьюжн.
"""

from __future__ import annotations

import asyncio
import logging

from src.rc.client import RocketChatClient
from src.rc.schemas import Hit, Room

log = logging.getLogger("s21.pipeline.search")


async def search_rooms_x_queries(
    *,
    client: RocketChatClient,
    rooms: list[Room],
    queries: list[str],
    count: int,
) -> list[list[Hit]]:
    """Декартово произведение rooms × queries → список ранжированных списков.

    Параллелизм ограничен внутренним семафором клиента; здесь дополнительно
    группируем все вызовы в один gather для явной структуры списков.
    """
    if not rooms or not queries:
        return []

    pairs: list[tuple[Room, str]] = [(r, q) for r in rooms for q in queries]
    log.info(
        "search: rooms=%s queries=%s pairs=%s",
        len(rooms),
        len(queries),
        len(pairs),
    )

    async def _one(room: Room, q: str) -> list[Hit]:
        hits = await client.chat_search(room=room, search_text=q, count=count)
        log.info(
            "search: room=%s q=%r → %s hits",
            room.name,
            q,
            len(hits),
        )
        return hits

    results = await asyncio.gather(
        *(_one(r, q) for r, q in pairs),
        return_exceptions=False,
    )
    return list(results)
