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
    """Декартово произведение rooms × queries → список ранжированных списков."""
    if not rooms or not queries:
        return []

    pairs: list[tuple[Room, str]] = [(r, q) for r in rooms for q in queries]
    timeout_s = client.search_timeout_s
    log.info(
        "search: rooms=%s queries=%s pairs=%s timeout_s=%s",
        len(rooms),
        len(queries),
        len(pairs),
        timeout_s,
    )

    async def _one(room: Room, q: str) -> list[Hit]:
        try:
            hits = await asyncio.wait_for(
                client.chat_search(room=room, search_text=q, count=count),
                timeout=timeout_s + 2.0,
            )
        except TimeoutError:
            log.warning(
                "search: timeout room=%s q=%r limit_s=%s",
                room.name,
                q,
                timeout_s,
            )
            return []
        except Exception:
            log.exception("search: failed room=%s q=%r", room.name, q)
            return []
        log.info(
            "search: room=%s q=%r → %s hits",
            room.name,
            q,
            len(hits),
        )
        return hits

    raw = await asyncio.gather(*(_one(r, q) for r, q in pairs), return_exceptions=True)
    out: list[list[Hit]] = []
    errors = 0
    for item in raw:
        if isinstance(item, BaseException):
            errors += 1
            log.warning("search: pair failed: %s", item)
            out.append([])
        else:
            out.append(item)
    if errors:
        log.warning("search: %s/%s pairs failed", errors, len(pairs))
    return out
