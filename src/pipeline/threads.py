"""Подгрузка тредов для топ-N hit-ов одного подвопроса.

В школьных чатах часто встречается: анонс + детали в реплаях. RC `chat.search`
не возвращает реплаи треда, поэтому для топ-N результатов с `tmid` (или
самих корней треда) подтягиваем `chat.getThreadMessages` и добавляем
сообщения треда как дополнительный контекст.
"""

from __future__ import annotations

import asyncio
import logging

from src.rc.client import RocketChatClient
from src.rc.schemas import Hit, Room

log = logging.getLogger("s21.pipeline.threads")


async def expand_threads(
    *,
    client: RocketChatClient,
    rooms_by_rid: dict[str, Room],
    hits: list[Hit],
    top_n: int,
    per_thread_count: int,
) -> list[Hit]:
    """Для топ-N хитов подгружает их треды и добавляет реплаи.

    Возвращает новый список: исходные топ-N + дедуплицированные реплаи треда
    + остальные hit-ы (после top_n) — без изменения позиций родителей.
    """
    if not hits or top_n <= 0:
        return list(hits)

    head = hits[: top_n]
    tail = hits[top_n :]

    # Список (room, tmid). tmid = thread_root_mid если есть, иначе сам mid (как корень).
    targets: list[tuple[Room, str]] = []
    seen: set[tuple[str, str]] = set()
    for h in head:
        tmid = h.thread_root_mid or h.mid
        room = rooms_by_rid.get(h.rid)
        if not room:
            continue
        key = (h.rid, tmid)
        if key in seen:
            continue
        seen.add(key)
        targets.append((room, tmid))

    if not targets:
        return list(hits)

    log.info("threads: подгружаем %s тредов", len(targets))

    async def _one(room: Room, tmid: str) -> list[Hit]:
        return await client.get_thread_messages(
            tmid=tmid, room=room, count=per_thread_count
        )

    fetched_raw = await asyncio.gather(
        *(_one(r, t) for r, t in targets),
        return_exceptions=True,
    )
    fetched: list[list[Hit]] = []
    for item in fetched_raw:
        if isinstance(item, BaseException):
            log.warning("threads: fetch failed: %s", item)
            fetched.append([])
        else:
            fetched.append(item)

    existing_mids = {h.mid for h in hits}
    extras: list[Hit] = []
    for thread_hits in fetched:
        for th in thread_hits:
            if th.mid in existing_mids:
                continue
            existing_mids.add(th.mid)
            extras.append(th)

    log.info("threads: добавлено %s новых сообщений из тредов", len(extras))
    return head + extras + tail
