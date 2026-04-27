"""TTL-кэш каталога комнат, чтобы не дёргать REST на каждый запрос."""

from __future__ import annotations

import asyncio
import logging
import time

from .client import RocketChatClient
from .config import RocketChatConfig
from .schemas import Room

log = logging.getLogger("s21.rc.catalog")


class RoomCatalog:
    """Кэш каталога с TTL и асинхронной защитой от race-условий при обновлении."""

    def __init__(self, client: RocketChatClient, cfg: RocketChatConfig) -> None:
        self._client = client
        self._cfg = cfg
        self._rooms: list[Room] = []
        self._fetched_at: float = 0.0
        self._lock = asyncio.Lock()

    def _is_fresh(self) -> bool:
        return (
            self._rooms
            and (time.time() - self._fetched_at) < self._cfg.catalog_ttl_s
        )

    async def get(self, force_refresh: bool = False) -> list[Room]:
        if not force_refresh and self._is_fresh():
            return list(self._rooms)
        async with self._lock:
            if not force_refresh and self._is_fresh():
                return list(self._rooms)
            log.info("Каталог комнат: refresh (TTL=%ss)", self._cfg.catalog_ttl_s)
            self._rooms = await self._client.list_accessible_rooms()
            self._fetched_at = time.time()
            log.info("Каталог комнат: загружено %s", len(self._rooms))
            return list(self._rooms)

    def find_by_name(self, name: str) -> Room | None:
        target = (name or "").strip().lstrip("#").lower()
        for r in self._rooms:
            if r.name.lower() == target:
                return r
        return None

    def find_by_rid(self, rid: str) -> Room | None:
        rid = (rid or "").strip()
        for r in self._rooms:
            if r.rid == rid:
                return r
        return None

    def seed_rooms(self) -> list[Room]:
        """Комнаты, в которых ищем всегда (из RC_ALWAYS_SEARCH_ROOMS)."""
        out: list[Room] = []
        for name in self._cfg.always_search_room_names:
            r = self.find_by_name(name)
            if r:
                out.append(r)
        return out
