"""Кэшируемая проверка доступности Rocket.Chat (короткий probe, без полного каталога)."""

from __future__ import annotations

import asyncio
import logging
import time

from .client import RocketChatClient

log = logging.getLogger("s21.rc.availability")


class RocketChatAvailability:
    """Probe login с коротким timeout и TTL-кэшем результата."""

    def __init__(
        self,
        *,
        probe_timeout_s: float = 8.0,
        cache_ttl_s: float = 30.0,
    ) -> None:
        self._probe_timeout_s = max(1.0, probe_timeout_s)
        self._cache_ttl_s = max(1.0, cache_ttl_s)
        self._available: bool | None = None
        self._error: str | None = None
        self._checked_at: float = 0.0
        self._lock = asyncio.Lock()

    def _cache_fresh(self) -> bool:
        return (
            self._available is not None
            and (time.time() - self._checked_at) < self._cache_ttl_s
        )

    def mark_unavailable(self, reason: str) -> None:
        self._available = False
        self._error = (reason or "unknown error").strip() or "unknown error"
        self._checked_at = time.time()

    def mark_available(self) -> None:
        self._available = True
        self._error = None
        self._checked_at = time.time()

    async def check(
        self,
        client: RocketChatClient,
        *,
        force: bool = False,
    ) -> tuple[bool, str | None]:
        if not force and self._cache_fresh():
            return self._available, self._error

        async with self._lock:
            if not force and self._cache_fresh():
                return self._available, self._error
            ok, err = await self._probe(client)
            self._available = ok
            self._error = err
            self._checked_at = time.time()
            if ok:
                log.info("RC availability: ok")
            else:
                log.warning("RC availability: unavailable (%s)", err)
            return ok, err

    async def _probe(self, client: RocketChatClient) -> tuple[bool, str | None]:
        try:
            await asyncio.wait_for(
                client.probe_login(),
                timeout=self._probe_timeout_s,
            )
            return True, None
        except asyncio.TimeoutError:
            return False, f"probe timeout ({self._probe_timeout_s:.0f}s)"
        except Exception as e:
            return False, str(e) or type(e).__name__
