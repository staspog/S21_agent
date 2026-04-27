"""Async REST-клиент Rocket.Chat.

Минимальный по поверхности: login (с авто-relogin при 401), список комнат
(channels.list.joined / groups.list / dm.list), chat.search и
chat.getThreadMessages. Все вызовы — асинхронные (httpx.AsyncClient).
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any

import httpx

from .config import RocketChatConfig
from .schemas import Hit, Room, RoomKind

log = logging.getLogger("s21.rc.client")


class RocketChatClient:
    """Тонкий async-клиент Rocket.Chat REST API."""

    def __init__(self, cfg: RocketChatConfig) -> None:
        self._cfg = cfg
        self._user_id: str | None = None
        self._auth_token: str | None = None
        self._auth_lock = asyncio.Lock()
        timeout = httpx.Timeout(
            connect=cfg.timeout_connect_s,
            read=cfg.timeout_read_s,
            write=cfg.timeout_read_s,
            pool=cfg.timeout_connect_s,
        )
        limits = httpx.Limits(
            max_connections=max(16, cfg.http_concurrency * 2),
            max_keepalive_connections=cfg.http_concurrency,
        )
        self._client = httpx.AsyncClient(
            base_url=cfg.base_url,
            timeout=timeout,
            limits=limits,
            headers={"Content-Type": "application/json"},
        )
        self._sem = asyncio.Semaphore(cfg.http_concurrency)

    async def aclose(self) -> None:
        await self._client.aclose()

    async def __aenter__(self) -> RocketChatClient:
        return self

    async def __aexit__(self, *args: Any) -> None:
        await self.aclose()

    async def _login(self) -> tuple[str, str]:
        """Логинимся; результат кэшируется до 401 от сервера."""
        async with self._auth_lock:
            if self._user_id and self._auth_token:
                return self._user_id, self._auth_token
            payload = {"user": self._cfg.user, "password": self._cfg.password}
            r = await self._client.post("/api/v1/login", json=payload)
            r.raise_for_status()
            data = r.json()
            if data.get("status") != "success" or "data" not in data:
                raise RuntimeError(f"Rocket.Chat login failed: {data!r}")
            d = data["data"]
            self._user_id = str(d["userId"])
            self._auth_token = str(d["authToken"])
            log.info("RC login ok user_id=%s", self._user_id)
            return self._user_id, self._auth_token

    async def _auth_headers(self) -> dict[str, str]:
        uid, token = await self._login()
        return {"X-User-Id": uid, "X-Auth-Token": token}

    async def _request_json(
        self,
        method: str,
        path: str,
        *,
        params: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Запрос с авто-rerelogin на 401 и обработкой 429 (до 2 ретраев)."""
        attempts = 0
        last_exc: Exception | None = None
        while attempts < 3:
            attempts += 1
            try:
                async with self._sem:
                    headers = await self._auth_headers()
                    r = await self._client.request(
                        method, path, headers=headers, params=params
                    )
                if r.status_code == 401:
                    async with self._auth_lock:
                        self._user_id = None
                        self._auth_token = None
                    log.warning("RC 401 on %s %s — повторный login", method, path)
                    continue
                if r.status_code == 429:
                    reset = r.headers.get("x-ratelimit-reset")
                    sleep_s = 1.0
                    if reset:
                        try:
                            # Заголовок — UTC ms epoch.
                            import time

                            sleep_s = max(
                                0.1, (int(reset) / 1000.0) - time.time()
                            )
                        except ValueError:
                            sleep_s = 1.0
                    sleep_s = min(sleep_s, 5.0)
                    log.warning(
                        "RC 429 on %s %s, sleep=%.2fs", method, path, sleep_s
                    )
                    await asyncio.sleep(sleep_s)
                    continue
                r.raise_for_status()
                return r.json()
            except (httpx.TransportError, httpx.HTTPStatusError) as e:
                last_exc = e
                if attempts >= 3:
                    break
                await asyncio.sleep(0.3 * attempts)
        if last_exc:
            raise last_exc
        raise RuntimeError(f"RC request failed: {method} {path}")

    async def list_accessible_rooms(self) -> list[Room]:
        """Все доступные комнаты пользователя: каналы, группы, DM (cap по конфигу)."""
        cfg = self._cfg
        rooms: list[Room] = []
        cap = cfg.catalog_max_rooms

        async def fetch_kind(
            path: str,
            data_key: str,
            kind: RoomKind,
            label_factory,
        ) -> None:
            offset = 0
            page = cfg.catalog_page_size
            while len(rooms) < cap:
                try:
                    data = await self._request_json(
                        "GET", path, params={"count": page, "offset": offset}
                    )
                except Exception:
                    log.exception("RC %s failed", path)
                    return
                items = data.get(data_key) or []
                for it in items:
                    if len(rooms) >= cap:
                        break
                    rid = str(it.get("_id") or "").strip()
                    if not rid:
                        continue
                    name = label_factory(it)
                    topic = (it.get("topic") or it.get("description") or "").strip()
                    rooms.append(Room(rid=rid, name=name, kind=kind, topic=topic))
                if len(items) < page:
                    return
                offset += page

        await fetch_kind(
            "/api/v1/channels.list.joined",
            "channels",
            "channel",
            lambda c: (c.get("name") or c.get("fname") or c["_id"]),
        )
        await fetch_kind(
            "/api/v1/groups.list",
            "groups",
            "group",
            lambda g: (g.get("name") or g.get("fname") or g["_id"]),
        )
        await fetch_kind(
            "/api/v1/dm.list",
            "ims",
            "im",
            lambda im: (
                ", ".join(im.get("usernames") or [])
                or im.get("fname")
                or im["_id"]
            ),
        )

        log.info("RC catalog size=%s (cap=%s)", len(rooms), cap)
        return rooms

    async def chat_search(
        self,
        *,
        room: Room,
        search_text: str,
        count: int | None = None,
    ) -> list[Hit]:
        """`/api/v1/chat.search`: подстрочный поиск по комнате."""
        c = count if count is not None else self._cfg.search_count
        try:
            data = await self._request_json(
                "GET",
                "/api/v1/chat.search",
                params={
                    "roomId": room.rid,
                    "searchText": search_text,
                    "count": c,
                    "offset": 0,
                },
            )
        except Exception:
            log.exception(
                "RC chat.search failed room=%s q=%r", room.name, search_text
            )
            return []
        if not data.get("success", True):
            log.warning(
                "RC chat.search not success room=%s err=%s",
                room.name,
                data.get("error"),
            )
            return []
        out: list[Hit] = []
        for m in data.get("messages") or []:
            mid = str(m.get("_id") or "").strip()
            if not mid:
                continue
            u = m.get("u") or {}
            tmid = m.get("tmid")
            out.append(
                Hit(
                    mid=mid,
                    rid=room.rid,
                    room_name=room.name,
                    room_kind=room.kind,
                    msg=str(m.get("msg") or ""),
                    ts=str(m.get("ts") or ""),
                    user=str(u.get("username") or ""),
                    rc_score=(
                        float(m["score"]) if isinstance(m.get("score"), (int, float))
                        else None
                    ),
                    thread_root_mid=str(tmid) if tmid else None,
                    is_thread_reply=bool(tmid),
                    permalink=build_permalink(self._cfg.base_url, room, mid),
                )
            )
        return out

    async def get_thread_messages(
        self,
        *,
        tmid: str,
        room: Room,
        count: int | None = None,
    ) -> list[Hit]:
        """`/api/v1/chat.getThreadMessages`: сообщения треда по корневому id."""
        c = count if count is not None else self._cfg.thread_messages_count
        try:
            data = await self._request_json(
                "GET",
                "/api/v1/chat.getThreadMessages",
                params={"tmid": tmid, "count": c, "offset": 0},
            )
        except Exception:
            log.exception("RC getThreadMessages failed tmid=%s", tmid)
            return []
        if not data.get("success", True):
            return []
        out: list[Hit] = []
        for m in data.get("messages") or []:
            mid = str(m.get("_id") or "").strip()
            if not mid:
                continue
            u = m.get("u") or {}
            out.append(
                Hit(
                    mid=mid,
                    rid=room.rid,
                    room_name=room.name,
                    room_kind=room.kind,
                    msg=str(m.get("msg") or ""),
                    ts=str(m.get("ts") or ""),
                    user=str(u.get("username") or ""),
                    thread_root_mid=tmid,
                    is_thread_reply=mid != tmid,
                    permalink=build_permalink(self._cfg.base_url, room, mid),
                )
            )
        return out


def build_permalink(base_url: str, room: Room, mid: str) -> str:
    """Сформировать веб-ссылку на конкретное сообщение.

    channel  -> /channel/<name>?msg=<mid>
    group    -> /group/<name>?msg=<mid>
    im (DM)  -> /direct/<rid>?msg=<mid>  (используем rid, т.к. имени у DM нет)
    """
    base = base_url.rstrip("/")
    if room.kind == "im":
        ident = room.rid
        seg = "direct"
    elif room.kind == "channel":
        ident = room.name
        seg = "channel"
    else:
        ident = room.name
        seg = "group"
    return f"{base}/{seg}/{ident}?msg={mid}"
