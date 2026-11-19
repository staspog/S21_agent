"""Request-scoped, in-memory MCP bridge for read-only platform data.

The bearer is captured by server-side tool closures, never tool arguments.
The caller must first authenticate it and must not persist this server.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Any
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

import httpx
from mcp.server.fastmcp import FastMCP

_CALENDAR_FIELDS = {
    "reviewList": ("review", ("projectName", "startTime", "endTime", "isOnline", "presenters")),
    "presentationList": ("presentation", ("projectName", "startTime", "endTime", "isOnline", "reviewer")),
    "freeSlotsList": ("free_slot", ("startTime", "endTime")),
    "eventsList": ("event", ("title", "startTime", "endTime", "location")),
    "p2pRequestsList": ("p2p_request", ("goalName", "startTime", "endTime", "isOnline")),
}


class PlatformFailure(Exception):
    def __init__(self, code: str):
        super().__init__(code)
        self.code = code


async def verified_subject(client: httpx.AsyncClient, token: str) -> str:
    """Resolve subject through the API and the upstream platform, not JWT claims."""
    if not token:
        raise PlatformFailure("unauthorized")
    try:
        response = await client.get("/v1/assistant/identity", headers={"Authorization": f"Bearer {token}"})
    except httpx.HTTPError:
        raise PlatformFailure("unavailable") from None
    if response.status_code in (401, 403):
        raise PlatformFailure("unauthorized" if response.status_code == 401 else "forbidden")
    if response.status_code != 200:
        raise PlatformFailure("unavailable")
    try:
        subject = response.json()["subject"]
    except (ValueError, KeyError, TypeError):
        raise PlatformFailure("invalid_response") from None
    if not isinstance(subject, str) or not subject.strip():
        raise PlatformFailure("invalid_response")
    return subject


def make_personal_mcp(client: httpx.AsyncClient, token: str) -> FastMCP:
    """Build one private MCP server per request; never expose via HTTP."""
    server = FastMCP("s21-personal-readonly")

    async def get(path: str, params: dict[str, str] | None = None) -> dict[str, Any]:
        try:
            response = await client.get(path, params=params, headers={"Authorization": f"Bearer {token}"})
        except httpx.HTTPError:
            raise PlatformFailure("unavailable") from None
        if response.status_code in (401, 403):
            raise PlatformFailure("unauthorized" if response.status_code == 401 else "forbidden")
        if response.status_code != 200:
            raise PlatformFailure("unavailable")
        try:
            body = response.json()
        except ValueError:
            raise PlatformFailure("invalid_response") from None
        if not isinstance(body, dict):
            raise PlatformFailure("invalid_response")
        return body

    @server.tool()
    async def get_my_calendar(start: str, end: str, timezone_name: str = "Europe/Moscow") -> dict[str, Any]:
        """Read the authenticated participant's calendar in an ISO time range (at most 31 days)."""
        try:
            zone = ZoneInfo(timezone_name)
            begin = datetime.fromisoformat(start.replace("Z", "+00:00"))
            finish = datetime.fromisoformat(end.replace("Z", "+00:00"))
            if begin.utcoffset() is None or finish.utcoffset() is None:
                raise ValueError
            duration = finish.astimezone(timezone.utc) - begin.astimezone(timezone.utc)
            if duration <= timedelta(0) or duration > timedelta(days=31):
                raise ValueError
        except (ValueError, OverflowError, ZoneInfoNotFoundError):
            return {"status": "invalid_range"}
        try:
            body = await get("/getCalendar", {"from": start, "to": end})
        except PlatformFailure as exc:
            return {"status": exc.code}
        items = []
        for key, (kind, fields) in _CALENDAR_FIELDS.items():
            values = body.get(key)
            if not isinstance(values, list) or not all(isinstance(item, dict) for item in values):
                return {"status": "invalid_response"}
            for item in values:
                selected = {field: item[field] for field in fields if field in item}
                start_time = None
                end_time = None
                for field in ("startTime", "endTime"):
                    if selected.get(field):
                        try:
                            value = datetime.fromisoformat(selected[field].replace("Z", "+00:00"))
                            if value.utcoffset() is None:
                                raise ValueError
                            if field == "startTime":
                                start_time = value.astimezone(timezone.utc)
                            else:
                                end_time = value.astimezone(timezone.utc)
                            selected[field] = value.astimezone(zone).isoformat()
                        except (AttributeError, ValueError, OverflowError):
                            return {"status": "invalid_response"}
                if start_time is None:
                    return {"status": "invalid_response"}
                if end_time is not None and end_time <= start_time:
                    return {"status": "invalid_response"}
                begin_utc, finish_utc = begin.astimezone(timezone.utc), finish.astimezone(timezone.utc)
                if start_time >= finish_utc or (end_time is not None and end_time <= begin_utc) or (end_time is None and start_time < begin_utc):
                    continue
                if "presenters" in selected:
                    presenters = selected["presenters"]
                    selected["presenters"] = [p.get("login") for p in presenters if isinstance(p, dict) and p.get("login")] if isinstance(presenters, list) else []
                items.append({"type": kind, **selected})
        if len(items) > 200:
            return {"status": "too_many_items"}
        return {"status": "ok", "items": items, "retrieved_at": datetime.now(timezone.utc).isoformat()}

    @server.tool()
    async def get_my_projects() -> dict[str, Any]:
        """Read current projects of the authenticated participant."""
        try:
            body = await get("/getProjects")
        except PlatformFailure as exc:
            return {"status": exc.code}
        projects = body.get("projects")
        if not isinstance(projects, list) or not all(isinstance(item, dict) for item in projects):
            return {"status": "invalid_response"}
        if len(projects) > 200:
            return {"status": "too_many_items"}
        return {"status": "ok", "projects": [
            {key: item[key] for key in ("name", "goalStatus", "displayedCourseStatus") if key in item}
            for item in projects
        ], "retrieved_at": datetime.now(timezone.utc).isoformat()}

    return server
