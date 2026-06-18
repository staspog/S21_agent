"""Выбор Rocket.Chat-комнат для одного подвопроса.

Комнаты берутся из seed-списка (`RoomCatalog.seed_rooms()` / RC_ALWAYS_SEARCH_ROOMS).
Порядок campus-aware: ADM-каналы кампуса из подвопроса — в начало; иначе все *_adm
и msk_adm/msk_general первыми.
"""

from __future__ import annotations

import logging
import re

from src.reference_intent import detect_campus
from src.rc.schemas import Room

log = logging.getLogger("s21.pipeline.select_rooms")

_CAMPUS_PREFIXES: dict[str, tuple[str, ...]] = {
    "Москва": ("msk_adm", "msk_general"),
    "Казань": ("kzn_adm", "kzn_volunteers"),
    "Новосибирск": ("nsk_adm",),
    "Сургут": ("skd_adm",),
    "Якутск": ("yks_adm", "tas_adm"),
    "Ярославль": ("yar_adm",),
    "Магас": ("mag_adm",),
    "Белгород": ("blg_adm",),
    "Южно-Сахалинск": ("yuz_adm",),
    "Челябинск": ("chel_adm",),
    "Нижний Новгород": ("nn_adm", "nn_helpers"),
    "Липецк": ("lip_adm",),
    "Уфа": ("ufa_adm",),
    "Омск": ("oms_adm",),
    "Волгоград": ("vlg_adm",),
    "Пермь": ("perm_adm",),
    "Ставрополь": ("stv_adm",),
    "Великий Новгород": ("vng_adm",),
}

_DEFAULT_ADM_PRIORITY: tuple[str, ...] = (
    "msk_adm",
    "msk_general",
    "nsk_adm",
    "kzn_adm",
    "tas_adm",
    "skd_adm",
    "yks_adm",
)

_CAMPUS_CODE_RE = re.compile(
    r"\b("
    r"msk|nsk|kzn|skd|yks|tas|yar|mag|blg|nn|ufa|oms|vlg|perm|stv|"
    r"lip|chel|vng|yuz"
    r")(?:_(?:adm|general))?\b",
    re.I,
)


def _priority_room_names(subquery: str) -> list[str]:
    """Имена комнат в порядке приоритета для подвопроса."""
    campus = detect_campus(subquery)
    if campus and campus in _CAMPUS_PREFIXES:
        names = list(_CAMPUS_PREFIXES[campus])
        names.extend(_DEFAULT_ADM_PRIORITY)
        return _dedupe_names(names)

    code_match = _CAMPUS_CODE_RE.search(subquery or "")
    if code_match:
        code = code_match.group(1).lower()
        names = [f"{code}_adm"]
        if code == "msk":
            names.append("msk_general")
        names.extend(_DEFAULT_ADM_PRIORITY)
        return _dedupe_names(names)

    return list(_DEFAULT_ADM_PRIORITY)


def _dedupe_names(names: list[str]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for name in names:
        key = name.lower()
        if key in seen:
            continue
        seen.add(key)
        out.append(name)
    return out


def _order_rooms(seed_rooms: list[Room], priority_names: list[str]) -> list[Room]:
    by_name: dict[str, Room] = {r.name.lower(): r for r in seed_rooms}
    ordered: list[Room] = []
    seen: set[str] = set()

    for name in priority_names:
        room = by_name.get(name.lower())
        if room and room.rid not in seen:
            ordered.append(room)
            seen.add(room.rid)

    adm_rest = [
        r for r in seed_rooms if r.name.lower().endswith("_adm") and r.rid not in seen
    ]
    adm_rest.sort(key=lambda r: r.name.lower())
    for room in adm_rest:
        ordered.append(room)
        seen.add(room.rid)

    for room in seed_rooms:
        if room.rid not in seen:
            ordered.append(room)
            seen.add(room.rid)

    return ordered


def adm_rooms_subset(rooms: list[Room]) -> list[Room]:
    """ADM-каналы из уже выбранного списка (для двухфазного поиска)."""
    adm = [r for r in rooms if r.name.lower().endswith("_adm")]
    if adm:
        return adm
    return list(rooms[: min(2, len(rooms))])


def select_rooms(
    *,
    subquery: str,
    catalog: list[Room],
    seed_rooms: list[Room],
    target: int = 4,
) -> list[Room]:
    """Campus-aware выбор комнат из seed-списка."""
    cap = max(1, int(target))
    base = list(seed_rooms) if seed_rooms else list(catalog)
    if not base:
        log.warning("select_rooms: пустой seed/catalog для subquery=%r", subquery[:60])
        return []

    priority = _priority_room_names(subquery)
    ordered = _order_rooms(base, priority)
    chosen = ordered[:cap]
    log.info(
        "select_rooms: subquery=%r → %s rooms (cap=%s): %s",
        subquery[:60],
        len(chosen),
        cap,
        ", ".join(r.name for r in chosen),
    )
    return chosen
