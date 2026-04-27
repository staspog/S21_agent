"""Выбор Rocket.Chat-комнат для одного подвопроса.

LLM получает компактный текстовый каталог (rid|kind|name|topic) и возвращает
3–5 rid через structured output (`RoomSelection`). Поверх результата всегда
добавляются seed-комнаты из `RC_ALWAYS_SEARCH_ROOMS`.
"""

from __future__ import annotations

import logging

from langchain_gigachat.chat_models import GigaChat

from src.llm_gate import invoke_with_retry
from src.rc.schemas import Room, RoomSelection

log = logging.getLogger("s21.pipeline.select_rooms")


_SYSTEM_PROMPT = (
    "Ты выбираешь Rocket.Chat-комнаты Школы 21, в которых нужно искать ответ.\n"
    "Тебе дан каталог комнат и один подвопрос пользователя.\n"
    "Верни 3–5 rid из каталога, наиболее релевантных по name + topic.\n"
    "Принципы:\n"
    "  • не выдумывай rid — только из каталога;\n"
    "  • предпочитай комнаты с чёткой темой (topic), совпадающей с подвопросом;\n"
    "  • общие/админские комнаты (msk_adm, общие анонсы) подходят для большинства "
    "вопросов про события и расписание;\n"
    "  • если в каталоге мало релевантных — возвращай меньше, не добивай мусором.\n"
    "Отвечай только согласно схеме RoomSelection."
)


def _format_catalog(rooms: list[Room], limit: int = 80) -> str:
    lines: list[str] = []
    for r in rooms[:limit]:
        topic = (r.topic or "").replace("\n", " ").strip()
        if len(topic) > 140:
            topic = topic[:137] + "…"
        lines.append(f"- rid={r.rid} | kind={r.kind} | name={r.name} | topic={topic or '—'}")
    return "\n".join(lines)


def select_rooms(
    *,
    llm: GigaChat,
    subquery: str,
    catalog: list[Room],
    seed_rooms: list[Room],
    target: int = 4,
) -> list[Room]:
    """Возвращает упорядоченный список Room для chat.search; seed_rooms всегда впереди."""
    by_rid: dict[str, Room] = {r.rid: r for r in catalog}
    chosen: list[Room] = []
    seen_rids: set[str] = set()
    for r in seed_rooms:
        if r.rid in by_rid and r.rid not in seen_rids:
            chosen.append(r)
            seen_rids.add(r.rid)

    if not catalog:
        return chosen

    catalog_text = _format_catalog(catalog)
    prompt = (
        f"Подвопрос: {subquery}\n\n"
        f"Каталог комнат (только из него можно выбирать rid):\n{catalog_text}"
    )

    structured = llm.with_structured_output(RoomSelection)
    try:
        sel: RoomSelection = invoke_with_retry(
            structured.invoke,
            [("system", _SYSTEM_PROMPT), ("human", prompt)],
            label="select_rooms",
        )
        candidate_rids = list(sel.selected_rids or [])
    except Exception:
        log.exception("select_rooms: structured_output failed, fallback к seed_rooms")
        candidate_rids = []

    for rid in candidate_rids:
        if rid in by_rid and rid not in seen_rids:
            chosen.append(by_rid[rid])
            seen_rids.add(rid)
        if len(chosen) >= max(target, len(seed_rooms)) + 4:
            break

    if not chosen:
        chosen = catalog[: target]

    chosen = chosen[: max(target, len(seed_rooms) + target - 1)]
    log.info(
        "select_rooms: subquery=%r → %s комнат: %s",
        subquery,
        len(chosen),
        ", ".join(f"{r.kind}/{r.name}" for r in chosen),
    )
    return chosen
