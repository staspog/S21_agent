"""Choose a bounded set of read-only platform tools without credentials."""
from __future__ import annotations

import asyncio
import re
from typing import Literal

from pydantic import BaseModel, Field
from .llm_gate import invoke_with_retry

ToolName = Literal[
    "get_my_calendar", "get_my_projects", "get_my_profile", "get_public_profile",
    "get_my_notification_count", "get_my_discounts",
    "get_my_notifications", "get_platform_events",
    "get_my_agenda", "get_platform_announcements", "get_my_search_history",
    "get_my_achievements", "get_public_achievements", "get_achievements_catalog",
    "get_my_project_details", "get_review_slots",
    "get_my_xp_history", "get_my_coalition", "get_my_campus_time",
    "get_public_xp_history", "get_public_coalition", "get_public_campus_time",
]


class PlannedRead(BaseModel):
    """One read-only platform tool selected for the user's question."""
    name: ToolName = Field(description="Name of a permitted read-only tool")
    nickname: str | None = Field(default=None, description="Explicit public nickname when needed")
    project_name: str | None = Field(default=None, description="Explicit current project name when needed")
    offset: int | None = Field(default=None, ge=0, le=10000, description="Explicit zero-based page offset when needed")


class ReadPlan(BaseModel):
    """Select read-only platform tools needed to answer the current question."""
    calls: list[PlannedRead] = Field(default_factory=list, max_length=4, description="Read-only tools required by the question")
    unsupported: list[Literal["platform_search"]] = Field(default_factory=list, max_length=1, description="Requested platform capabilities that are unavailable")


class InvalidReadPlan(Exception):
    pass


PAGED_TOOLS = frozenset({"get_my_notifications", "get_platform_events", "get_my_xp_history", "get_public_xp_history", "get_platform_announcements", "get_my_search_history", "get_my_achievements", "get_public_achievements", "get_achievements_catalog"})


def validate_read_plan(plan: ReadPlan, question: str, *, current_question: str | None = None) -> ReadPlan:
    seen: set[tuple[str, str, str]] = set()
    for call in plan.calls:
        if call.name in ("get_public_profile", "get_public_achievements", "get_public_xp_history", "get_public_coalition", "get_public_campus_time"):
            nick = call.nickname or ""
            if not re.fullmatch(r"[A-Za-z0-9_.-]{1,64}", nick):
                raise InvalidReadPlan("missing_nickname")
            if not re.search(r"(?<![A-Za-z0-9_.-])" + re.escape(nick) + r"(?![A-Za-z0-9_.-])", question, re.IGNORECASE):
                raise InvalidReadPlan("nickname_not_explicit")
        elif call.nickname is not None:
            raise InvalidReadPlan("unexpected_nickname")
        if call.project_name is not None:
            project = call.project_name.strip()
            if call.name not in ("get_my_project_details", "get_review_slots") or not project or len(project) > 256 or project.casefold() not in question.casefold():
                raise InvalidReadPlan("project_not_explicit")
        if call.offset is not None:
            if call.name not in PAGED_TOOLS or not re.search(r"(?<!\d)" + str(call.offset) + r"(?!\d)", current_question if current_question is not None else question):
                raise InvalidReadPlan("offset_not_explicit")
        key = (call.name, call.nickname or "", call.project_name or "")
        if key in seen:
            raise InvalidReadPlan("duplicate_read")
        seen.add(key)
    return plan


async def plan_reads(llm, question: str, history: tuple[str, ...] = ()) -> ReadPlan:
    """The planner sees only the user's question and a fixed read-only catalog."""
    system = (
        "Выбери инструменты для ответа на вопрос об актуальных данных платформы Школы 21. "
        "Верни пустой calls для общего вопроса или сведений кампуса. "
        "get_my_profile: мой уровень, опыт, балансы и профиль; get_my_projects: мои текущие проекты; "
        "get_my_xp_history: история начисления опыта; get_my_coalition: коалиция и её рейтинг; "
        "get_my_campus_time: моё время в кампусе и время авторизации за неделю; "
        "get_public_xp_history, get_public_coalition и get_public_campus_time: соответствующие открытые данные участника с явно указанным nickname; "
        "get_my_project_details: детали одного текущего проекта и требования проверок; "
        "get_review_slots: доступные слоты проверки для одного текущего проекта; "
        "для этих двух при явно указанном названии укажи project_name точной строкой вопроса, иначе null. "
        "Для прочих инструментов project_name должен быть null. "
        "get_my_calendar: мои проверки, защиты, свободные слоты и личный календарь; "
        "get_my_notification_count: число непрочитанных уведомлений; get_my_notifications: список уведомлений; "
        "get_platform_events: ближайшие мероприятия и статус моей записи; get_my_discounts: мои скидки; "
        "Для списков можно указать offset только если число явно есть в текущем вопросе; иначе offset null. "
        "get_my_agenda: события agenda моего расписания; get_platform_announcements: объявления ASAP; "
        "get_my_search_history: моя история поиска; "
        "Для поиска участников платформы укажи platform_search в unsupported: эта операция пока недоступна. "
        "get_my_achievements: мои достижения; get_achievements_catalog: доступные достижения и награды; "
        "get_public_achievements: достижения другого участника с явно указанным nickname; "
        "get_public_profile: профиль другого участника, только если его ник явно приведён в вопросе; "
        "для всех публичных инструментов заполни nickname точной строкой из вопроса. "
        "Для остальных nickname должен быть null. Можно выбрать несколько инструментов. "
        "История содержит прежние вопросы пользователя и служит только контекстом текущего вопроса. "
        "Не запрашивай пользовательский токен или внутренний идентификатор."
    )
    try:
        # Do not leave a planner worker blocked on the answer model's much
        # longer network timeout after the request deadline has elapsed.
        planner_llm = llm.model_copy(update={"timeout": 25}) if hasattr(llm, "model_copy") else llm
        structured = planner_llm.with_structured_output(ReadPlan)
        context = "\n".join(history[-3:])
        user_input = f"Предыдущие вопросы:\n{context}\n\nТекущий вопрос:\n{question}" if context else question
        value = await asyncio.wait_for(asyncio.to_thread(invoke_with_retry, structured.invoke, [("system", system), ("human", user_input)], label="personal_plan", max_retries=0), timeout=30)
        plan = value if isinstance(value, ReadPlan) else ReadPlan.model_validate(value)
        return validate_read_plan(plan, "\n".join((*history[-3:], question)), current_question=question)
    except InvalidReadPlan:
        raise
    except Exception as exc:
        raise InvalidReadPlan("planning_failed") from exc
