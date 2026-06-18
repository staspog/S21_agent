"""Pydantic v2 схемы для Rocket.Chat пайплайна.

Здесь лежат все типы, которые ходят между узлами LangGraph, а также
структуры для structured output LLM (decompose, expand_query).
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator


RoomKind = Literal["channel", "group", "im"]


class Room(BaseModel):
    """Комната Rocket.Chat (канал, приватная группа или DM)."""

    model_config = ConfigDict(frozen=True)

    rid: str = Field(description="Внутренний идентификатор комнаты")
    name: str = Field(description="Имя комнаты (без #)")
    kind: RoomKind = Field(description="Тип: channel | group | im")
    topic: str = Field(default="", description="Topic/описание комнаты")

    @field_validator("rid", "name")
    @classmethod
    def _strip(cls, v: str) -> str:
        s = (v or "").strip()
        if not s:
            raise ValueError("Field must not be empty")
        return s


class Hit(BaseModel):
    """Нормализованное сообщение Rocket.Chat — единица evidence для ответа."""

    model_config = ConfigDict(frozen=False)

    mid: str = Field(description="Rocket.Chat message id")
    rid: str = Field(description="Идентификатор комнаты")
    room_name: str = Field(description="Имя комнаты")
    room_kind: RoomKind = Field(default="group")
    msg: str = Field(default="", description="Текст сообщения")
    ts: str = Field(default="", description="ISO timestamp")
    user: str = Field(default="", description="Username автора")
    rc_score: float | None = Field(default=None, description="score от chat.search")
    bm25_score: float | None = None
    ce_score: float | None = None
    final_score: float | None = None
    thread_root_mid: str | None = Field(
        default=None,
        description="Для реплая — id корневого сообщения треда",
    )
    is_thread_reply: bool = False
    permalink: str = Field(default="", description="Веб-ссылка на сообщение")


class SubqueryPlan(BaseModel):
    """Structured output: 1..N подвопросов, на которые LLM раскладывает запрос."""

    subqueries: list[str] = Field(
        default_factory=list,
        description=(
            "Список самостоятельных подвопросов (1..N). "
            "Каждый — короткое утверждение/вопрос на русском языке."
        ),
    )

    @field_validator("subqueries")
    @classmethod
    def _normalize(cls, v: list[str]) -> list[str]:
        out: list[str] = []
        seen: set[str] = set()
        for x in v or []:
            s = (x or "").strip().strip('"').strip("'").strip()
            if not s:
                continue
            key = s.lower()
            if key in seen:
                continue
            seen.add(key)
            out.append(s)
        return out[:5]


class SearchIntent(BaseModel):
    """Structured output: search intent из истории диалога."""

    search_query: str = Field(
        description=(
            "Самодостаточный поисковый запрос на русском (1–2 коротких предложения "
            "или фраза), отражающий ЧТО искать. Без мета-команд «поищи в рокете». "
            "Сохраняй сущности: кампус, проект, язык, событие, процессы Школы 21. "
            "Для оперативных вопросов про кампус укажи, какой блок дайджеста или "
            "тип объявления нужен (логистика, мероприятия, клубы…)."
        )
    )
    scope: Literal["today", "general"] = Field(
        default="general",
        description=(
            "today — вопрос про оперативное **сейчас/сегодня** в кампусе "
            "(кластеры, логистика, «что в кампусе», расписание дня) без другой явной даты; "
            "general — справочные/общие темы или явно другая дата"
        ),
    )


class SearchQueries(BaseModel):
    """Structured output: короткие строки для chat.search + временной scope."""

    scope: Literal["today", "general"] = Field(
        default="general",
        description=(
            "today — подвопрос про оперативное в кампусе на **сегодня** "
            "(см. блок «Сегодня» и «Оперативные объявления кампуса»); "
            "general — иначе"
        ),
    )
    queries: list[str] = Field(
        default_factory=list,
        description=(
            "2–4 короткие поисковые строки для Rocket.Chat. "
            "Фразы должны встречаться в реальных постах ADM (заголовки блоков дайджеста, "
            "даты, `on:ДД/ММ/ГГГГ`). Сформулируй **сам** по смыслу подвопроса — "
            "не копируй длинный вопрос пользователя."
        ),
    )

    @field_validator("queries")
    @classmethod
    def _normalize(cls, v: list[str]) -> list[str]:
        out: list[str] = []
        seen: set[str] = set()
        for x in v or []:
            s = (x or "").strip().strip('"').strip("'").strip()
            if not s:
                continue
            key = s.lower()
            if key in seen:
                continue
            seen.add(key)
            out.append(s)
        return out[:6]


class RagChunkMetadata(BaseModel):
    """Метаданные одного RAG-чанка из jsonl/FAISS."""

    model_config = ConfigDict(extra="allow")

    source: str = ""
    url: str | None = None
    page_title: str = ""
    section: str = ""
    slug: str = ""
    part: int | None = None
    parts: int | None = None
    chunk_file: str = ""
    rag_ce_score: float | None = None


class RagChunk(BaseModel):
    """Один чанк RAG-корпуса (adm_info / FAQ)."""

    model_config = ConfigDict(extra="allow")

    id: str
    text: str
    metadata: RagChunkMetadata = Field(default_factory=RagChunkMetadata)

    @classmethod
    def from_record(cls, raw: dict[str, Any]) -> RagChunk:
        data = dict(raw or {})
        meta = data.pop("metadata", None) or {}
        if not isinstance(meta, dict):
            meta = {}
        chunk_id = str(data.pop("id", "") or "")
        text = str(data.pop("text", "") or "")
        return cls(
            id=chunk_id,
            text=text,
            metadata=RagChunkMetadata.model_validate(meta),
            **data,
        )

    def to_record(self) -> dict[str, Any]:
        return self.model_dump(exclude_none=True)


class SourceItem(BaseModel):
    """Источник ответа: permalink Rocket.Chat и человекочитаемая подпись."""

    model_config = ConfigDict(frozen=True)

    url: str
    label: str


class Evidence(BaseModel):
    """Доказательная база для одного подвопроса (после rerank-а и треда)."""

    model_config = ConfigDict(frozen=False)

    subquery: str
    hits: list[Hit] = Field(default_factory=list)


class FinalAnswer(BaseModel):
    """Structured output финальной генерации.

    Разделяем «то, что показываем пользователю» и «то, по чему сервер строит блок
    Источники». Так пользователь видит чистый markdown без технических меток
    `[rc:<mid>]`, а сервер знает, какие сообщения цитировались, и сам подставляет
    блок «## Источники».
    """

    answer_markdown: str = Field(
        description=(
            "Markdown-ответ на русском языке для пользователя.\n"
            "СТРОГО:\n"
            "  • НЕ вставляй технические метки вида [rc:<mid>], [rc=<mid>], "
            "[mid:<...>] и подобные;\n"
            "  • НЕ добавляй блок 'Источники' / 'Ссылки' / 'Permalink' — "
            "сервер подставит его сам;\n"
            "  • не дублируй текст EVIDENCE дословно длинными цитатами — "
            "пересказывай;\n"
            "  • если для подвопроса в EVIDENCE нет данных — честно скажи "
            "'в доступных сообщениях нет данных'."
        )
    )
    cited_mids: list[str] = Field(
        default_factory=list,
        description=(
            "Список mid сообщений из EVIDENCE, на которые опирается ответ. "
            "Только реальные mid, встречающиеся в EVIDENCE. Без выдуманных. "
            "Порядок — по релевантности/использованию в ответе."
        ),
    )

    @field_validator("cited_mids")
    @classmethod
    def _normalize_mids(cls, v: list[str]) -> list[str]:
        out: list[str] = []
        seen: set[str] = set()
        for x in v or []:
            s = (x or "").strip()
            if not s or s in seen:
                continue
            seen.add(s)
            out.append(s)
        return out
