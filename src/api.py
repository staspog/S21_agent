"""FastAPI: один эндпоинт /ask поверх Rocket.Chat-агента."""

from __future__ import annotations

import asyncio
import logging
import os
import time
import uuid
from contextlib import asynccontextmanager
from datetime import datetime, timedelta, timezone
from urllib.parse import urlsplit
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from dotenv import load_dotenv

load_dotenv()

from .pipeline.rerank import apply_cpu_threads

apply_cpu_threads()

from fastapi import FastAPI, HTTPException, Request
import httpx
from mcp import ClientSession
from mcp.client.streamable_http import streamable_http_client
from .deployment import DeploymentMiddleware
from .log_filters import install_health_access_filter, quiet_health_request
from langchain_core.messages import HumanMessage
from langchain_gigachat.chat_models import GigaChat
from pydantic import BaseModel, Field

from .graph import build_rc_graph
from .llm_gate import get_default_gate
from .personal_intent import agenda_bounds, calendar_bounds, personal_kind
from .personal_mcp import PlatformFailure, verified_subject
from .personal_planner import InvalidReadPlan, PAGED_TOOLS, plan_reads
from .session_identity import scoped_thread_id
from .pipeline.rag_faiss import RagConfig, load_rag_assets, rag_status
from .pipeline.rerank import warmup_reranker
from .rc.availability import RocketChatAvailability
from .rc.catalog import RoomCatalog
from .rc.client import RocketChatClient
from .rc.config import load_rc_config
from .rc.schemas import SourceItem


def _setup_logging() -> None:
    root = logging.getLogger("s21")
    root.setLevel(logging.INFO)
    if root.handlers:
        return
    handler = logging.StreamHandler()
    handler.setFormatter(
        logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s")
    )
    root.addHandler(handler)


_setup_logging()
logging.getLogger("httpx").setLevel(logging.WARNING)
install_health_access_filter()
log = logging.getLogger("s21.api")

api_key = os.getenv("GIGACHAT_API_KEY")
if not api_key:
    raise RuntimeError("GIGACHAT_API_KEY не найден в .env")

rc_cfg = load_rc_config()
rc_client = RocketChatClient(rc_cfg)
rc_catalog = RoomCatalog(rc_client, rc_cfg)
rc_availability = RocketChatAvailability(
    probe_timeout_s=rc_cfg.probe_timeout_s,
    cache_ttl_s=rc_cfg.probe_cache_ttl_s,
)
llm = GigaChat(
    credentials=api_key,
    verify_ssl_certs=False,
    model="GigaChat-2-Max",
    scope="GIGACHAT_API_CORP",
    timeout=600,
)
try:
    logging.info("GigaChat client initialized (models list on first /ask)")
except Exception as e:
    logging.error(f"Error: {e}")
    raise e
get_default_gate()
graph = build_rc_graph(
    llm=llm,
    rc_client=rc_client,
    rc_catalog=rc_catalog,
    rc_cfg=rc_cfg,
    rc_availability=rc_availability,
)
log.info("Граф собран. Reranker=%s", rc_cfg.reranker_model)

_ASK_TIMEOUT_RAW = os.environ.get("ASK_TIMEOUT_S", "300").strip()
try:
    ASK_TIMEOUT_S = float(_ASK_TIMEOUT_RAW)
except ValueError:
    ASK_TIMEOUT_S = 300.0
if ASK_TIMEOUT_S <= 0:
    ASK_TIMEOUT_S = 300.0
log.info("ASK_TIMEOUT_S=%s (504 если граф дольше)", ASK_TIMEOUT_S)

_ask_semaphore = asyncio.Semaphore(1)
_platform_semaphore = asyncio.Semaphore(5)


@asynccontextmanager
async def lifespan(app: FastAPI):
    async def _bg():
        try:
            ok, err = await rc_availability.check(rc_client)
            if ok:
                await asyncio.wait_for(
                    rc_catalog.get(),
                    timeout=rc_cfg.catalog_fetch_timeout_s,
                )
            else:
                log.warning("startup: RC unavailable (%s), skip catalog preload", err)
        except asyncio.TimeoutError:
            rc_availability.mark_unavailable(
                f"catalog timeout ({rc_cfg.catalog_fetch_timeout_s:.0f}s)"
            )
            log.warning("startup: каталог комнат не успел загрузиться")
        except Exception:
            log.exception("startup: каталог комнат не подгрузился, попробуем позже")
        try:
            await asyncio.get_running_loop().run_in_executor(
                None, lambda: warmup_reranker(rc_cfg.reranker_model)
            )
        except Exception:
            log.exception("startup: warmup реранкера упал")
        try:
            rag_cfg = RagConfig.from_env()
            await asyncio.get_running_loop().run_in_executor(
                None, lambda: load_rag_assets(rag_cfg)
            )
        except Exception:
            log.exception("startup: RAG assets preload failed")

    warmup_task = asyncio.create_task(_bg())
    log.info("API готов к приёму запросов")
    try:
        yield
    finally:
        warmup_task.cancel()
        try:
            await warmup_task
        except asyncio.CancelledError:
            pass
        await rc_client.aclose()
        log.info("API остановлен, RC-клиент закрыт")


app = FastAPI(title="S21 RC Agent API", lifespan=lifespan)
app.add_middleware(DeploymentMiddleware)


@app.middleware("http")
async def log_http_requests(request: Request, call_next):
    start = time.perf_counter()
    response = await call_next(request)
    elapsed = time.perf_counter() - start
    if not quiet_health_request(request.method, request.url.path, response.status_code):
        log.info(
            "%s %s -> %s in %.3fs",
            request.method,
            request.url.path,
            response.status_code,
            elapsed,
        )
    return response


class QueryRequest(BaseModel):
    question: str = Field(min_length=1)
    session_id: str | None = None
    include_sources: bool = True
    deep_search: bool = False
    rocket_search: bool = False
    timezone: str = "Europe/Moscow"


class QueryResponse(BaseModel):
    answer: str
    session_id: str
    sources: list[SourceItem] | None = None


@app.post("/ask", response_model=QueryResponse)
async def ask_question(request: QueryRequest, http_request: Request):
    authorization = http_request.headers.get("authorization", "")
    scheme, separator, token = authorization.partition(" ")
    if not separator or scheme.lower() != "bearer" or not token.strip():
        raise HTTPException(status_code=401, detail="authentication_required")
    try:
        ZoneInfo(request.timezone)
    except (ZoneInfoNotFoundError, ValueError):
        raise HTTPException(status_code=400, detail="invalid_timezone") from None
    api_base = os.getenv("S21_SERVER_API_BASE", "https://s21-api.gabrielg.ru").rstrip("/")
    parsed_api_base = urlsplit(api_base)
    api_host = (parsed_api_base.hostname or "").lower().rstrip(".")
    if (parsed_api_base.scheme != "https" or not parsed_api_base.hostname
            or parsed_api_base.username or parsed_api_base.password
            or api_host in {"platform.21-school.ru", "auth.21-school.ru"}
            or api_host.endswith(".platform.21-school.ru")):
        raise HTTPException(status_code=503, detail="identity_service_misconfigured")
    personal_context = None
    session_id = request.session_id or str(uuid.uuid4())
    try:
        async with asyncio.timeout(65), _platform_semaphore, httpx.AsyncClient(base_url=api_base, timeout=30.0, follow_redirects=False, headers={"Authorization": f"Bearer {token.strip()}"}) as client:
            subject = await verified_subject(client, token.strip())
            thread_id = scoped_thread_id(subject, session_id)
            history = ()
            if request.session_id and hasattr(graph, "aget_state"):
                snapshot = await graph.aget_state({"configurable": {"thread_id": thread_id}})
                messages = snapshot.values.get("messages", []) if snapshot and snapshot.values else []
                history = tuple(str(m.content)[:600] for m in messages if isinstance(m, HumanMessage))[-3:]
            plan = await plan_reads(llm, request.question, history)
            if plan.unsupported:
                return QueryResponse(answer="Поиск участников платформы пока недоступен. Открытый профиль можно запросить по точному нику.", session_id=request.session_id or str(uuid.uuid4()), sources=[])
            if not plan.calls and personal_kind(request.question) is not None:
                raise InvalidReadPlan("personal_read_not_selected")
            if plan.calls:
                async with streamable_http_client(f"{api_base}/v1/assistant/mcp", http_client=client, terminate_on_close=False) as (read, write, _), ClientSession(read, write) as mcp:
                    await mcp.initialize()
                    results = []
                    kinds = {"get_my_calendar":"calendar", "get_my_projects":"projects", "get_my_project_details":"project_details", "get_review_slots":"review_slots", "get_my_profile":"profile", "get_my_xp_history":"xp_history", "get_my_coalition":"coalition", "get_my_campus_time":"campus_time", "get_public_profile":"public_profile", "get_public_xp_history":"public_xp_history", "get_public_coalition":"public_coalition", "get_public_campus_time":"public_campus_time", "get_my_notification_count":"notification_count", "get_my_discounts":"discounts", "get_my_notifications":"notifications", "get_platform_events":"events", "get_my_agenda":"agenda", "get_platform_announcements":"announcements", "get_my_search_history":"search_history", "get_my_achievements":"achievements", "get_public_achievements":"public_achievements", "get_achievements_catalog":"achievements_catalog"}
                    for call in plan.calls:
                        args = {}
                        if call.name in ("get_my_calendar", "get_my_agenda", "get_review_slots"):
                            try:
                                bounds = calendar_bounds if call.name == "get_my_calendar" else agenda_bounds
                                start, end = bounds(request.question, request.timezone, datetime.now(timezone.utc))
                            except ValueError:
                                return QueryResponse(answer="Пока я могу проверить календарь за сегодня, завтра или за неделю. Уточни период.", session_id=request.session_id or str(uuid.uuid4()), sources=[])
                            args = {"start": start, "end": end}
                            args["timezone_name"] = request.timezone
                            if call.name == "get_review_slots" and call.project_name:
                                args["project_name"] = call.project_name
                        elif call.name == "get_my_project_details":
                            args = {"project_name": call.project_name} if call.project_name else {}
                        elif call.name in ("get_public_profile", "get_public_achievements", "get_public_xp_history", "get_public_coalition"):
                            args = {"login": call.nickname}
                        elif call.name == "get_platform_events":
                            args = {"timezone_name": request.timezone}
                        elif call.name in ("get_my_campus_time", "get_public_campus_time"):
                            try:
                                start, _ = calendar_bounds(request.question, request.timezone, datetime.now(timezone.utc))
                            except ValueError:
                                return QueryResponse(answer="Уточните неделю для просмотра времени в кампусе.", session_id=request.session_id or str(uuid.uuid4()), sources=[])
                            date = datetime.fromisoformat(start).date()
                            args = {"week_start": (date - timedelta(days=date.weekday())).isoformat()}
                            if call.name == "get_public_campus_time":
                                args["login"] = call.nickname
                        if call.name in PAGED_TOOLS and call.offset is not None:
                            args["offset"] = call.offset
                        tool_result = await mcp.call_tool(call.name, args)
                        if tool_result.isError or not isinstance(tool_result.structuredContent, dict):
                            raise PlatformFailure("unavailable")
                        context = {"kind": kinds[call.name], "timezone": request.timezone, **tool_result.structuredContent}
                        if context.get("status") == "ambiguous":
                            return QueryResponse(answer="Уточните, для какого проекта нужна информация.", session_id=request.session_id or str(uuid.uuid4()), sources=[])
                        if context.get("status") in ("unauthorized", "forbidden"):
                            raise PlatformFailure(context["status"])
                        if context.get("status") != "ok":
                            log.warning("personal context unavailable kind=%s", kinds[call.name])
                        results.append(context)
                    personal_context = results[0] if len(results) == 1 else {"kind": "multi", "status": "ok" if all(r.get("status") == "ok" for r in results) else "unavailable", "timezone": request.timezone, "results": results}
    except InvalidReadPlan as exc:
        if str(exc) in ("missing_nickname", "nickname_not_explicit"):
            return QueryResponse(answer="Укажите точный ник участника, чей профиль нужно посмотреть.", session_id=request.session_id or str(uuid.uuid4()), sources=[])
        if str(exc) == "project_not_explicit":
            return QueryResponse(answer="Уточните название проекта.", session_id=request.session_id or str(uuid.uuid4()), sources=[])
        if str(exc) == "offset_not_explicit":
            return QueryResponse(answer="Уточните, с какого номера показать записи.", session_id=request.session_id or str(uuid.uuid4()), sources=[])
        log.warning("personal read planning unavailable")
        raise HTTPException(status_code=503, detail="planning_unavailable") from None
    except PlatformFailure as exc:
        status = {"unauthorized": 401, "forbidden": 403}.get(exc.code, 503)
        log.warning("platform request failed category=%s", exc.code)
        raise HTTPException(status_code=status, detail=exc.code) from None
    except Exception as exc:
        # MCP transport errors may be wrapped in an exception group. Do not
        # expose their contents: they may include request headers or payloads.
        pending = [exc]
        while pending:
            error = pending.pop()
            if isinstance(error, PlatformFailure):
                status = {"unauthorized": 401, "forbidden": 403}.get(error.code, 503)
                raise HTTPException(status_code=status, detail=error.code) from None
            if isinstance(error, httpx.HTTPStatusError) and error.response.status_code in (401, 403):
                raise HTTPException(status_code=error.response.status_code, detail="platform_authorization_failed") from None
            if isinstance(error, BaseExceptionGroup):
                pending.extend(error.exceptions)
        log.warning("personal MCP transport unavailable")
        raise HTTPException(status_code=503, detail="personal_context_unavailable") from None
    request_id = str(uuid.uuid4())
    deep = request.deep_search
    rocket = request.rocket_search
    if deep and not rocket:
        deep = False
    log.info(
        "ask start request_id=%s question_len=%s deep=%s rocket=%s include_sources=%s",
        request_id,
        len(request.question),
        deep,
        rocket,
        request.include_sources,
    )
    t0 = time.perf_counter()
    try:
        async with _ask_semaphore:
            result = await asyncio.wait_for(
                graph.ainvoke(
                    {
                        "messages": [HumanMessage(content=request.question)],
                        "deep_search": deep,
                        "rocket_search": rocket,
                        "personal_context": personal_context,
                    },
                    {"configurable": {"thread_id": thread_id}},
                ),
                timeout=ASK_TIMEOUT_S,
            )
        elapsed = time.perf_counter() - t0
        messages = result["messages"]
        last = messages[-1]
        answer = getattr(last, "content", "") or ""
        if not isinstance(answer, str):
            answer = str(answer)

        sources: list[SourceItem] | None = None
        if request.include_sources:
            srcs = list(result.get("final_sources") or [])
            seen: set[str] = set()
            sources = []
            for s in srcs:
                item = SourceItem.model_validate(s)
                if not item.url or item.url in seen:
                    continue
                seen.add(item.url)
                if not item.label:
                    item = SourceItem(url=item.url, label=item.url)
                sources.append(item)

        log.info(
            "ask done request_id=%s deep=%s rocket=%s graph_s=%.3f answer_len=%s sources=%s",
            request_id,
            deep,
            rocket,
            elapsed,
            len(answer),
            len(sources or []),
        )
        return QueryResponse(answer=answer, session_id=session_id, sources=sources)
    except TimeoutError:
        elapsed = time.perf_counter() - t0
        log.warning(
            "ask timeout request_id=%s limit_s=%.3f elapsed_s=%.3f",
            request_id,
            ASK_TIMEOUT_S,
            elapsed,
        )
        raise HTTPException(
            status_code=504,
            detail="Ask processing timed out",
        ) from None
    except Exception as e:
        log.error(
            "ask failed request_id=%s error_type=%s deep=%s rocket=%s",
            request_id,
            type(e).__name__,
            deep,
            rocket,
        )
        raise HTTPException(status_code=500, detail="ask_failed") from None


@app.get("/health")
async def health() -> dict:
    rc_ok, rc_err = await rc_availability.check(rc_client)
    status: dict = {
        "status": "ok",
        "rocket_chat_available": rc_ok,
    }
    if rc_err:
        status["rocket_chat_error"] = rc_err
    status.update(rag_status())
    return status
