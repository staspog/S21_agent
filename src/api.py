"""FastAPI: один эндпоинт /ask поверх Rocket.Chat-агента."""

from __future__ import annotations

import asyncio
import logging
import os
import time
import uuid
from contextlib import asynccontextmanager

from dotenv import load_dotenv

load_dotenv()

from .pipeline.rerank import apply_cpu_threads

apply_cpu_threads()

from fastapi import FastAPI, HTTPException, Request
from langchain_core.messages import HumanMessage
from langchain_gigachat.chat_models import GigaChat
from pydantic import BaseModel, Field

from .graph import build_rc_graph
from .llm_gate import get_default_gate
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

    asyncio.create_task(_bg())
    log.info("API готов к приёму запросов")
    try:
        yield
    finally:
        await rc_client.aclose()
        log.info("API остановлен, RC-клиент закрыт")


app = FastAPI(title="S21 RC Agent API", lifespan=lifespan)


@app.middleware("http")
async def log_http_requests(request: Request, call_next):
    start = time.perf_counter()
    response = await call_next(request)
    elapsed = time.perf_counter() - start
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


class QueryResponse(BaseModel):
    answer: str
    session_id: str
    sources: list[SourceItem] | None = None


@app.post("/ask", response_model=QueryResponse)
async def ask_question(request: QueryRequest):
    session_id = request.session_id or str(uuid.uuid4())
    deep = request.deep_search
    rocket = request.rocket_search
    if deep and not rocket:
        deep = False
    log.info(
        "ask start session_id=%s question_len=%s deep=%s rocket=%s include_sources=%s",
        session_id,
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
                    },
                    {"configurable": {"thread_id": session_id}},
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
            "ask done session_id=%s deep=%s rocket=%s graph_s=%.3f answer_len=%s sources=%s",
            session_id,
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
            "ask timeout session_id=%s limit_s=%.3f elapsed_s=%.3f",
            session_id,
            ASK_TIMEOUT_S,
            elapsed,
        )
        raise HTTPException(
            status_code=504,
            detail="Ask processing timed out",
        ) from None
    except Exception as e:
        log.exception(
            "ask failed session_id=%s question=%r deep=%s rocket=%s",
            session_id,
            request.question[:120],
            deep,
            rocket,
        )
        raise HTTPException(status_code=500, detail=str(e) or type(e).__name__) from e


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
