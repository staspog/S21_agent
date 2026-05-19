"""FastAPI: один эндпоинт /ask поверх Rocket.Chat-агента."""

from __future__ import annotations

import asyncio
import logging
import os
import time
import uuid
from contextlib import asynccontextmanager
from pathlib import Path

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Request
from langchain_core.messages import HumanMessage
from langchain_gigachat.chat_models import GigaChat
from pydantic import BaseModel, Field

from .graph import build_rc_graph
from .llm_gate import get_default_gate
from .pipeline.rag_faiss import RagConfig, load_rag_assets
from .pipeline.rerank import warmup_reranker
from .rc.catalog import RoomCatalog
from .rc.client import RocketChatClient
from .rc.config import load_rc_config


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

load_dotenv()

api_key = os.getenv("GIGACHAT_API_KEY")
if not api_key:
    raise RuntimeError("GIGACHAT_API_KEY не найден в .env")

rc_cfg = load_rc_config()
rc_client = RocketChatClient(rc_cfg)
rc_catalog = RoomCatalog(rc_client, rc_cfg)
llm = GigaChat(
    credentials=api_key,
    verify_ssl_certs=False,
    model="GigaChat-2-Max",
    scope="GIGACHAT_API_CORP",
    timeout=600,
)
try:
    logging.info(f"available LLM models: {llm.get_models()}")
except Exception as e:
    logging.error(f"Error: {e}")
    raise e
get_default_gate()  # инициализация singleton до первой LLM-нагрузки
graph = build_rc_graph(
    llm=llm,
    rc_client=rc_client,
    rc_catalog=rc_catalog,
    rc_cfg=rc_cfg,
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


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Прогрев каталога и реранкера в фоне (не блокируем старт).
    async def _bg():
        try:
            await rc_catalog.get()
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
    # Сохраняем старое имя поля для совместимости с test_client.py — игнорируется.
    top_k: int | None = Field(default=None, exclude=True)


class QueryResponse(BaseModel):
    answer: str
    session_id: str
    sources: list[str] | None = None


@app.post("/ask", response_model=QueryResponse)
async def ask_question(request: QueryRequest):
    session_id = request.session_id or str(uuid.uuid4())
    log.info(
        "ask start session_id=%s question_len=%s include_sources=%s",
        session_id,
        len(request.question),
        request.include_sources,
    )
    t0 = time.perf_counter()
    try:
        result = await asyncio.wait_for(
            graph.ainvoke(
                {"messages": [HumanMessage(content=request.question)]},
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

        sources: list[str] | None = None
        if request.include_sources:
            srcs = list(result.get("final_sources") or [])
            seen: set[str] = set()
            sources = []
            for s in srcs:
                if s and s not in seen:
                    seen.add(s)
                    sources.append(s)

        log.info(
            "ask done session_id=%s graph_s=%.3f answer_len=%s sources=%s",
            session_id,
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
        log.exception("ask failed session_id=%s", session_id)
        raise HTTPException(status_code=500, detail=str(e)) from e


@app.get("/health")
async def health() -> dict:
    return {"status": "ok"}
