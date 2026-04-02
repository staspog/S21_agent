import logging
import os
import time
import uuid

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Request
from langchain_core.messages import HumanMessage
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer

from .agent_graph import build_rag_graph, chunk_sources
from .embeddings import load_index_and_map


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
log = logging.getLogger("s21.agent")

load_dotenv()
api_key = os.getenv("GIGACHAT_API_KEY")
if not api_key:
    raise RuntimeError("GIGACHAT_API_KEY не найден в .env")

log.info("Загружаем модель эмбеддингов и FAISS индекс...")
model = SentenceTransformer("all-MiniLM-L6-v2")
index, chunks_map = load_index_and_map()
graph = build_rag_graph(model, index, chunks_map, api_key)
log.info("Готово к приёму запросов.")

app = FastAPI(title="S21 Agent API")


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
    question: str
    top_k: int = 3
    session_id: str | None = None
    include_sources: bool = False


class QueryResponse(BaseModel):
    answer: str
    session_id: str
    sources: list[str] | None = None


@app.post("/ask", response_model=QueryResponse)
def ask_question(request: QueryRequest):
    session_id = request.session_id or str(uuid.uuid4())
    q_len = len(request.question)
    log.info(
        "ask start session_id=%s top_k=%s question_len=%s include_sources=%s",
        session_id,
        request.top_k,
        q_len,
        request.include_sources,
    )
    t0 = time.perf_counter()
    try:
        log.info(
            "graph invoke: узлы START → retrieve → generate → END | thread_id=%s",
            session_id,
        )
        result = graph.invoke(
            {
                "messages": [HumanMessage(content=request.question)],
                "top_k": request.top_k,
            },
            {"configurable": {"thread_id": session_id}},
        )
        elapsed = time.perf_counter() - t0
        messages = result["messages"]
        last = messages[-1]
        answer = getattr(last, "content", "") or ""
        if not isinstance(answer, str):
            answer = str(answer)
        sources = None
        if request.include_sources:
            sources = chunk_sources(list(result.get("retrieved_chunks") or []))
        log.info(
            "ask done session_id=%s graph_s=%.3f answer_len=%s",
            session_id,
            elapsed,
            len(answer),
        )
        return QueryResponse(
            answer=answer, session_id=session_id, sources=sources
        )
    except Exception as e:
        log.exception("ask failed session_id=%s", session_id)
        raise HTTPException(status_code=500, detail=str(e)) from e
