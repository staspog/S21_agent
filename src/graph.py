"""LangGraph: главный граф агента.

Поток:

    START
      ↓
    decompose_query    (LLM → 1..N подвопросов)
      ↓
    fetch_catalog      (REST, TTL-кэш)
      ↓
    fanout             (Send per subquery, динамический фан-аут)
      ↓
    per_subquery       (select_rooms → expand_query → search → fuse → rerank → threads)
      ↓ (Annotated[list, add] — fan-in)
    generate           (cross-subquery dedupe по mid; GigaChat ответ + источники)
      ↓
    END

Параллелизм по подвопросам — через `Send` API (langgraph.types.Send).
Reducer на ключе `evidence` (Annotated[list, operator.add]) аккуратно склеивает
результаты параллельных веток без потерь. Граф компилируется async — все узлы
с REST/CPU-инференсом написаны как `async def` и используются через `ainvoke`.
"""

from __future__ import annotations

import asyncio
import logging
import operator
from typing import Annotated, NotRequired

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage
from langchain_gigachat.chat_models import GigaChat
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, MessagesState, StateGraph
from langgraph.types import Send

from .pipeline.answer import generate_answer
from .pipeline.decompose import decompose_question
from .pipeline.expand_query import expand_query
from .pipeline.fuse import rrf_fuse
from .pipeline.rerank import hybrid_rerank
from .pipeline.search import search_rooms_x_queries
from .pipeline.select_rooms import select_rooms
from .pipeline.threads import expand_threads
from .rc.catalog import RoomCatalog
from .rc.client import RocketChatClient
from .rc.config import RocketChatConfig
from .rc.schemas import Evidence, Hit, Room

log = logging.getLogger("s21.graph")


class AgentState(MessagesState):
    """Состояние графа.

    Параллельные ветки пишут только в `evidence` (через reducer add); остальное
    пишется только последовательными узлами, что исключает гонки.
    """

    subqueries: NotRequired[list[str]]
    rooms_catalog: NotRequired[list[Room]]
    seed_rooms: NotRequired[list[Room]]
    evidence: Annotated[list[Evidence], operator.add]
    final_sources: NotRequired[list[str]]


def _last_human_text(messages: list[BaseMessage]) -> str:
    for msg in reversed(messages):
        if isinstance(msg, HumanMessage):
            c = msg.content
            return c if isinstance(c, str) else str(c)
    raise ValueError("В истории нет сообщения пользователя")


def build_rc_graph(
    *,
    llm: GigaChat,
    rc_client: RocketChatClient,
    rc_catalog: RoomCatalog,
    rc_cfg: RocketChatConfig,
):
    """Собрать и скомпилировать граф."""

    # ---------- nodes ----------

    async def node_decompose(state: AgentState) -> dict:
        log.info("graph: decompose start")
        question = _last_human_text(list(state["messages"]))
        # decompose_question — sync LLM-вызов, выполним в default executor чтобы не блокировать loop
        loop = asyncio.get_running_loop()
        subs = await loop.run_in_executor(
            None,
            lambda: decompose_question(
                llm=llm,
                question=question,
                max_subqueries=rc_cfg.max_subqueries,
            ),
        )
        return {"subqueries": subs}

    async def node_fetch_catalog(state: AgentState) -> dict:
        log.info("graph: fetch_catalog start")
        rooms = await rc_catalog.get()
        seeds = rc_catalog.seed_rooms()
        log.info(
            "graph: fetch_catalog end rooms=%s seeds=%s",
            len(rooms),
            len(seeds),
        )
        return {"rooms_catalog": rooms, "seed_rooms": seeds}

    def fanout(state: AgentState) -> list[Send]:
        subs: list[str] = list(state.get("subqueries") or [])
        rooms: list[Room] = list(state.get("rooms_catalog") or [])
        seeds: list[Room] = list(state.get("seed_rooms") or [])
        if not subs:
            log.warning("fanout: пустой список подвопросов — ничего не делаем")
            return []
        log.info("fanout: %s подвопросов в параллель", len(subs))
        return [
            Send(
                "per_subquery",
                {
                    "subquery": s,
                    "rooms_catalog": rooms,
                    "seed_rooms": seeds,
                },
            )
            for s in subs
        ]

    async def node_per_subquery(payload: dict) -> dict:
        subquery: str = payload["subquery"]
        catalog: list[Room] = payload.get("rooms_catalog") or []
        seeds: list[Room] = payload.get("seed_rooms") or []
        log.info("per_subquery: start q=%r", subquery)

        loop = asyncio.get_running_loop()

        rooms = await loop.run_in_executor(
            None,
            lambda: select_rooms(
                llm=llm,
                subquery=subquery,
                catalog=catalog,
                seed_rooms=seeds,
                target=rc_cfg.rooms_per_subquery,
            ),
        )
        if not rooms:
            log.warning("per_subquery: rooms пуст — пропускаем поиск")
            return {"evidence": [Evidence(subquery=subquery, hits=[])]}

        queries = await loop.run_in_executor(
            None,
            lambda: expand_query(
                llm=llm,
                subquery=subquery,
                target=rc_cfg.queries_per_subquery,
            ),
        )

        ranked_lists = await search_rooms_x_queries(
            client=rc_client,
            rooms=rooms,
            queries=queries,
            count=rc_cfg.search_count,
        )
        fused = rrf_fuse(ranked_lists)

        top: list[Hit] = await loop.run_in_executor(
            None,
            lambda: hybrid_rerank(
                subquery=subquery,
                hits=fused,
                model_name=rc_cfg.reranker_model,
                bm25_weight=rc_cfg.bm25_weight,
                top_n=rc_cfg.rerank_top_n,
            ),
        )

        rooms_by_rid: dict[str, Room] = {r.rid: r for r in rooms}
        with_threads = await expand_threads(
            client=rc_client,
            rooms_by_rid=rooms_by_rid,
            hits=top,
            top_n=rc_cfg.thread_expand_top,
            per_thread_count=rc_cfg.thread_messages_count,
        )

        log.info(
            "per_subquery: q=%r → %s evidence-hits", subquery, len(with_threads)
        )
        return {"evidence": [Evidence(subquery=subquery, hits=with_threads)]}

    def _dedupe_across_subqueries(ev_list: list[Evidence]) -> list[Evidence]:
        """Один и тот же mid в нескольких подвопросах оставляем только в первом."""
        seen: set[str] = set()
        out: list[Evidence] = []
        for ev in ev_list:
            kept: list[Hit] = []
            for h in ev.hits:
                if h.mid in seen:
                    continue
                seen.add(h.mid)
                kept.append(h)
            out.append(Evidence(subquery=ev.subquery, hits=kept))
        return out

    async def node_generate(state: AgentState) -> dict:
        log.info("graph: generate start")
        question = _last_human_text(list(state["messages"]))
        raw_ev = list(state.get("evidence") or [])
        ev_list = _dedupe_across_subqueries(raw_ev)
        history = list(state["messages"])[:-1]
        loop = asyncio.get_running_loop()
        answer_text, sources = await loop.run_in_executor(
            None,
            lambda: generate_answer(
                llm=llm,
                history=history,
                user_question=question,
                evidence_list=ev_list,
            ),
        )
        return {
            "messages": [AIMessage(content=answer_text)],
            "final_sources": sources,
        }

    # ---------- assembly ----------

    g: StateGraph = StateGraph(AgentState)
    g.add_node("decompose", node_decompose)
    g.add_node("fetch_catalog", node_fetch_catalog)
    g.add_node("per_subquery", node_per_subquery)
    g.add_node("generate", node_generate)

    g.add_edge(START, "decompose")
    g.add_edge("decompose", "fetch_catalog")
    g.add_conditional_edges("fetch_catalog", fanout, ["per_subquery"])
    g.add_edge("per_subquery", "generate")
    g.add_edge("generate", END)

    checkpointer = MemorySaver()
    return g.compile(checkpointer=checkpointer)
