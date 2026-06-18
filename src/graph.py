"""LangGraph: главный граф агента.

Поток:

    START
      ↓
    prepare            (RAG + catalog параллельно)
      ↓
    [rag_first?]  → generate (каталог clubs/benefits, skip RC)
      ↓ else [rocket_search?]
      ↓ no                          ↓ yes
    generate (RAG-only)         decompose → fanout → per_subquery → generate
      ↓                               ↓
    END                              END
"""

from __future__ import annotations

import asyncio
import logging
import operator
import time
from typing import Annotated, NotRequired

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage
from langchain_gigachat.chat_models import GigaChat
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, MessagesState, StateGraph
from langgraph.types import Overwrite, Send

from .campus_intent import is_broad_campus_today
from .followup_intent import resolve_followup_topic
from .reference_intent import detect_campus, detect_reference_topic, resolve_reference_context
from .pipeline.answer import generate_answer
from .pipeline.decompose import decompose_question
from .pipeline.expand_query import expand_query, keyword_queries_for_search
from .pipeline.fuse import rrf_fuse
from .pipeline.rag_faiss import RagConfig, retrieve_rag_chunks
from .pipeline.rerank import hybrid_rerank
from .pipeline.resolve_query import resolve_search_intent
from .pipeline.search import search_rooms_x_queries
from .pipeline.select_rooms import adm_rooms_subset, select_rooms
from .pipeline.threads import expand_threads
from .rc.availability import RocketChatAvailability
from .rc.catalog import RoomCatalog
from .rc.client import RocketChatClient
from .rc.config import RocketChatConfig
from .rc.schemas import Evidence, Hit, RagChunk, Room, SourceItem

log = logging.getLogger("s21.graph")


class AgentState(MessagesState):
    """Состояние графа."""

    deep_search: NotRequired[bool]
    rocket_search: NotRequired[bool]
    rc_unavailable: NotRequired[bool]
    subqueries: NotRequired[list[str]]
    search_intent: NotRequired[str]
    search_scope: NotRequired[str]
    rag_chunks: NotRequired[list[dict]]
    reference_topic: NotRequired[str | None]
    reference_campus: NotRequired[str | None]
    rag_first: NotRequired[bool]
    rooms_catalog: NotRequired[list[Room]]
    seed_rooms: NotRequired[list[Room]]
    evidence: Annotated[list[Evidence], operator.add]
    final_sources: NotRequired[list[dict[str, str]]]


def _rag_chunks_from_state(raw: list | None) -> list[RagChunk]:
    return [RagChunk.from_record(d) for d in (raw or [])]


def _rag_chunks_to_state(chunks: list[RagChunk]) -> list[dict]:
    return [c.to_record() for c in chunks]


def _last_human_text(messages: list[BaseMessage]) -> str:
    for msg in reversed(messages):
        if isinstance(msg, HumanMessage):
            c = msg.content
            return c if isinstance(c, str) else str(c)
    raise ValueError("В истории нет сообщения пользователя")


def _log_node_done(name: str, t0: float, **extra: object) -> None:
    parts = " ".join(f"{k}={v}" for k, v in extra.items())
    log.info("graph: %s done in %.3fs%s", name, time.perf_counter() - t0, f" {parts}" if parts else "")


def build_rc_graph(
    *,
    llm: GigaChat,
    rc_client: RocketChatClient,
    rc_catalog: RoomCatalog,
    rc_cfg: RocketChatConfig,
    rc_availability: RocketChatAvailability,
):
    """Собрать и скомпилировать граф."""

    async def node_prepare(state: AgentState) -> dict:
        t0 = time.perf_counter()
        log.info("graph: prepare start")
        messages = list(state["messages"])
        loop = asyncio.get_running_loop()

        async def _rag_and_intent() -> tuple[list[RagChunk], str, str, str | None, str | None]:
            last_q = _last_human_text(messages)
            rag_query = resolve_followup_topic(messages) or last_q
            ref_topic = detect_reference_topic(rag_query)
            ref_campus = detect_campus(rag_query) if ref_topic else None
            rag_cfg = RagConfig.from_env()

            def _retrieve(q: str) -> list[RagChunk]:
                return retrieve_rag_chunks(
                    query=q,
                    cfg=rag_cfg,
                    reranker_model_name=rc_cfg.reranker_model,
                    reference_topic=ref_topic,
                    reference_campus=ref_campus,
                )

            rag_chunks = await loop.run_in_executor(None, lambda: _retrieve(rag_query))
            intent, scope = await loop.run_in_executor(
                None,
                lambda: resolve_search_intent(
                    llm=llm,
                    messages=messages,
                    rag_chunks=rag_chunks,
                ),
            )
            if intent and intent.strip().lower() != last_q.strip().lower():
                rag_chunks = await loop.run_in_executor(None, lambda q=intent: _retrieve(q))
            return rag_chunks, intent, scope, ref_topic, ref_campus

        rag_chunks, search_intent, search_scope, reference_topic, reference_campus = await _rag_and_intent()
        topic, campus, rag_first = resolve_reference_context(
            _last_human_text(messages),
            rag_chunks,
            bool(state.get("rocket_search")),
        )
        if campus and not reference_campus:
            reference_campus = campus
        if topic and not reference_topic:
            reference_topic = topic
        rooms: list[Room] = []
        seeds: list[Room] = []
        rocket = bool(state.get("rocket_search"))
        if rag_first:
            log.info(
                "graph: prepare rag_first skip RC topic=%s campus=%s",
                reference_topic,
                reference_campus,
            )
            rocket = False

        rc_unavailable = False

        if rocket:
            ok, err = await rc_availability.check(rc_client)
            if not ok:
                log.warning(
                    "graph: prepare RC unavailable (%s) — fallback to RAG-only",
                    err,
                )
                rocket = False
                rc_unavailable = True
            else:
                try:
                    rooms = await asyncio.wait_for(
                        rc_catalog.get(),
                        timeout=rc_cfg.catalog_fetch_timeout_s,
                    )
                except asyncio.TimeoutError:
                    msg = f"catalog timeout ({rc_cfg.catalog_fetch_timeout_s:.0f}s)"
                    log.warning("graph: prepare %s — fallback to RAG-only", msg)
                    rc_availability.mark_unavailable(msg)
                    rocket = False
                    rc_unavailable = True
                    rooms = []
                except Exception as e:
                    msg = str(e) or type(e).__name__
                    log.warning(
                        "graph: prepare catalog failed (%s) — fallback to RAG-only",
                        msg,
                    )
                    rc_availability.mark_unavailable(msg)
                    rocket = False
                    rc_unavailable = True
                    rooms = []
                else:
                    rc_availability.mark_available()
                    seeds = rc_catalog.seed_rooms()
        else:
            log.info("graph: prepare skip RC catalog (rocket_search=false)")

        rag_slugs = sorted(
            {
                ch.metadata.slug.strip()
                for ch in rag_chunks
                if ch.metadata.slug
            }
        )
        _log_node_done(
            "prepare",
            t0,
            rag_chunks=len(rag_chunks),
            search_intent=search_intent[:80] if search_intent else "",
            search_scope=search_scope,
            rooms=len(rooms),
            seeds=len(seeds),
            rocket_search=rocket,
            rc_unavailable=rc_unavailable,
            reference_topic=reference_topic or "",
            reference_campus=reference_campus or "",
            rag_first=rag_first,
            rag_slugs=",".join(rag_slugs[:5]),
        )
        out: dict = {
            "rag_chunks": _rag_chunks_to_state(rag_chunks),
            "search_intent": search_intent,
            "search_scope": search_scope,
            "rooms_catalog": rooms,
            "seed_rooms": seeds,
            "reference_topic": reference_topic,
            "reference_campus": reference_campus,
            "rag_first": rag_first,
            "evidence": Overwrite([]),
        }
        if rc_unavailable or rag_first:
            out["rocket_search"] = False
        if rc_unavailable:
            out["rc_unavailable"] = True
        return out

    def route_after_prepare(state: AgentState) -> str:
        if state.get("rag_first"):
            return "generate"
        if state.get("rocket_search"):
            return "decompose"
        return "generate"

    async def node_decompose(state: AgentState) -> dict:
        t0 = time.perf_counter()
        log.info("graph: decompose start")
        search_intent = (state.get("search_intent") or "").strip()
        if not search_intent:
            search_intent = _last_human_text(list(state["messages"]))
        rag_chunks = _rag_chunks_from_state(state.get("rag_chunks"))
        search_scope = (state.get("search_scope") or "general").strip()
        deep = bool(state.get("deep_search"))
        profile = rc_cfg.profile_for(deep=deep)

        if (
            search_scope == "today"
            and is_broad_campus_today(search_intent)
        ):
            subs = [search_intent]
            log.info("graph: decompose broad campus today → 1 subquery")
        elif not deep:
            subs = [search_intent]
            log.info("graph: decompose fast-path → 1 subquery from intent")
        else:
            loop = asyncio.get_running_loop()
            subs = await loop.run_in_executor(
                None,
                lambda: decompose_question(
                    llm=llm,
                    question=search_intent,
                    max_subqueries=profile.max_subqueries,
                    rag_chunks=rag_chunks,
                ),
            )
        _log_node_done("decompose", t0, subqueries=len(subs), intent=search_intent[:60])
        return {"subqueries": subs}

    def fanout(state: AgentState) -> list[Send]:
        subs: list[str] = list(state.get("subqueries") or [])
        rooms: list[Room] = list(state.get("rooms_catalog") or [])
        seeds: list[Room] = list(state.get("seed_rooms") or [])
        rag_chunks: list[dict] = list(state.get("rag_chunks") or [])
        search_scope = (state.get("search_scope") or "general").strip()
        deep = bool(state.get("deep_search"))
        if not subs:
            log.warning("fanout: пустой список подвопросов — ничего не делаем")
            return []
        log.info("fanout: %s подвопросов deep=%s", len(subs), deep)
        return [
            Send(
                "per_subquery",
                {
                    "subquery": s,
                    "rooms_catalog": rooms,
                    "seed_rooms": seeds,
                    "deep_search": deep,
                    "rag_chunks": rag_chunks,
                    "search_scope": search_scope,
                },
            )
            for s in subs
        ]

    async def node_per_subquery(payload: dict) -> dict:
        t0 = time.perf_counter()
        subquery: str = payload["subquery"]
        catalog: list[Room] = payload.get("rooms_catalog") or []
        seeds: list[Room] = payload.get("seed_rooms") or []
        rag_chunks = _rag_chunks_from_state(payload.get("rag_chunks"))
        search_scope = (payload.get("search_scope") or "general").strip()
        deep = bool(payload.get("deep_search"))
        profile = rc_cfg.profile_for(deep=deep)
        log.info("per_subquery: start q=%r deep=%s", subquery, deep)

        loop = asyncio.get_running_loop()

        rooms = await loop.run_in_executor(
            None,
            lambda: select_rooms(
                subquery=subquery,
                catalog=catalog,
                seed_rooms=seeds,
                target=profile.rooms_per_subquery,
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
                target=profile.queries_per_subquery,
                rag_chunks=rag_chunks,
                intent_scope=search_scope,
            ),
        )

        keyword_queries = keyword_queries_for_search(subquery, queries)
        adm_rooms = adm_rooms_subset(rooms)
        ranked_lists: list[list[Hit]] = []
        phase = "full"

        if adm_rooms and keyword_queries:
            ranked_lists = await search_rooms_x_queries(
                client=rc_client,
                rooms=adm_rooms,
                queries=keyword_queries,
                count=profile.search_count,
            )
            fused = rrf_fuse(ranked_lists)
            phase_a_hits = len(fused)
            log.info(
                "per_subquery: phase_a rooms=%s queries=%s hits=%s",
                [r.name for r in adm_rooms],
                keyword_queries,
                phase_a_hits,
            )
            if phase_a_hits:
                phase = "adm_keywords"
            else:
                phase_b_rooms = rooms[: min(4, len(rooms))]
                ranked_lists = await search_rooms_x_queries(
                    client=rc_client,
                    rooms=phase_b_rooms,
                    queries=keyword_queries,
                    count=profile.search_count,
                )
                fused = rrf_fuse(ranked_lists)
                phase = "keywords_4rooms"
        else:
            ranked_lists = await search_rooms_x_queries(
                client=rc_client,
                rooms=rooms,
                queries=queries,
                count=profile.search_count,
            )
            fused = rrf_fuse(ranked_lists)

        top: list[Hit] = await loop.run_in_executor(
            None,
            lambda: hybrid_rerank(
                subquery=subquery,
                hits=fused,
                model_name=rc_cfg.reranker_model,
                bm25_weight=profile.bm25_weight,
                top_n=profile.rerank_top_n,
                rerank_max_candidates=profile.rerank_max_candidates,
                rerank_mode=profile.rerank_mode,
            ),
        )

        rooms_by_rid: dict[str, Room] = {r.rid: r for r in rooms}
        with_threads = await expand_threads(
            client=rc_client,
            rooms_by_rid=rooms_by_rid,
            hits=top,
            top_n=profile.thread_expand_top,
            per_thread_count=profile.thread_messages_count,
        )

        _log_node_done(
            "per_subquery",
            t0,
            q=subquery[:40],
            hits=len(with_threads),
            mode=profile.rerank_mode,
            phase=phase,
            rooms=[r.name for r in rooms],
            queries=queries,
        )
        return {"evidence": [Evidence(subquery=subquery, hits=with_threads)]}

    def _dedupe_across_subqueries(ev_list: list[Evidence]) -> list[Evidence]:
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
        t0 = time.perf_counter()
        log.info("graph: generate start")
        question = _last_human_text(list(state["messages"]))
        rocket = bool(state.get("rocket_search"))
        rc_unavailable = bool(state.get("rc_unavailable"))
        rag_only = not rocket
        raw_ev = list(state.get("evidence") or [])
        ev_list = _dedupe_across_subqueries(raw_ev) if rocket else []
        history = list(state["messages"])[:-1]
        rag_chunks = _rag_chunks_from_state(state.get("rag_chunks"))
        reference_topic = state.get("reference_topic")
        reference_campus = state.get("reference_campus")
        loop = asyncio.get_running_loop()
        answer_text, sources = await loop.run_in_executor(
            None,
            lambda: generate_answer(
                llm=llm,
                history=history,
                user_question=question,
                evidence_list=ev_list,
                rag_chunks=rag_chunks,
                rag_only=rag_only,
                rc_unavailable=rc_unavailable,
                reference_topic=reference_topic,
                reference_campus=reference_campus,
                rag_first=bool(state.get("rag_first")),
            ),
        )
        _log_node_done(
            "generate",
            t0,
            rag_only=rag_only,
            sources=len(sources),
            reference_topic=reference_topic or "",
            reference_campus=reference_campus or "",
            rag_first=bool(state.get("rag_first")),
        )
        return {
            "messages": [AIMessage(content=answer_text)],
            "final_sources": [s.model_dump() for s in sources],
        }

    g: StateGraph = StateGraph(AgentState)
    g.add_node("prepare", node_prepare)
    g.add_node("decompose", node_decompose)
    g.add_node("per_subquery", node_per_subquery)
    g.add_node("generate", node_generate)

    g.add_edge(START, "prepare")
    g.add_conditional_edges(
        "prepare",
        route_after_prepare,
        {"decompose": "decompose", "generate": "generate"},
    )
    g.add_conditional_edges("decompose", fanout, ["per_subquery"])
    g.add_edge("per_subquery", "generate")
    g.add_edge("generate", END)

    checkpointer = MemorySaver()
    return g.compile(checkpointer=checkpointer)
