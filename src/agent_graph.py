from __future__ import annotations

import logging
from typing_extensions import NotRequired

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage
from langchain_gigachat.chat_models import GigaChat
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, MessagesState, StateGraph

from .search import search_chunks_ranked

log = logging.getLogger("s21.agent.graph")


class AgentState(MessagesState):
    retrieved_chunks: NotRequired[list[dict]]
    top_k: NotRequired[int]


def _last_human_text(messages: list[BaseMessage]) -> str:
    for msg in reversed(messages):
        if isinstance(msg, HumanMessage):
            c = msg.content
            return c if isinstance(c, str) else str(c)
    raise ValueError("В истории нет сообщения пользователя")


def _format_context(chunks: list[dict]) -> str:
    return "\n\n---\n\n".join(
        f"Источник: {c['source_file']} / {c['section']}\n{c['content']}" for c in chunks
    )


def chunk_sources(chunks: list[dict]) -> list[str]:
    return [f"{c['source_file']} / {c['section']}" for c in chunks]


def _preview_text(text: str, limit: int = 120) -> str:
    one_line = " ".join((text or "").split())
    return one_line if len(one_line) <= limit else one_line[: limit - 1] + "…"


def build_rag_graph(
    embed_model,
    index,
    chunks_map: dict,
    gigachat_credentials: str,
):
    llm = GigaChat(credentials=gigachat_credentials, verify_ssl_certs=False)

    def retrieve(state: AgentState) -> dict:
        log.info("node retrieve: start (graph step: START → retrieve)")
        top_k = int(state.get("top_k") or 3)
        query = _last_human_text(list(state["messages"]))
        chunks, ranked_meta = search_chunks_ranked(
            query, embed_model, index, chunks_map, top_k=top_k
        )
        q_preview = _preview_text(query, 180)
        log.info("node retrieve: query_preview=%r top_k=%s", q_preview, top_k)
        if ranked_meta:
            lines = [
                f"  #{m['rank']} faiss_id={m['faiss_id']} L2={m['l2']:.4f} {m['source']}\n"
                f"      preview: {m['preview']}"
                for m in ranked_meta
            ]
            log.info("node retrieve: RAG hits (%d):\n%s", len(lines), "\n".join(lines))
        else:
            log.warning("node retrieve: RAG hits пусто")
        log.info("node retrieve: end → переход к generate")
        return {"retrieved_chunks": chunks}

    def generate(state: AgentState) -> dict:
        log.info("node generate: start (graph step: retrieve → generate)")
        chunks = list(state.get("retrieved_chunks") or [])
        context_text = _format_context(chunks)
        system_text = (
            "Ты — умный и вежливый помощник.\n\n"
            "Политика фактов:\n"
            "- Опирайся на факты из блока «Контекст» ниже при ответе на вопрос пользователя.\n"
            "- Если в контексте нет нужной информации, скажи об этом прямо; не придумывай факты, "
            "которых нет в контексте.\n\n"
            "Политика диалога:\n"
            "- Учитывай устойчивые просьбы пользователя из переписки (обращение по имени, тон, стиль), "
            "если это не мешает честности ответа по контексту.\n"
            "- Если пользователь просил обращаться к нему определённым образом, продолжай делать это "
            "(например, в начале ответа), когда это уместно.\n\n"
            f"Контекст:\n{context_text}"
        )
        system = SystemMessage(content=system_text)
        convo = list(state["messages"])
        sys_chars = len(system_text)
        log.info(
            "node generate: convo_messages=%s context_chunks=%s system_prompt_chars=%s",
            len(convo),
            len(chunks),
            sys_chars,
        )
        response = llm.invoke([system] + convo)
        raw = getattr(response, "content", "") or ""
        if not isinstance(raw, str):
            raw = str(raw)
        log.info(
            "node generate: end → END (answer_preview=%r, chars=%s)",
            _preview_text(raw, 160),
            len(raw),
        )
        return {"messages": [AIMessage(content=raw)]}

    graph = StateGraph(AgentState)
    graph.add_node("retrieve", retrieve)
    graph.add_node("generate", generate)
    graph.add_edge(START, "retrieve")
    graph.add_edge("retrieve", "generate")
    graph.add_edge("generate", END)

    checkpointer = MemorySaver()
    return graph.compile(checkpointer=checkpointer)
