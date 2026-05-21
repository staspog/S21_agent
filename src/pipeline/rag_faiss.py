"""FAISS RAG: локальный индекс справочных HTML/FAQ документов.

Задача:
  - по пользовательскому запросу найти top-K кандидатов в FAISS,
  - затем переранжировать cross-encoder'ом (тем же, что используется в RC-pipeline),
  - вернуть top-N чанков как «динамический доменный контекст» (а не источники).
"""

from __future__ import annotations

import json
import logging
import os
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

log = logging.getLogger("s21.pipeline.rag_faiss")


@dataclass(frozen=True)
class RagConfig:
    index_path: Path = Path("faiss_store/index.faiss")
    chunks_path: Path = Path("faiss_store/chunks.jsonl")
    config_path: Path = Path("faiss_store/config.json")
    embed_model: str = "intfloat/multilingual-e5-small"
    top_k: int = 5
    rerank_top_n: int = 0

    @staticmethod
    def from_env() -> "RagConfig":
        def _env_int(name: str, default: int) -> int:
            raw = os.getenv(name)
            if raw is None or not raw.strip():
                return default
            try:
                return int(raw)
            except ValueError:
                return default

        return RagConfig(
            index_path=Path(os.getenv("RAG_INDEX_PATH", "faiss_store/index.faiss")),
            chunks_path=Path(os.getenv("RAG_CHUNKS_PATH", "faiss_store/chunks.jsonl")),
            config_path=Path(os.getenv("RAG_CONFIG_PATH", "faiss_store/config.json")),
            embed_model=os.getenv("RAG_EMBED_MODEL", "intfloat/multilingual-e5-small"),
            top_k=_env_int("RAG_TOPK", 5),
            rerank_top_n=_env_int("RAG_RERANK_TOP", 0),
        )


_load_lock = threading.Lock()
_cached: dict[str, Any] = {"loaded": False}


def _read_json(path: Path) -> dict:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = (line or "").strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def _safe_env_for_cpu() -> None:
    from src.pipeline.rerank import apply_cpu_threads

    apply_cpu_threads()


def load_rag_assets(cfg: RagConfig) -> None:
    """Загрузить индекс FAISS + chunks + embedder (lazy, с кэшем)."""
    with _load_lock:
        if _cached.get("loaded"):
            return

        _safe_env_for_cpu()

        import faiss  # локальный импорт, чтобы не грузить faiss при старте без RAG
        from sentence_transformers import SentenceTransformer

        store_cfg = _read_json(cfg.config_path)
        model_name = (store_cfg.get("model") or cfg.embed_model).strip() or cfg.embed_model

        index = faiss.read_index(str(cfg.index_path))
        chunks = _read_jsonl(cfg.chunks_path)

        embedder = SentenceTransformer(model_name, device="cpu")

        _cached.update(
            {
                "loaded": True,
                "index": index,
                "chunks": chunks,
                "embedder": embedder,
                "model_name": model_name,
            }
        )
        log.info(
            "rag: loaded index dim=%s ntotal=%s chunks=%s model=%s",
            getattr(index, "d", "?"),
            getattr(index, "ntotal", "?"),
            len(chunks),
            model_name,
        )


def _embed_query(text: str) -> np.ndarray:
    embedder = _cached["embedder"]
    vec = embedder.encode(
        [f"query: {text}"],
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=False,
    ).astype(np.float32)
    return np.ascontiguousarray(vec)


def _format_chunk_for_prompt(ch: dict[str, Any], *, max_chars: int = 900) -> str:
    meta = ch.get("metadata") or {}
    section = (meta.get("section") or "").strip()
    source = (
        (meta.get("url") or meta.get("source") or meta.get("slug") or "")
        .strip()
    )
    txt = (ch.get("text") or "").strip()
    if len(txt) > max_chars:
        txt = txt[: max_chars - 1] + "…"

    header = []
    if section:
        header.append(f"### {section}")
    if source:
        header.append(f"(source: {source})")
    if header:
        return "\n".join(header) + "\n" + txt
    return txt


def retrieve_rag_chunks(
    *,
    query: str,
    cfg: RagConfig,
    reranker_model_name: str,
) -> list[dict[str, Any]]:
    """Top-K из FAISS -> rerank -> top-N. Возвращает список chunk dict-ов."""
    if not (query or "").strip():
        return []

    load_rag_assets(cfg)

    import faiss
    from .rerank import rerank_texts  # reuse CrossEncoder loader/lock

    index = _cached["index"]
    chunks: list[dict[str, Any]] = _cached["chunks"]

    qv = _embed_query(query)
    faiss.normalize_L2(qv)
    k = max(1, int(cfg.top_k))
    scores, idxs = index.search(qv, k)
    idx_list = [int(i) for i in (idxs[0].tolist() if len(idxs) else []) if int(i) >= 0]
    candidates = [chunks[i] for i in idx_list if i < len(chunks)]
    if not candidates:
        return []

    rerank_n = int(cfg.rerank_top_n)
    if rerank_n <= 0:
        return candidates[: max(1, int(cfg.top_k))]

    texts = [c.get("text") or "" for c in candidates]
    reranked = rerank_texts(
        query=query,
        texts=texts,
        model_name=reranker_model_name,
        top_n=max(1, int(cfg.rerank_top_n)),
    )

    out: list[dict[str, Any]] = []
    for rr in reranked:
        i = int(rr["index"])
        if i < 0 or i >= len(candidates):
            continue
        ch = dict(candidates[i])
        meta = dict(ch.get("metadata") or {})
        meta["rag_ce_score"] = float(rr["score"])
        ch["metadata"] = meta
        out.append(ch)

    return out


def rag_status(cfg: RagConfig | None = None) -> dict[str, Any]:
    """Diagnostics for /health: whether FAISS index is loaded."""
    cfg = cfg or RagConfig.from_env()
    loaded = bool(_cached.get("loaded"))
    chunks = _cached.get("chunks") if loaded else None
    ntotal = None
    if loaded:
        index = _cached.get("index")
        ntotal = getattr(index, "ntotal", None)
    index_exists = cfg.index_path.is_file()
    return {
        "rag_loaded": loaded,
        "rag_chunks": len(chunks) if chunks is not None else (None if not index_exists else 0),
        "rag_index_ntotal": ntotal,
        "rag_index_path_exists": index_exists,
    }


def rag_context_text(chunks: list[dict[str, Any]] | None) -> str:
    """Строка для промптов: короткий справочный контекст из top чанков."""
    items = list(chunks or [])
    if not items:
        return ""
    parts = [_format_chunk_for_prompt(c) for c in items]
    return "\n\n".join(parts).strip()


__all__ = [
    "RagConfig",
    "load_rag_assets",
    "retrieve_rag_chunks",
    "rag_context_text",
    "rag_status",
]

