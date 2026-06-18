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
import re
import threading
from pathlib import Path
from typing import Any

import numpy as np
from pydantic import BaseModel, ConfigDict

from src.rc.schemas import RagChunk
from src.reference_intent import (
    BENEFITS_CITY_HEADERS,
    CLUB_CAMPUS_HEADERS,
    GUEST_CAMPUS_HEADERS,
    ReferenceTopic,
    topic_slug,
)

_URL_RE = re.compile(r"https?://[^\s\)\]\"'<>]+", re.IGNORECASE)
_MD_LINK_RE = re.compile(r"\[([^\]]+)\]\(([^)]+)\)")
log = logging.getLogger("s21.pipeline.rag_faiss")

_CATALOG_SLUGS = frozenset({"education/club_chats", "benefits"})
_CATALOG_MAX_CHARS = 1500
_SLUG_MERGE_CAP = 12
_H2_RE = re.compile(r"^##\s+(.+?)\s*$", re.MULTILINE)
_APPLICANT_ATTR_MARKERS = (
    "из справочника applicant",
    "взяты из applicant",
    "ссылки — из applicant",
    "подробнее на applicant",
)
APPLICANT_LINKS_FOOTER = "\n\n_Ссылки взяты из справочника applicant._"


class RagConfig(BaseModel):
    model_config = ConfigDict(frozen=True)

    index_path: Path = Path("faiss_store/index.faiss")
    chunks_path: Path = Path("faiss_store/chunks.jsonl")
    config_path: Path = Path("faiss_store/config.json")
    embed_model: str = "intfloat/multilingual-e5-small"
    top_k: int = 5
    rerank_top_n: int = 0

    @classmethod
    def from_env(cls) -> RagConfig:
        def _env_int(name: str, default: int) -> int:
            raw = os.getenv(name)
            if raw is None or not raw.strip():
                return default
            try:
                return int(raw)
            except ValueError:
                return default

        return cls(
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


def _read_jsonl(path: Path) -> list[RagChunk]:
    rows: list[RagChunk] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = (line or "").strip()
            if not line:
                continue
            rows.append(RagChunk.from_record(json.loads(line)))
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


def _chunk_slug(ch: RagChunk) -> str:
    return ch.metadata.slug.strip()


def _chunk_id(ch: RagChunk) -> str:
    return ch.id


def _chunks_by_slug(all_chunks: list[RagChunk], slug: str) -> list[RagChunk]:
    return [c for c in all_chunks if _chunk_slug(c) == slug]


def _h2_headers_in_text(text: str) -> list[str]:
    return [m.group(1).strip() for m in _H2_RE.finditer(text or "")]


def _campus_from_chunk(ch: RagChunk, headers: tuple[str, ...]) -> str | None:
    header_set = set(headers)
    meta_section = ch.metadata.section.strip()
    if meta_section in header_set:
        return meta_section
    text = ch.text.strip()
    if text:
        first_line = text.split("\n", 1)[0].strip()
        if first_line in header_set:
            return first_line
    for h in _h2_headers_in_text(text):
        if h in header_set:
            return h
    return None


def _select_section_slug_chunks(
    slug_chunks: list[RagChunk],
    section: str,
    headers: tuple[str, ...],
) -> list[RagChunk]:
    """Все part-ы кампуса/города до следующего кампуса (включая H3 без campus в section)."""
    if not section or not slug_chunks:
        return []
    ordered = sorted(slug_chunks, key=_chunk_id)
    out: list[RagChunk] = []
    current: str | None = None
    for ch in ordered:
        campus = _campus_from_chunk(ch, headers)
        if campus:
            if campus == section:
                current = campus
            elif current == section:
                break
            else:
                current = campus if campus == section else None
        if current == section:
            out.append(ch)
    return out


def _merge_slug_chunks(
    faiss_candidates: list[RagChunk],
    *,
    reference_topic: ReferenceTopic | None,
    reference_campus: str | None,
    all_chunks: list[RagChunk],
) -> list[RagChunk]:
    slug = topic_slug(reference_topic)
    if not slug:
        return faiss_candidates

    slug_chunks = _chunks_by_slug(all_chunks, slug)
    if reference_campus:
        if reference_topic == "clubs":
            headers = CLUB_CAMPUS_HEADERS
        elif reference_topic == "guests":
            headers = GUEST_CAMPUS_HEADERS
        else:
            headers = BENEFITS_CITY_HEADERS
        section_chunks = _select_section_slug_chunks(slug_chunks, reference_campus, headers)
        if section_chunks:
            slug_chunks = section_chunks

    seen: set[str] = set()
    merged: list[RagChunk] = []
    for ch in faiss_candidates + slug_chunks:
        cid = _chunk_id(ch)
        if cid and cid in seen:
            continue
        if cid:
            seen.add(cid)
        merged.append(ch)
        if len(merged) >= _SLUG_MERGE_CAP:
            break
    return merged


def _format_max_chars(ch: RagChunk) -> int:
    if _chunk_slug(ch) in _CATALOG_SLUGS:
        return _CATALOG_MAX_CHARS
    return 900


def _format_chunk_for_prompt(ch: RagChunk, *, max_chars: int | None = None) -> str:
    limit = max_chars if max_chars is not None else _format_max_chars(ch)
    meta = ch.metadata
    section = meta.section.strip()
    source = (meta.url or meta.source or meta.slug or "").strip()
    txt = ch.text.strip()
    if len(txt) > limit:
        txt = txt[: limit - 1] + "…"

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
    reference_topic: ReferenceTopic | None = None,
    reference_campus: str | None = None,
) -> list[RagChunk]:
    """Top-K из FAISS -> rerank -> top-N. Возвращает список RagChunk."""
    if not (query or "").strip():
        return []

    load_rag_assets(cfg)

    import faiss
    from .rerank import rerank_texts  # reuse CrossEncoder loader/lock

    index = _cached["index"]
    chunks: list[RagChunk] = _cached["chunks"]

    qv = _embed_query(query)
    faiss.normalize_L2(qv)
    k = max(1, int(cfg.top_k))
    scores, idxs = index.search(qv, k)
    idx_list = [int(i) for i in (idxs[0].tolist() if len(idxs) else []) if int(i) >= 0]
    candidates = [chunks[i] for i in idx_list if i < len(chunks)]
    if not candidates:
        return []

    merged = _merge_slug_chunks(
        candidates,
        reference_topic=reference_topic,
        reference_campus=reference_campus,
        all_chunks=chunks,
    )

    rerank_n = int(cfg.rerank_top_n)
    if rerank_n <= 0:
        return merged[: max(len(merged), 1)]

    texts = [c.text for c in merged]
    reranked = rerank_texts(
        query=query,
        texts=texts,
        model_name=reranker_model_name,
        top_n=max(1, int(cfg.rerank_top_n)),
    )

    out: list[RagChunk] = []
    for rr in reranked:
        i = int(rr["index"])
        if i < 0 or i >= len(merged):
            continue
        ch = merged[i]
        meta = ch.metadata.model_copy(update={"rag_ce_score": float(rr["score"])})
        out.append(ch.model_copy(update={"metadata": meta}))

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


def rag_context_text(chunks: list[RagChunk] | None) -> str:
    """Строка для промптов: короткий справочный контекст из top чанков."""
    items = list(chunks or [])
    if not items:
        return ""
    parts = [_format_chunk_for_prompt(c) for c in items]
    return "\n\n".join(parts).strip()


def normalize_url(url: str) -> str:
    u = (url or "").strip().rstrip(".,;:!?)»\"'""''")
    if not u:
        return ""
    if "?" in u:
        base, qs = u.split("?", 1)
        if any(p in qs for p in ("roistat_visit=", "utm_", "roistat=")):
            u = base
    return u.rstrip("/").lower()


def extract_urls_from_text(text: str) -> set[str]:
    out: set[str] = set()
    for m in _URL_RE.finditer(text or ""):
        n = normalize_url(m.group(0))
        if n:
            out.add(n)
    return out


def extract_markdown_link_urls(text: str) -> set[str]:
    """URL из markdown-ссылок ``[label](url)`` в тексте ответа."""
    out: set[str] = set()
    for m in _MD_LINK_RE.finditer(text or ""):
        href = (m.group(2) or "").strip()
        if href.startswith(("http://", "https://")):
            n = normalize_url(href)
            if n:
                out.add(n)
    return out


def extract_bare_urls(text: str) -> set[str]:
    """Голые https:// в тексте, кроме служебного ``(source: url)`` из RAG-промпта."""
    out: set[str] = set()
    for m in _URL_RE.finditer(text or ""):
        start = m.start()
        prefix = (text or "")[max(0, start - 24) : start].lower()
        if "source:" in prefix:
            continue
        n = normalize_url(m.group(0))
        if n:
            out.add(n)
    return out


def extract_user_facing_urls(text: str) -> set[str]:
    """Ссылки, которые модель реально отдала пользователю в ответе."""
    md = extract_markdown_link_urls(text)
    if md:
        return md
    return extract_bare_urls(text)


def is_adm_info_chunk(ch: RagChunk) -> bool:
    chunk_file = ch.metadata.chunk_file.lower()
    if "adm_info" in chunk_file:
        return True
    return ch.id.startswith("adm_info_")


def adm_info_urls_from_chunks(chunks: list[RagChunk] | None) -> set[str]:
    """URL из тела adm_info-чанков (без metadata source — это URL страницы, не ссылка в тексте)."""
    urls: set[str] = set()
    for ch in chunks or []:
        if not is_adm_info_chunk(ch):
            continue
        urls.update(extract_urls_from_text(ch.text))
    return urls


def answer_uses_adm_info_urls(answer: str, adm_urls: set[str]) -> bool:
    if not adm_urls:
        return False
    return bool(extract_user_facing_urls(answer) & adm_urls)


def applicant_links_footer_if_needed(
    answer: str,
    rag_chunks: list[RagChunk] | None,
) -> str:
    """Дописывает дисклеймер, если в ответе есть URL из adm_info RAG."""
    if not (answer or "").strip() or not rag_chunks:
        return ""
    lower = answer.lower()
    if any(marker in lower for marker in _APPLICANT_ATTR_MARKERS):
        return ""
    adm_urls = adm_info_urls_from_chunks(rag_chunks)
    if answer_uses_adm_info_urls(answer, adm_urls):
        return APPLICANT_LINKS_FOOTER
    return ""


__all__ = [
    "RagConfig",
    "APPLICANT_LINKS_FOOTER",
    "load_rag_assets",
    "retrieve_rag_chunks",
    "rag_context_text",
    "rag_status",
    "normalize_url",
    "extract_urls_from_text",
    "extract_markdown_link_urls",
    "extract_bare_urls",
    "extract_user_facing_urls",
    "is_adm_info_chunk",
    "adm_info_urls_from_chunks",
    "answer_uses_adm_info_urls",
    "applicant_links_footer_if_needed",
]

