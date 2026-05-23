"""Гибридный rerank на CPU: BM25 (sparse) + cross-encoder (dense)."""

from __future__ import annotations

import logging
import os
import re
import threading
from functools import lru_cache
from typing import Literal

import numpy as np
from rank_bm25 import BM25Okapi

from src.rc.config import RerankMode
from src.rc.schemas import Hit

log = logging.getLogger("s21.pipeline.rerank")

RerankModeArg = RerankMode

_TOKEN_PAT = re.compile(r"[А-Яа-яЁёA-Za-z0-9]+", flags=re.UNICODE)

_ce_lock = threading.Lock()
_threads_applied = False


def _env_int(name: str, default: int) -> int:
    raw = os.getenv(name)
    if raw is None or not raw.strip():
        return default
    try:
        return int(raw)
    except ValueError:
        return default


def apply_cpu_threads() -> int:
    """Выставить потоки BLAS/torch из env (default 2). Идемпотентно."""
    global _threads_applied
    n = max(1, _env_int("TORCH_NUM_THREADS", _env_int("OMP_NUM_THREADS", 2)))
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    os.environ["OMP_NUM_THREADS"] = str(n)
    os.environ["MKL_NUM_THREADS"] = str(n)
    os.environ["OPENBLAS_NUM_THREADS"] = str(n)
    try:
        import torch

        torch.set_num_threads(n)
    except ImportError:
        pass
    if not _threads_applied:
        log.info("rerank: cpu_threads=%s", n)
        _threads_applied = True
    return n


def _tokenize(text: str) -> list[str]:
    return [t.lower() for t in _TOKEN_PAT.findall(text or "")]


def _normalize(scores: list[float]) -> list[float]:
    if not scores:
        return []
    arr = np.asarray(scores, dtype=float)
    lo = float(np.min(arr))
    hi = float(np.max(arr))
    if hi - lo < 1e-9:
        return [0.5] * len(scores)
    return ((arr - lo) / (hi - lo)).tolist()


@lru_cache(maxsize=2)
def _load_ce(model_name: str):
    from sentence_transformers import CrossEncoder

    apply_cpu_threads()
    log.info("rerank: загрузка CrossEncoder %s …", model_name)
    model = CrossEncoder(model_name, max_length=512, device="cpu")
    log.info("rerank: модель готова")
    return model


def warmup_reranker(model_name: str) -> None:
    """Прогрев модели на старте сервера (фоном из api.py)."""
    apply_cpu_threads()
    with _ce_lock:
        model = _load_ce(model_name)
    try:
        model.predict([("разогрев", "тест прогрева модели")])
    except Exception:
        log.exception("rerank: warmup failed")


def _bm25_scores(subquery: str, docs: list[str]) -> list[float]:
    tokenized_corpus = [_tokenize(d) for d in docs]
    if not any(tokenized_corpus):
        return [0.0] * len(docs)
    bm25 = BM25Okapi(tokenized_corpus)
    return bm25.get_scores(_tokenize(subquery)).tolist()


def hybrid_rerank(
    *,
    subquery: str,
    hits: list[Hit],
    model_name: str,
    bm25_weight: float = 0.4,
    top_n: int = 8,
    rerank_max_candidates: int = 15,
    rerank_mode: RerankMode = "hybrid",
) -> list[Hit]:
    """BM25 + optional cross-encoder rerank → top_n."""
    if not hits:
        return []

    candidates = hits[:rerank_max_candidates]
    docs = [h.msg or "" for h in candidates]
    bm25_raw = _bm25_scores(subquery, docs)
    bm25_norm = _normalize(bm25_raw)

    use_ce = rerank_mode == "hybrid"
    ce_raw: list[float]
    if use_ce:
        try:
            with _ce_lock:
                model = _load_ce(model_name)
            pairs = [(subquery, d) for d in docs]
            ce_arr = model.predict(pairs, show_progress_bar=False, batch_size=16)
            ce_raw = [float(x) for x in np.asarray(ce_arr).tolist()]
        except Exception:
            log.exception("rerank: cross-encoder failed, fallback к BM25-only")
            ce_raw = [0.0] * len(candidates)
            use_ce = False
    else:
        ce_raw = [0.0] * len(candidates)

    ce_norm = _normalize(ce_raw) if use_ce else [0.0] * len(candidates)
    w = max(0.0, min(1.0, float(bm25_weight)))
    if not use_ce:
        w = 1.0

    scored: list[tuple[float, Hit]] = []
    for i, h in enumerate(candidates):
        out = h.model_copy(deep=False)
        out.bm25_score = float(bm25_raw[i])
        out.ce_score = float(ce_raw[i]) if use_ce else None
        final = w * bm25_norm[i] + (1.0 - w) * ce_norm[i]
        out.final_score = float(final)
        scored.append((final, out))

    scored.sort(key=lambda t: t[0], reverse=True)
    top = [h for _, h in scored[:top_n]]
    log.info(
        "rerank: mode=%s candidates=%s top_n=%s top_score=%.3f",
        rerank_mode,
        len(candidates),
        len(top),
        top[0].final_score if top else 0.0,
    )
    return top


def rerank_texts(
    *,
    query: str,
    texts: list[str],
    model_name: str,
    top_n: int = 5,
) -> list[dict[str, float | int]]:
    q = (query or "").strip()
    if not q or not texts:
        return []

    n = max(1, int(top_n))
    docs = [t or "" for t in texts]
    try:
        with _ce_lock:
            model = _load_ce(model_name)
        pairs = [(q, d) for d in docs]
        ce_arr = model.predict(pairs, show_progress_bar=False, batch_size=16)
        raw = [float(x) for x in np.asarray(ce_arr).tolist()]
    except Exception:
        log.exception("rerank_texts: cross-encoder failed")
        return []

    scored: list[tuple[float, int]] = [(raw[i], i) for i in range(len(raw))]
    scored.sort(key=lambda t: t[0], reverse=True)
    return [{"score": float(s), "index": int(i)} for s, i in scored[:n]]
