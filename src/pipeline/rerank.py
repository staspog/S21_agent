"""Гибридный rerank на CPU: BM25 (sparse) + cross-encoder (dense).

Используется на каждый подвопрос после RRF-фьюжна. Возвращает top-N
переранжированных Hit-ов с проставленными `bm25_score`, `ce_score`, `final_score`.

Реранкер по умолчанию: `qilowoq/bge-reranker-v2-m3-en-ru` — труcированный
до en+ru версии BAAI/bge-reranker-v2-m3, ~278M->~180M параметров,
multilingual, ~130мс на батч 16 на CPU.
"""

from __future__ import annotations

import logging
import re
import threading
from functools import lru_cache

import numpy as np
from rank_bm25 import BM25Okapi

from src.rc.schemas import Hit

log = logging.getLogger("s21.pipeline.rerank")


_TOKEN_PAT = re.compile(r"[А-Яа-яЁёA-Za-z0-9]+", flags=re.UNICODE)


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


_ce_lock = threading.Lock()


@lru_cache(maxsize=2)
def _load_ce(model_name: str):
    """Lazy-load cross-encoder. lru_cache + lock защищает от дублирующей загрузки."""
    from sentence_transformers import CrossEncoder

    log.info("rerank: загрузка CrossEncoder %s …", model_name)
    model = CrossEncoder(model_name, max_length=512, device="cpu")
    log.info("rerank: модель готова")
    return model


def warmup_reranker(model_name: str) -> None:
    """Прогрев модели на старте сервера (фоном из api.py)."""
    with _ce_lock:
        model = _load_ce(model_name)
    try:
        model.predict([("разогрев", "тест прогрева модели")])
    except Exception:
        log.exception("rerank: warmup failed")


def hybrid_rerank(
    *,
    subquery: str,
    hits: list[Hit],
    model_name: str,
    bm25_weight: float = 0.4,
    top_n: int = 8,
    rerank_max_candidates: int = 40,
) -> list[Hit]:
    """BM25 + cross-encoder rerank → top_n. Не мутирует входной список.

    Алгоритм:
        1. Усекаем кандидатов до `rerank_max_candidates` (по порядку из RRF).
        2. BM25 по тексту сообщения; нормализуем 0..1.
        3. Cross-encoder predict [(subquery, msg)]; нормализуем 0..1.
        4. final = bm25_weight * BM25 + (1 - bm25_weight) * CE.
    """
    if not hits:
        return []

    candidates = hits[:rerank_max_candidates]
    docs = [h.msg or "" for h in candidates]

    tokenized_corpus = [_tokenize(d) for d in docs]
    if any(tokenized_corpus):
        bm25 = BM25Okapi(tokenized_corpus)
        bm25_raw = bm25.get_scores(_tokenize(subquery)).tolist()
    else:
        bm25_raw = [0.0] * len(candidates)
    bm25_norm = _normalize(bm25_raw)

    ce_raw: list[float]
    try:
        with _ce_lock:
            model = _load_ce(model_name)
        pairs = [(subquery, d) for d in docs]
        ce_arr = model.predict(pairs, show_progress_bar=False, batch_size=16)
        ce_raw = [float(x) for x in np.asarray(ce_arr).tolist()]
    except Exception:
        log.exception("rerank: cross-encoder failed, fallback к BM25-only")
        ce_raw = [0.0] * len(candidates)
    ce_norm = _normalize(ce_raw)

    w = max(0.0, min(1.0, float(bm25_weight)))
    scored: list[tuple[float, Hit]] = []
    for i, h in enumerate(candidates):
        out = h.model_copy(deep=False)
        out.bm25_score = float(bm25_raw[i])
        out.ce_score = float(ce_raw[i])
        final = w * bm25_norm[i] + (1.0 - w) * ce_norm[i]
        out.final_score = float(final)
        scored.append((final, out))

    scored.sort(key=lambda t: t[0], reverse=True)
    top = [h for _, h in scored[: top_n]]
    log.info(
        "rerank: candidates=%s top_n=%s top_score=%.3f",
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
    """Утилита: rerank произвольных текстов cross-encoder'ом.

    Возвращает список dict-ов: {index: <позиция во входном texts>, score: <raw_score>},
    отсортированный по score (desc). Использует тот же кэш/lock `_load_ce()`.
    """
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
