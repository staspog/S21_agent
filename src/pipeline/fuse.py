"""Reciprocal Rank Fusion (RRF) для объединения нескольких ранжированных списков.

Cormack et al., 2009: score(d) = Σ_l 1 / (k + rank_l(d)). k=60 — стандартный.

Используется для слияния результатов разных searchText по разным комнатам в
рамках одного подвопроса. Дубликаты по `mid` дедуплицируются (берём
сообщение из первого списка, где увидели его).
"""

from __future__ import annotations

import logging
from collections import defaultdict

from src.rc.schemas import Hit

log = logging.getLogger("s21.pipeline.fuse")


def rrf_fuse(lists: list[list[Hit]], k: int = 60, max_out: int = 80) -> list[Hit]:
    """Сливает несколько ранжированных списков по RRF, возвращает дедуп-список Hit-ов."""
    score: dict[str, float] = defaultdict(float)
    first_seen: dict[str, Hit] = {}
    list_count = 0

    for lst in lists:
        if not lst:
            continue
        list_count += 1
        for rank, h in enumerate(lst, start=1):
            mid = h.mid
            if mid not in first_seen:
                first_seen[mid] = h
            score[mid] += 1.0 / (k + rank)

    fused = sorted(
        first_seen.values(),
        key=lambda h: score[h.mid],
        reverse=True,
    )

    log.info(
        "fuse: lists_in=%s unique_hits=%s top_score=%.4f",
        list_count,
        len(fused),
        score[fused[0].mid] if fused else 0.0,
    )
    return fused[:max_out]
