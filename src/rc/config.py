"""Конфигурация Rocket.Chat и параметров пайплайна — из переменных окружения."""

from __future__ import annotations

import math
import os
from dataclasses import dataclass
from typing import Literal

RerankMode = Literal["hybrid", "bm25_only"]

_DEEP_PROFILE_BOOST = 1.2


def _deep_scale(n: int) -> int:
    """+20% к лимитам deep-профиля (ceil, минимум 1)."""
    return max(1, math.ceil(n * _DEEP_PROFILE_BOOST))


def _env_str(name: str, default: str = "") -> str:
    return (os.getenv(name) or default).strip()


def _env_int(name: str, default: int) -> int:
    raw = os.getenv(name)
    if raw is None or not raw.strip():
        return default
    try:
        return int(raw)
    except ValueError:
        return default


def _env_float(name: str, default: float) -> float:
    raw = os.getenv(name)
    if raw is None or not raw.strip():
        return default
    try:
        return float(raw)
    except ValueError:
        return default


def _env_csv(name: str, default: tuple[str, ...]) -> tuple[str, ...]:
    raw = os.getenv(name)
    if raw is None:
        return default
    parts = [p.strip().lstrip("#") for p in raw.split(",")]
    parts = [p for p in parts if p]
    return tuple(parts) if parts else default


def _env_rerank_mode(name: str, default: RerankMode) -> RerankMode:
    raw = _env_str(name, default).lower()
    if raw in ("bm25_only", "bm25", "bm25-only"):
        return "bm25_only"
    return "hybrid"


@dataclass(frozen=True)
class PipelineProfile:
    """Параметры одного прогона per_subquery (fast или deep)."""

    max_subqueries: int
    rooms_per_subquery: int
    queries_per_subquery: int
    search_count: int
    rerank_top_n: int
    rerank_max_candidates: int
    rerank_mode: RerankMode
    bm25_weight: float
    thread_expand_top: int
    thread_messages_count: int


@dataclass(frozen=True)
class RocketChatConfig:
    """Все параметры для Rocket.Chat-клиента и пайплайна."""

    base_url: str
    user: str
    password: str

    # HTTP
    timeout_connect_s: float = 10.0
    timeout_read_s: float = 60.0
    probe_timeout_s: float = 8.0
    probe_cache_ttl_s: float = 30.0
    catalog_fetch_timeout_s: float = 25.0
    search_timeout_s: float = 20.0
    thread_timeout_s: float = 15.0
    http_concurrency: int = 12

    # Catalog
    catalog_ttl_s: int = 600
    catalog_page_size: int = 50
    catalog_max_rooms: int = 200
    always_search_room_names: tuple[str, ...] = (
        "msk_adm",
        "msk_general",
        "nsk_adm",
        "kzn_adm",
        "tas_adm",
        "skd_adm",
        "yks_adm",
        "PIN_info",
        "product_feedback",
        "product_info",
        "s21_feedback",
        "content_help",
        "general",
        "dir_talks",
        "Natasha_is_listening",
        "kzn_volunteers",
        "Internship_kzn",
        "contests",
        "cybersec_21",
        "lab_kzn",
        "chgk",
        "Buddy-komandy",
        "ft_memes",
        "nn_helpers",
        "random",
        "born_to_code",
        "Random_coffee",
        "SQL",
    )

    # Pipeline (deep defaults; env overrides apply at load)
    max_subqueries: int = 2
    rooms_per_subquery: int = 6
    queries_per_subquery: int = 2
    search_count: int = 20
    rerank_top_n: int = 8
    rerank_max_candidates: int = 15
    rerank_mode: RerankMode = "hybrid"
    bm25_weight: float = 0.4
    thread_expand_top: int = 2
    thread_messages_count: int = 30

    reranker_model: str = "qilowoq/bge-reranker-v2-m3-en-ru"

    def profile_for(self, *, deep: bool) -> PipelineProfile:
        """Профиль пайплайна: deep (+20% к базовым лимитам) или fast (урезанный)."""
        if deep:
            return PipelineProfile(
                max_subqueries=_deep_scale(self.max_subqueries),
                rooms_per_subquery=_deep_scale(self.rooms_per_subquery),
                queries_per_subquery=_deep_scale(self.queries_per_subquery),
                search_count=_deep_scale(self.search_count),
                rerank_top_n=_deep_scale(self.rerank_top_n),
                rerank_max_candidates=_deep_scale(self.rerank_max_candidates),
                rerank_mode=self.rerank_mode,
                bm25_weight=self.bm25_weight,
                thread_expand_top=_deep_scale(self.thread_expand_top),
                thread_messages_count=_deep_scale(self.thread_messages_count),
            )
        return PipelineProfile(
            max_subqueries=1,
            rooms_per_subquery=min(4, self.rooms_per_subquery),
            queries_per_subquery=min(2, self.queries_per_subquery),
            search_count=min(15, self.search_count),
            rerank_top_n=min(6, self.rerank_top_n),
            rerank_max_candidates=min(10, self.rerank_max_candidates),
            rerank_mode="bm25_only",
            bm25_weight=self.bm25_weight,
            thread_expand_top=min(1, self.thread_expand_top),
            thread_messages_count=min(20, self.thread_messages_count),
        )

    @property
    def web_message_path_for_kind(self) -> dict[str, str]:
        return {"channel": "channel", "group": "group", "im": "direct"}


def load_rc_config() -> RocketChatConfig:
    """Загрузка конфига из .env. Кидает ясную ошибку если обязательные поля пусты."""
    base = _env_str("ROCKETCHAT_BASE_URL").rstrip("/")
    user = _env_str("ROCKETCHAT_USER")
    password = os.getenv("ROCKETCHAT_PASSWORD") or ""
    missing = [n for n, v in (
        ("ROCKETCHAT_BASE_URL", base),
        ("ROCKETCHAT_USER", user),
        ("ROCKETCHAT_PASSWORD", password),
    ) if not v]
    if missing:
        raise RuntimeError(
            f"Не заданы переменные окружения Rocket.Chat: {', '.join(missing)}"
        )

    return RocketChatConfig(
        base_url=base,
        user=user,
        password=password,
        timeout_connect_s=_env_float("RC_TIMEOUT_CONNECT_S", 10.0),
        timeout_read_s=_env_float("RC_TIMEOUT_READ_S", 60.0),
        probe_timeout_s=_env_float("RC_PROBE_TIMEOUT_S", 8.0),
        probe_cache_ttl_s=_env_float("RC_PROBE_CACHE_TTL_S", 30.0),
        catalog_fetch_timeout_s=_env_float("RC_CATALOG_FETCH_TIMEOUT_S", 25.0),
        search_timeout_s=_env_float("RC_SEARCH_TIMEOUT_S", 20.0),
        thread_timeout_s=_env_float("RC_THREAD_TIMEOUT_S", 15.0),
        http_concurrency=_env_int("RC_HTTP_CONCURRENCY", 12),
        catalog_ttl_s=_env_int("RC_CATALOG_TTL_S", 600),
        catalog_page_size=_env_int("RC_CATALOG_PAGE_SIZE", 50),
        catalog_max_rooms=_env_int("RC_CATALOG_MAX_ROOMS", 200),
        always_search_room_names=_env_csv(
            "RC_ALWAYS_SEARCH_ROOMS",
            RocketChatConfig.always_search_room_names,
        ),
        max_subqueries=_env_int("RC_MAX_SUBQUERIES", 2),
        rooms_per_subquery=_env_int("RC_ROOMS_PER_SUBQUERY", 6),
        queries_per_subquery=_env_int("RC_QUERIES_PER_SUBQUERY", 2),
        search_count=_env_int("RC_SEARCH_COUNT", 20),
        rerank_top_n=_env_int("RC_RERANK_TOP_N", 8),
        rerank_max_candidates=_env_int("RC_RERANK_MAX_CANDIDATES", 15),
        rerank_mode=_env_rerank_mode("RC_RERANK_MODE", "hybrid"),
        bm25_weight=_env_float("RC_BM25_WEIGHT", 0.4),
        thread_expand_top=_env_int("RC_THREAD_EXPAND_TOP", 2),
        thread_messages_count=_env_int("RC_THREAD_MESSAGES_COUNT", 30),
        reranker_model=_env_str("RC_RERANKER_MODEL", "qilowoq/bge-reranker-v2-m3-en-ru"),
    )
