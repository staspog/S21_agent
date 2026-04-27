"""Конфигурация Rocket.Chat и параметров пайплайна — из переменных окружения."""

from __future__ import annotations

import os
from dataclasses import dataclass


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


@dataclass(frozen=True)
class RocketChatConfig:
    """Все параметры для Rocket.Chat-клиента и пайплайна."""

    base_url: str
    user: str
    password: str

    # HTTP
    timeout_connect_s: float = 5.0
    timeout_read_s: float = 30.0
    http_concurrency: int = 8

    # Catalog
    catalog_ttl_s: int = 600
    catalog_page_size: int = 50
    catalog_max_rooms: int = 200
    always_search_room_names: tuple[str, ...] = ("msk_adm",)

    # Pipeline
    max_subqueries: int = 5
    rooms_per_subquery: int = 4
    queries_per_subquery: int = 3
    search_count: int = 30
    rerank_top_n: int = 8
    bm25_weight: float = 0.4
    thread_expand_top: int = 3
    thread_messages_count: int = 80

    # Models
    reranker_model: str = "qilowoq/bge-reranker-v2-m3-en-ru"

    @property
    def web_message_path_for_kind(self) -> dict[str, str]:
        # Rocket.Chat: для public channels url = /channel/<name>?msg=, для private groups = /group/<name>?msg=,
        # для DM — /direct/<userId>?msg=. Используется в build_permalink.
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
        timeout_connect_s=_env_float("RC_TIMEOUT_CONNECT_S", 5.0),
        timeout_read_s=_env_float("RC_TIMEOUT_READ_S", 30.0),
        http_concurrency=_env_int("RC_HTTP_CONCURRENCY", 8),
        catalog_ttl_s=_env_int("RC_CATALOG_TTL_S", 600),
        catalog_page_size=_env_int("RC_CATALOG_PAGE_SIZE", 50),
        catalog_max_rooms=_env_int("RC_CATALOG_MAX_ROOMS", 200),
        always_search_room_names=_env_csv("RC_ALWAYS_SEARCH_ROOMS", ("msk_adm",)),
        max_subqueries=_env_int("RC_MAX_SUBQUERIES", 5),
        rooms_per_subquery=_env_int("RC_ROOMS_PER_SUBQUERY", 4),
        queries_per_subquery=_env_int("RC_QUERIES_PER_SUBQUERY", 3),
        search_count=_env_int("RC_SEARCH_COUNT", 30),
        rerank_top_n=_env_int("RC_RERANK_TOP_N", 8),
        bm25_weight=_env_float("RC_BM25_WEIGHT", 0.4),
        thread_expand_top=_env_int("RC_THREAD_EXPAND_TOP", 3),
        thread_messages_count=_env_int("RC_THREAD_MESSAGES_COUNT", 80),
        reranker_model=_env_str("RC_RERANKER_MODEL", "qilowoq/bge-reranker-v2-m3-en-ru"),
    )
