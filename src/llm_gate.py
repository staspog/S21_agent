"""Единая «дверца» для всех LLM-вызовов в графе.

Зачем: LangGraph фан-аут (`Send`) запускает N веток `per_subquery` в параллель,
каждая из которых делает 2 LLM-вызова (`select_rooms` → `expand_query`). Плюс
финальный `generate`. На GigaChat это превращается в пиковый бёрст в одну
миллисекунду и ломается о 429 «Too Many Requests».

Решение системное:
  • `LLMRateLimiter` — process-wide thread-safe gate. Перед каждым вызовом
    обеспечивает минимальный интервал между фактическими отправками
    (по монотонным часам), а также удерживает не более N одновременных
    запросов через `BoundedSemaphore`.
  • `invoke_with_retry` — обёртка с экспоненциальным backoff именно на
    `RateLimitError(429)`. Учитывает заголовки `Retry-After` /
    `x-ratelimit-reset`, если их прислал сервер.

Работает одинаково для sync и для async-узлов: sync-узлы вызывают через
`run_in_executor`, и `LLMRateLimiter` синхронизирует их по threading.Lock.
"""

from __future__ import annotations

import logging
import os
import random
import threading
import time
from typing import Any, Callable, TypeVar

log = logging.getLogger("s21.llm_gate")

T = TypeVar("T")


def _env_float(name: str, default: float) -> float:
    raw = os.getenv(name)
    if raw is None or not raw.strip():
        return default
    try:
        return float(raw)
    except ValueError:
        return default


def _env_int(name: str, default: int) -> int:
    raw = os.getenv(name)
    if raw is None or not raw.strip():
        return default
    try:
        return int(raw)
    except ValueError:
        return default


class LLMRateLimiter:
    """Token-bucket-like ограничитель: минимальный интервал + max concurrency.

    Параметры (env):
      LLM_MIN_INTERVAL_S    — мин. пауза между УХОДАМИ запросов (сек, default 0.35)
      LLM_MAX_CONCURRENCY   — сколько LLM-вызовов одновременно держим в полёте
                              (default 2). Меньше — мягче по 429, больше — быстрее.
      LLM_JITTER_S          — случайный jitter, добавляемый к мин-интервалу
                              (default 0.05). Раскидывает идеально синхронные
                              старты во времени.
    """

    def __init__(
        self,
        *,
        min_interval_s: float | None = None,
        max_concurrency: int | None = None,
        jitter_s: float | None = None,
    ) -> None:
        self._min_interval = (
            min_interval_s
            if min_interval_s is not None
            else _env_float("LLM_MIN_INTERVAL_S", 0.35)
        )
        self._jitter = (
            jitter_s if jitter_s is not None else _env_float("LLM_JITTER_S", 0.05)
        )
        cc = (
            max_concurrency
            if max_concurrency is not None
            else _env_int("LLM_MAX_CONCURRENCY", 2)
        )
        self._sema = threading.BoundedSemaphore(max(1, cc))
        self._lock = threading.Lock()
        self._last_call = 0.0
        log.info(
            "LLMRateLimiter: min_interval=%.3fs jitter=%.3fs concurrency=%s",
            self._min_interval,
            self._jitter,
            cc,
        )

    def acquire(self) -> None:
        """Блокирует поток, пока не безопасно отправлять очередной запрос."""
        self._sema.acquire()
        try:
            with self._lock:
                now = time.monotonic()
                spacing = self._min_interval + (
                    random.random() * self._jitter if self._jitter > 0 else 0.0
                )
                wait = max(0.0, self._last_call + spacing - now)
                if wait > 0:
                    # держим лок, чтобы соседние потоки не пролезли с тем же
                    # last_call и не запустились одновременно
                    time.sleep(wait)
                # фиксируем время фактического «ухода» запроса в сеть
                self._last_call = time.monotonic()
        except BaseException:
            # семафор отдаём только при ошибке здесь; в нормальном пути его
            # отдаст release() из контекстного менеджера ниже
            self._sema.release()
            raise

    def release(self) -> None:
        try:
            self._sema.release()
        except ValueError:
            # двойной release — игнорируем, такое возможно при гонке исключений
            pass


_DEFAULT_GATE: LLMRateLimiter | None = None
_DEFAULT_GATE_LOCK = threading.Lock()


def get_default_gate() -> LLMRateLimiter:
    """Process-wide singleton. Все LLM-вызовы графа идут через него."""
    global _DEFAULT_GATE
    with _DEFAULT_GATE_LOCK:
        if _DEFAULT_GATE is None:
            _DEFAULT_GATE = LLMRateLimiter()
        return _DEFAULT_GATE


def _retry_after_seconds(exc: BaseException) -> float | None:
    """Достаёт паузу из заголовков 429 ответа GigaChat (если есть).

    Поддерживаем стандартный `Retry-After` (секунды или HTTP-date — последний
    нам не нужен) и распространённый `x-ratelimit-reset` (epoch seconds или
    «осталось секунд»).
    """
    headers = getattr(exc, "headers", None)
    if headers is None:
        return None
    try:
        get = headers.get  # httpx.Headers / dict-like
    except AttributeError:
        return None

    raw = get("retry-after") or get("Retry-After")
    if raw:
        try:
            return max(0.0, float(raw))
        except (TypeError, ValueError):
            pass

    raw = get("x-ratelimit-reset")
    if raw:
        try:
            v = float(raw)
        except (TypeError, ValueError):
            return None
        # эвристика: epoch seconds vs «осталось»
        now = time.time()
        if v > now:
            return max(0.0, v - now)
        return max(0.0, v)
    return None


def _is_rate_limit(exc: BaseException) -> bool:
    """Распознаём 429 — без жёсткого импорта класса исключения."""
    if exc.__class__.__name__ == "RateLimitError":
        return True
    status = getattr(exc, "status_code", None)
    if status == 429:
        return True
    msg = str(exc)
    return "429" in msg and ("Too Many Requests" in msg or "rate" in msg.lower())


def invoke_with_retry(
    func: Callable[..., T],
    *args: Any,
    gate: LLMRateLimiter | None = None,
    max_retries: int | None = None,
    base_backoff_s: float | None = None,
    max_backoff_s: float | None = None,
    label: str = "llm",
    **kwargs: Any,
) -> T:
    """Вызвать LLM-callable через gate с retry на 429.

    Ретраим ТОЛЬКО RateLimitError(429). Остальные ошибки прокидываем сразу,
    чтобы не маскировать таймауты/auth/etc.

    backoff: base * 2^attempt + jitter, ограниченный max_backoff. Учитываем
    `Retry-After` из заголовков 429 — если он больше посчитанного backoff,
    спим столько, сколько просит сервер.
    """
    g = gate or get_default_gate()
    retries = (
        max_retries if max_retries is not None else _env_int("LLM_MAX_RETRIES", 2)
    )
    base = (
        base_backoff_s
        if base_backoff_s is not None
        else _env_float("LLM_RETRY_BACKOFF_S", 1.5)
    )
    cap = (
        max_backoff_s
        if max_backoff_s is not None
        else _env_float("LLM_RETRY_BACKOFF_MAX_S", 8.0)
    )

    attempt = 0
    while True:
        g.acquire()
        backoff = 0.0
        retryable = False
        try:
            return func(*args, **kwargs)
        except Exception as e:
            if not _is_rate_limit(e) or attempt >= retries:
                raise
            retryable = True
            wait_hint = _retry_after_seconds(e)
            backoff = min(cap, base * (2**attempt))
            backoff += random.random() * 0.5  # jitter
            if wait_hint is not None:
                backoff = max(backoff, wait_hint)
            log.warning(
                "%s: 429 (attempt %s/%s), backoff %.2fs",
                label,
                attempt + 1,
                retries,
                backoff,
            )
        finally:
            g.release()

        # снаружи семафора, чтобы соседние ветки могли проехать пока мы ждём
        if not retryable:
            # сюда не попадаем (return/raise выше) — но на всякий случай
            raise RuntimeError("invoke_with_retry: unexpected fallthrough")
        time.sleep(backoff)
        attempt += 1


__all__ = [
    "LLMRateLimiter",
    "get_default_gate",
    "invoke_with_retry",
]
