"""Suppress successful health access logs while keeping failures visible."""
from __future__ import annotations

import logging

HEALTH_PATHS = frozenset(("/health", "/healthz", "/health/live", "/health/ready"))


def quiet_health_request(method: str, path: str, status: int) -> bool:
    return method == "GET" and path.split("?", 1)[0] in HEALTH_PATHS and 200 <= status < 300


class SuccessfulHealthAccessFilter(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        args = record.args
        # Uvicorn access records: client, method, path, HTTP version, status.
        if isinstance(args, tuple) and len(args) == 5:
            _, method, path, _, status = args
            if isinstance(method, str) and isinstance(path, str):
                try:
                    return not quiet_health_request(method, path, int(status))
                except (TypeError, ValueError):
                    pass
        return True


def install_health_access_filter() -> None:
    logger = logging.getLogger("uvicorn.access")
    if not any(isinstance(item, SuccessfulHealthAccessFilter) for item in logger.filters):
        logger.addFilter(SuccessfulHealthAccessFilter())
