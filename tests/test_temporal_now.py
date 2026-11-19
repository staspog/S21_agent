from datetime import datetime
from zoneinfo import ZoneInfo

from src import temporal


def test_now_is_not_cached(monkeypatch):
    class Clock:
        calls = 0

        @classmethod
        def now(cls, tz):
            cls.calls += 1
            return datetime(2030, 1, cls.calls, tzinfo=tz)

    monkeypatch.setattr(temporal, "datetime", Clock)
    monkeypatch.setattr(temporal, "agent_timezone", lambda: ZoneInfo("Europe/Moscow"))
    assert temporal.now().day == 1
    assert temporal.now().day == 2
