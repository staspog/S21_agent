import unittest
from datetime import datetime, timezone

from src.personal_intent import agenda_bounds, calendar_bounds, day_bounds, personal_kind


class PersonalIntentTest(unittest.TestCase):
    def test_personal_vs_campus(self):
        self.assertEqual(personal_kind("Что у меня сегодня?"), "calendar")
        self.assertEqual(personal_kind("Когда у меня проверка?"), "calendar")
        self.assertEqual(personal_kind("Какие мои проекты?"), "projects")
        self.assertEqual(personal_kind("Какой у меня уровень?"), "profile")
        self.assertEqual(personal_kind("Сколько у меня баллов?"), "profile")
        self.assertEqual(personal_kind("Сколько баллов?"), "profile")
        self.assertEqual(personal_kind("Какой уровень?"), "profile")
        self.assertIsNone(personal_kind("Что сегодня в кампусе?"))
        self.assertIsNone(personal_kind("Как записаться на проверку?"))

    def test_day_bounds_follow_user_zone_and_clock_transition(self):
        start, end = day_bounds("Asia/Tokyo", datetime(2030, 1, 1, 18, tzinfo=timezone.utc))
        self.assertEqual(start, "2030-01-02T00:00:00+09:00")
        self.assertEqual(end, "2030-01-03T00:00:00+09:00")
        start, end = day_bounds("Europe/Berlin", datetime(2030, 3, 31, 10, tzinfo=timezone.utc))
        self.assertEqual(start, "2030-03-31T00:00:00+01:00")
        self.assertEqual(end, "2030-04-01T00:00:00+02:00")

    def test_rejects_invalid_timezone(self):
        with self.assertRaises(ValueError):
            day_bounds("invalid/timezone", datetime.now(timezone.utc))

    def test_week_and_unsupported_periods(self):
        now = datetime(2030, 1, 2, 10, tzinfo=timezone.utc)
        start, end = calendar_bounds("мои проверки на этой неделе", "Europe/Moscow", now)
        self.assertEqual(start, "2029-12-31T00:00:00+03:00")
        self.assertEqual(end, "2030-01-07T00:00:00+03:00")
        with self.assertRaises(ValueError):
            calendar_bounds("мои проверки 12.03", "Europe/Moscow", now)

    def test_agenda_defaults_to_30_local_days(self):
        now = datetime(2030, 3, 15, 10, tzinfo=timezone.utc)
        start, end = agenda_bounds("Мои события", "Europe/Berlin", now)
        self.assertEqual(start, "2030-03-15T00:00:00+01:00")
        self.assertEqual(end, "2030-04-14T00:00:00+02:00")
        with self.assertRaises(ValueError):
            agenda_bounds("Мои события 12.03", "Europe/Moscow", now)


if __name__ == "__main__":
    unittest.main()
