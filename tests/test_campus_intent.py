from src.campus_intent import infer_search_scope, is_broad_campus_today


def test_what_in_campus_without_today_is_not_broad_digest():
    assert not is_broad_campus_today("что в кампусе?")
    assert infer_search_scope("что в кампусе?") == "general"


def test_what_today_in_campus_is_broad_digest():
    assert is_broad_campus_today("что сегодня в кампусе?")
    assert infer_search_scope("что сегодня в кампусе?") == "today"
