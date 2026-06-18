from src.pipeline.answer import (
    _dedupe_source_labels,
    _label_from_msg_text,
    _resolve_sources,
    _source_label,
    _source_label_from_hit,
)
from src.rc.schemas import Evidence, Hit, SourceItem


MSK_URL = "https://rocketchat-student.21-school.ru/group/msk_adm?msg=abc"


def _hit(msg: str, *, mid: str = "mid1") -> Hit:
    return Hit(
        mid=mid,
        rid="rid1",
        room_name="msk_adm",
        permalink=MSK_URL,
        msg=msg,
    )


def test_label_from_msg_question_with_q_prefix():
    label = _label_from_msg_text("Q: Как устроено peer-to-peer?")
    assert label == "Как устроено peer-to-peer?"


def test_label_from_msg_long_paragraph_truncated():
    msg = (
        "Принцип peer-to-peer в Школе 21 предполагает взаимную помощь "
        "и ответственность каждого участника сообщества за свой прогресс."
    )
    label = _label_from_msg_text(msg)
    assert "peer-to-peer" in label
    assert len(label) <= 56
    assert label.endswith("…")


def test_source_label_from_hit_empty_msg_uses_subquery():
    hit = _hit("")
    subquery = "Как устроено обучение peer-to-peer?"
    label = _source_label_from_hit(hit, subquery, MSK_URL, 0)
    assert label == subquery


def test_source_label_from_hit_empty_msg_no_subquery_falls_back_to_room():
    hit = _hit("")
    label = _source_label_from_hit(hit, None, MSK_URL, 0)
    assert label == "msk_adm"


def test_dedupe_source_labels_adds_suffix():
    items = [
        SourceItem(url=MSK_URL, label="Один и тот же вопрос?"),
        SourceItem(url=MSK_URL + "2", label="Один и тот же вопрос?"),
    ]
    out = _dedupe_source_labels(items)
    assert out[0].label == "Один и тот же вопрос?"
    assert out[1].label == "Один и тот же вопрос? · 2"


def test_resolve_sources_uses_message_label():
    hit = _hit("Q: Как устроено peer-to-peer?", mid="m1")
    by_mid = {hit.mid: hit}
    evidence = [Evidence(subquery="peer-to-peer обучение", hits=[hit])]
    sources = _resolve_sources(
        cited_mids=[hit.mid],
        by_mid=by_mid,
        evidence_list=evidence,
    )
    assert len(sources) == 1
    assert sources[0].url == MSK_URL
    assert sources[0].label == "Как устроено peer-to-peer?"


def test_source_label_url_fallback_numbered():
    assert _source_label("https://example.com/docs/page", 1) == "Источник 2"
