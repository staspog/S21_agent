from langchain_core.messages import AIMessage, HumanMessage

from src.campus_intent import infer_search_scope
from src.followup_intent import (
    is_short_contextual_followup,
    resolve_contextual_followup,
    resolve_followup_topic,
)


def test_short_contextual_followup_after_guest_question():
    messages = [
        HumanMessage(content="Как отправить заявку на гостя?"),
        AIMessage(content="Общий порядок и формы по кампусам."),
        HumanMessage(content="что в кампусе?"),
    ]
    assert is_short_contextual_followup("что в кампусе?")
    topic = resolve_contextual_followup(messages)
    assert topic is not None
    assert "заявку на гостя" in topic
    assert "что в кампусе?" in topic
    assert resolve_followup_topic(messages) == topic


def test_contextual_followup_scope_is_general_not_today():
    messages = [
        HumanMessage(content="Как отправить заявку на гостя?"),
        AIMessage(content="Ответ."),
        HumanMessage(content="что в кампусе?"),
    ]
    topic = resolve_contextual_followup(messages)
    assert topic is not None
    assert infer_search_scope(topic) == "general"
