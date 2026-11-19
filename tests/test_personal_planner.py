import unittest
from types import SimpleNamespace
from unittest.mock import patch

from pydantic import ValidationError
from langchain_gigachat.chat_models import GigaChat

from src.personal_planner import InvalidReadPlan, PlannedRead, ReadPlan, plan_reads, validate_read_plan


class PersonalPlannerTest(unittest.IsolatedAsyncioTestCase):
    def test_gigachat_accepts_structured_planner_schema(self):
        model = GigaChat(credentials="synthetic", scope="GIGACHAT_API_CORP")
        self.assertIsNotNone(model.with_structured_output(ReadPlan))

    def test_rejects_unlisted_tool_and_fabricated_nickname(self):
        with self.assertRaises(ValidationError):
            PlannedRead(name="delete_slot")
        with self.assertRaises(InvalidReadPlan):
            validate_read_plan(ReadPlan(calls=[PlannedRead(name="get_public_profile", nickname="other")]), "покажи профиль someone")

    def test_public_nickname_must_be_explicit(self):
        plan = ReadPlan(calls=[PlannedRead(name="get_public_profile", nickname="peer.name")])
        self.assertEqual(validate_read_plan(plan, "Покажи профиль peer.name"), plan)
        with self.assertRaises(InvalidReadPlan):
            validate_read_plan(plan, "Покажи профиль peer.name-other")

    def test_duplicate_and_extra_nickname_are_rejected(self):
        with self.assertRaises(InvalidReadPlan):
            validate_read_plan(ReadPlan(calls=[PlannedRead(name="get_my_profile"), PlannedRead(name="get_my_profile")]), "мой уровень")
        with self.assertRaises(InvalidReadPlan):
            validate_read_plan(ReadPlan(calls=[PlannedRead(name="get_my_profile", nickname="other")]), "other")

    async def test_multiple_reads_and_no_token_in_prompt(self):
        result = ReadPlan(calls=[PlannedRead(name="get_my_profile"), PlannedRead(name="get_my_projects")])
        captured = []
        def fake_invoke(func, messages, **kwargs):
            captured.append(messages)
            return result
        llm = SimpleNamespace(with_structured_output=lambda schema: SimpleNamespace(invoke=lambda *_: None))
        with patch("src.personal_planner.invoke_with_retry", side_effect=fake_invoke):
            plan = await plan_reads(llm, "Мой уровень и текущие проекты?")
        self.assertEqual([x.name for x in plan.calls], ["get_my_profile", "get_my_projects"])
        self.assertNotIn("Bearer", repr(captured))

    async def test_model_failure_is_not_treated_as_empty_data(self):
        llm = SimpleNamespace(with_structured_output=lambda schema: SimpleNamespace(invoke=lambda *_: None))
        with patch("src.personal_planner.invoke_with_retry", side_effect=RuntimeError("secret")):
            with self.assertRaises(InvalidReadPlan):
                await plan_reads(llm, "Мой уровень?")
