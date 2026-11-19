import unittest
import json
import httpx
import os
from datetime import datetime, timedelta
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

from fastapi import HTTPException
from starlette.requests import Request
from langchain_core.messages import HumanMessage

os.environ["GIGACHAT_API_KEY"] = "synthetic"
os.environ["ROCKETCHAT_BASE_URL"] = "https://example.invalid"
os.environ["ROCKETCHAT_USER"] = "synthetic"
os.environ["ROCKETCHAT_PASSWORD"] = "synthetic"

from src import api
from src.graph import build_rc_graph
from src.personal_mcp import PlatformFailure
from src.personal_intent import personal_kind
from src.personal_planner import ReadPlan, PlannedRead


def request_with_token(token: str = "") -> Request:
    headers = [(b"authorization", token.encode())] if token else []
    return Request({"type": "http", "method": "POST", "path": "/ask", "headers": headers})


def mock_mcp(req):
    if req.method != "POST":
        return httpx.Response(405)
    body = json.loads(req.content)
    method = body["method"]
    if "id" not in body:
        return httpx.Response(202)
    if method == "initialize":
        result = {"protocolVersion": "2025-06-18", "capabilities": {"tools": {}}, "serverInfo": {"name": "test", "version": "1"}}
    elif method == "tools/call":
        if body["params"]["name"] == "get_my_profile":
            result = {"content": [], "structuredContent": {"status": "ok", "profile": {"level": 7, "xp": 123}}}
        else:
            day = datetime.fromisoformat(body["params"]["arguments"]["start"])
            start = day + timedelta(hours=12, minutes=15)
            result = {"content": [], "structuredContent": {"status": "ok", "items": [{"type": "free_slot", "startTime": start.isoformat(), "endTime": (start + timedelta(minutes=15)).isoformat()}]}}
    elif method == "tools/list":
        result = {"tools": [{"name": "get_my_calendar", "inputSchema": {"type": "object"}}, {"name": "get_my_profile", "inputSchema": {"type": "object"}}]}
    else:
        return httpx.Response(400)
    return httpx.Response(200, json={"jsonrpc": "2.0", "id": body["id"], "result": result})


class PersonalAPITest(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        async def fake_plan(_llm, question, history=()):
            kind = personal_kind(question)
            names = {"calendar": "get_my_calendar", "projects": "get_my_projects", "profile": "get_my_profile"}
            return ReadPlan(calls=[PlannedRead(name=names[kind])] if kind else [])
        self.planner = patch.object(api, "plan_reads", side_effect=fake_plan)
        self.planner.start()
        self.addCleanup(self.planner.stop)

    async def test_profile_question_fetches_mcp_without_leaking_token(self):
        real_client = httpx.AsyncClient
        def upstream(req):
            if req.url.path == "/v1/assistant/identity":
                return httpx.Response(200, json={"subject": "synthetic-user"})
            if req.url.path == "/v1/assistant/mcp":
                return mock_mcp(req)
            return httpx.Response(404)
        def make_client(*args, **kwargs):
            return real_client(*args, transport=httpx.MockTransport(upstream), **kwargs)
        graph = SimpleNamespace(ainvoke=AsyncMock(return_value={"messages": [SimpleNamespace(content="Уровень: 7")], "final_sources": []}))
        with patch.object(api.httpx, "AsyncClient", side_effect=make_client), patch.object(api, "graph", graph):
            answer = await api.ask_question(api.QueryRequest(question="Какой у меня уровень?"), request_with_token("Bearer synthetic-token"))
        self.assertEqual(answer.answer, "Уровень: 7")
        state = graph.ainvoke.call_args.args[0]
        self.assertEqual(state["personal_context"]["profile"]["level"], 7)
        self.assertNotIn("synthetic-token", repr(state))

    async def test_multi_tool_plan_aggregates_only_authenticated_results(self):
        real_client = httpx.AsyncClient
        def upstream(req):
            if req.url.path == "/v1/assistant/identity":
                return httpx.Response(200, json={"subject": "synthetic-user"})
            if req.url.path == "/v1/assistant/mcp":
                self.assertEqual(req.headers.get("Authorization"), "Bearer synthetic-token")
                return mock_mcp(req)
            return httpx.Response(404)
        def make_client(*args, **kwargs):
            return real_client(*args, transport=httpx.MockTransport(upstream), **kwargs)
        graph = SimpleNamespace(ainvoke=AsyncMock(return_value={"messages": [SimpleNamespace(content="ok")], "final_sources": []}))
        plan=ReadPlan(calls=[PlannedRead(name="get_my_profile"),PlannedRead(name="get_my_calendar")])
        with patch.object(api.httpx, "AsyncClient", side_effect=make_client), patch.object(api, "graph", graph), patch.object(api, "plan_reads", AsyncMock(return_value=plan)):
            await api.ask_question(api.QueryRequest(question="Мой уровень и что у меня сегодня?"), request_with_token("Bearer synthetic-token"))
        context=graph.ainvoke.call_args.args[0]["personal_context"]
        self.assertEqual(context["kind"],"multi")
        self.assertEqual([item["kind"] for item in context["results"]],["profile","calendar"])
        self.assertNotIn("synthetic-token",repr(context))

    async def test_missing_token_is_rejected_before_graph(self):
        with patch.object(api, "graph") as graph:
            with self.assertRaises(HTTPException) as error:
                await api.ask_question(api.QueryRequest(question="hello"), request_with_token())
            self.assertEqual(error.exception.status_code, 401)
            graph.ainvoke.assert_not_called()

    async def test_invalid_platform_token_is_rejected(self):
        with patch.object(api, "verified_subject", AsyncMock(side_effect=PlatformFailure("unauthorized"))), patch.object(api, "graph") as graph:
            with self.assertRaises(HTTPException) as error:
                await api.ask_question(api.QueryRequest(question="hello"), request_with_token("Bearer test-token"))
            self.assertEqual(error.exception.status_code, 401)
            graph.ainvoke.assert_not_called()

    async def test_same_session_id_is_scoped_to_verified_user(self):
        fake_graph = SimpleNamespace(ainvoke=AsyncMock(return_value={"messages": [SimpleNamespace(content="ok")], "final_sources": []}))
        with patch.object(api, "verified_subject", AsyncMock(side_effect=["user-a", "user-b"])), patch.object(api, "graph", fake_graph):
            body = api.QueryRequest(question="hello", session_id="same-session")
            await api.ask_question(body, request_with_token("Bearer token-a"))
            await api.ask_question(body, request_with_token("Bearer token-b"))
        first, second = fake_graph.ainvoke.call_args_list
        self.assertNotEqual(first.args[1]["configurable"]["thread_id"], second.args[1]["configurable"]["thread_id"])
        self.assertNotIn("token-a", repr(first))
        self.assertNotIn("token-b", repr(second))

    async def test_planner_history_is_scoped_to_verified_user(self):
        snapshots=[]
        async def get_state(config):
            key=config["configurable"]["thread_id"]
            snapshots.append(key)
            return SimpleNamespace(values={"messages":[HumanMessage(content="история-"+key[:8])]})
        graph=SimpleNamespace(aget_state=AsyncMock(side_effect=get_state),ainvoke=AsyncMock(return_value={"messages":[SimpleNamespace(content="ok")],"final_sources":[]}))
        planner=AsyncMock(return_value=ReadPlan())
        with patch.object(api,"verified_subject",AsyncMock(side_effect=["user-a","user-b"])), patch.object(api,"graph",graph), patch.object(api,"plan_reads",planner):
            body=api.QueryRequest(question="hello",session_id="same-session")
            await api.ask_question(body,request_with_token("Bearer token-a"))
            await api.ask_question(body,request_with_token("Bearer token-b"))
        self.assertNotEqual(snapshots[0],snapshots[1])
        self.assertNotEqual(planner.await_args_list[0].args[2],planner.await_args_list[1].args[2])

    async def test_invalid_timezone_does_not_call_platform(self):
        with patch.object(api, "verified_subject", AsyncMock()) as verify:
            with self.assertRaises(HTTPException) as error:
                await api.ask_question(api.QueryRequest(question="hello", timezone="invalid/zone"), request_with_token("Bearer token"))
            self.assertEqual(error.exception.status_code, 400)
            verify.assert_not_called()

    async def test_platform_and_auth_hosts_are_not_api_service(self):
        for url in ("https://platform.21-school.ru", "https://AUTH.21-school.ru."):
            with self.subTest(url=url), patch.dict(os.environ, {"S21_SERVER_API_BASE": url}), patch.object(api, "verified_subject", AsyncMock()) as verify:
                with self.assertRaises(HTTPException) as error:
                    await api.ask_question(api.QueryRequest(question="Какой у меня уровень?"), request_with_token("Bearer synthetic"))
                self.assertEqual(error.exception.status_code, 503)
                self.assertEqual(error.exception.detail, "identity_service_misconfigured")
                verify.assert_not_called()

    async def test_mcp_transport_rejection_propagates_unauthorized(self):
        real_client = httpx.AsyncClient
        def upstream(req):
            if req.url.path == "/v1/assistant/identity":
                return httpx.Response(200, json={"subject": "synthetic-user"})
            return httpx.Response(401)
        def make_client(*args, **kwargs):
            return real_client(*args, transport=httpx.MockTransport(upstream), **kwargs)
        with patch.object(api.httpx, "AsyncClient", side_effect=make_client), patch.object(api, "graph") as graph:
            with self.assertRaises(HTTPException) as error:
                await api.ask_question(api.QueryRequest(question="Что у меня сегодня?"), request_with_token("Bearer synthetic-token"))
            self.assertEqual(error.exception.status_code, 401)
            self.assertNotIn("synthetic-token", str(error.exception))
            graph.ainvoke.assert_not_called()

    async def test_personal_calendar_flows_through_private_mcp(self):
        def upstream(req):
            if req.url.path == "/v1/assistant/identity":
                return httpx.Response(200, json={"subject": "test-user"})
            if req.url.path == "/v1/assistant/mcp":
                return mock_mcp(req)
            if req.url.path == "/getCalendar":
                day = datetime.fromisoformat(req.url.params["from"])
                start = day + timedelta(hours=12, minutes=15)
                return httpx.Response(200, json={
                    "reviewList": [], "presentationList": [],
                    "freeSlotsList": [{"id": "slot", "startTime": start.isoformat(), "endTime": (start + timedelta(minutes=15)).isoformat()}],
                    "eventsList": [], "p2pRequestsList": [],
                })
            return httpx.Response(404)

        real_client = httpx.AsyncClient
        def make_client(*args, **kwargs):
            return real_client(*args, transport=httpx.MockTransport(upstream), **kwargs)

        fake_graph = SimpleNamespace(ainvoke=AsyncMock(return_value={"messages": [SimpleNamespace(content="ok")], "final_sources": []}))
        with patch.object(api.httpx, "AsyncClient", side_effect=make_client), patch.object(api, "graph", fake_graph):
            await api.ask_question(api.QueryRequest(question="Что у меня сегодня?", timezone="Europe/Moscow"), request_with_token("Bearer synthetic-token"))
        state = fake_graph.ainvoke.call_args.args[0]
        item = state["personal_context"]["items"][0]
        self.assertEqual(item["type"], "free_slot")
        self.assertTrue(item["startTime"].endswith("T12:15:00+03:00"))
        self.assertNotIn("synthetic-token", repr(state))

    async def test_calendar_reaches_real_graph_with_model_stub(self):
        def upstream(req):
            if req.url.path == "/v1/assistant/identity":
                return httpx.Response(200, json={"subject": "synthetic-user"})
            if req.url.path == "/v1/assistant/mcp":
                return mock_mcp(req)
            if req.url.path == "/getCalendar":
                day = datetime.fromisoformat(req.url.params["from"])
                start = day + timedelta(hours=12, minutes=15)
                return httpx.Response(200, json={
                    "reviewList": [], "presentationList": [],
                    "freeSlotsList": [{"id": "slot", "startTime": start.isoformat(), "endTime": (start + timedelta(minutes=15)).isoformat()}],
                    "eventsList": [], "p2pRequestsList": [],
                })
            return httpx.Response(404)

        real_client = httpx.AsyncClient
        def make_client(*args, **kwargs):
            return real_client(*args, transport=httpx.MockTransport(upstream), **kwargs)

        def model_stub(**kwargs):
            slot = kwargs["personal_context"]["items"][0]
            self.assertEqual(slot["type"], "free_slot")
            self.assertTrue(slot["startTime"].endswith("T12:15:00+03:00"))
            return "У тебя свободный слот в 12:15.", []

        graph = build_rc_graph(llm=api.llm, rc_client=api.rc_client, rc_catalog=api.rc_catalog,
                               rc_cfg=api.rc_cfg, rc_availability=api.rc_availability)
        with patch.object(api.httpx, "AsyncClient", side_effect=make_client), patch.object(api, "graph", graph), \
             patch("src.graph.retrieve_rag_chunks", return_value=[]), \
             patch("src.graph.resolve_search_intent", return_value=("Что у меня сегодня?", "today")), \
             patch("src.graph.generate_answer", side_effect=model_stub):
            response = await api.ask_question(api.QueryRequest(question="Что у меня сегодня?"), request_with_token("Bearer synthetic-token"))
        self.assertIn("12:15", response.answer)
