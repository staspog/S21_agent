import unittest

import httpx
from mcp.shared.memory import create_connected_server_and_client_session

from src.personal_mcp import PlatformFailure, make_personal_mcp, verified_subject


class PersonalMCPTest(unittest.IsolatedAsyncioTestCase):
    async def test_tools_use_request_credentials_without_accepting_user_id(self):
        calls = []

        def upstream(req):
            calls.append((req.url.path, req.headers.get("Authorization")))
            if req.url.path == "/v1/assistant/identity":
                return httpx.Response(200, json={"subject": "synthetic-user"})
            return httpx.Response(200, json={
                "reviewList": [], "presentationList": [],
                "freeSlotsList": [{"id": "private-id", "startTime": "2030-01-01T12:15:00+03:00", "endTime": "2030-01-01T12:30:00+03:00"}],
                "eventsList": [], "p2pRequestsList": [],
            })

        async with httpx.AsyncClient(base_url="https://example.test", transport=httpx.MockTransport(upstream)) as client:
            self.assertEqual(await verified_subject(client, "synthetic-token"), "synthetic-user")
            server = make_personal_mcp(client, "synthetic-token")
            async with create_connected_server_and_client_session(server) as session:
                tools = await session.list_tools()
                self.assertEqual({t.name for t in tools.tools}, {"get_my_calendar", "get_my_projects"})
                for tool in tools.tools:
                    self.assertNotIn("user_id", tool.inputSchema.get("properties", {}))
                    self.assertNotIn("token", tool.inputSchema.get("properties", {}))
                result = await session.call_tool("get_my_calendar", {"start": "2030-01-01T00:00:00+03:00", "end": "2030-01-02T00:00:00+03:00"})
                self.assertFalse(result.isError)
                self.assertEqual(result.structuredContent["items"][0]["type"], "free_slot")
                self.assertNotIn("private-id", str(result))
                self.assertNotIn("synthetic-token", str(result))
        self.assertEqual(calls, [("/v1/assistant/identity", "Bearer synthetic-token"), ("/getCalendar", "Bearer synthetic-token")])

    async def test_rejects_invalid_identity_and_preserves_error_distinction(self):
        def upstream(req):
            return httpx.Response(401 if req.url.path.endswith("identity") else 503)

        async with httpx.AsyncClient(base_url="https://example.test", transport=httpx.MockTransport(upstream)) as client:
            with self.assertRaises(PlatformFailure) as error:
                await verified_subject(client, "synthetic-token")
            self.assertEqual(error.exception.code, "unauthorized")
            async with create_connected_server_and_client_session(make_personal_mcp(client, "synthetic-token")) as session:
                result = await session.call_tool("get_my_projects", {})
                self.assertEqual(result.structuredContent["status"], "unavailable")

    async def test_projects_are_scoped_and_minimized(self):
        def upstream(req):
            token = req.headers.get("Authorization")
            return httpx.Response(200, json={"projects": [{
                "name": "project-a" if token == "Bearer a" else "project-b",
                "goalStatus": "ACTIVE", "description": "unneeded private text",
            }]})

        async with httpx.AsyncClient(base_url="https://example.test", transport=httpx.MockTransport(upstream)) as client:
            for token in ("a", "b"):
                async with create_connected_server_and_client_session(make_personal_mcp(client, token)) as session:
                    result = await session.call_tool("get_my_projects", {})
                    self.assertEqual(result.structuredContent["projects"], [{"name": "project-" + token, "goalStatus": "ACTIVE"}])
                    self.assertNotIn("unneeded private text", str(result))

    async def test_empty_calendar_is_not_a_failed_request(self):
        body = {key: [] for key in ("reviewList", "presentationList", "freeSlotsList", "eventsList", "p2pRequestsList")}
        async with httpx.AsyncClient(base_url="https://example.test", transport=httpx.MockTransport(lambda req: httpx.Response(200, json=body))) as client:
            async with create_connected_server_and_client_session(make_personal_mcp(client, "token")) as session:
                result = await session.call_tool("get_my_calendar", {"start": "2030-01-01T00:00:00Z", "end": "2030-01-02T00:00:00Z"})
                self.assertEqual(result.structuredContent["status"], "ok")
                self.assertEqual(result.structuredContent["items"], [])


if __name__ == "__main__":
    unittest.main()
