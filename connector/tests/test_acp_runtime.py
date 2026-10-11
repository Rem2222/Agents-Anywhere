from __future__ import annotations

import asyncio
from typing import Any

from connector.runtime_protocol import RuntimeConfig
from connector.runtimes.acp.runtime import AcpRuntime


class _FakeClient:
    connected = True

    def __init__(self) -> None:
        self.requests: list[tuple[str, dict]] = []

    async def request(self, method: str, params: Any = None, *, timeout: float | None = None) -> Any:
        self.requests.append((method, dict(params or {})))
        if method == "session/new":
            return {
                "sessionId": "ext-1",
                "models": {"availableModels": [{"modelId": "m1", "name": "Model"}], "currentModelId": "m1"},
                "modes": {"availableModes": [{"id": "default", "name": "Default"}], "currentModeId": "default"},
            }
        if method == "session/prompt":
            return {"stopReason": "end_turn"}
        if method == "session/list":
            return {"sessions": [{"sessionId": "ext-1", "title": "t", "cwd": "/x", "updatedAt": "2026-01-01T00:00:00Z"}]}
        if method == "session/load":
            return {}
        if method in {"session/cancel", "session/set_model", "session/set_mode"}:
            return {}
        raise AssertionError(f"unexpected method {method}")


class _Host:
    connector_id = "test"

    def __init__(self) -> None:
        self.meta_upserts: list[tuple[str, str | None, str | None]] = []
        self.state_updates: list[tuple[str, str | None, str | None]] = []
        self.timeline: list[Any] = []
        self.turn_ends: list[tuple[str, str | None, str]] = []

    async def session_meta_upsert(self, session_id, runtime, external_session_id=None, title=None, cwd=None, ordering_time=None, metadata=None):
        self.meta_upserts.append((session_id, external_session_id, title))

    async def session_state_update(self, session_id, runtime, status=None, selections=None, external_session_id=None, status_reason=None, error=None, metadata=None):
        self.state_updates.append((session_id, status, external_session_id))

    async def timeline_item_upsert(self, item):
        self.timeline.append(item)

    async def timeline_sync(self, *args, **kwargs):
        pass

    async def session_turn_ended(self, session_id, runtime, external_session_id=None, turn_id=None, outcome="completed", metadata=None):
        self.turn_ends.append((session_id, turn_id, outcome))

    async def notice_upsert(self, notice):
        pass

    async def model_catalog_update(self, catalog):
        pass

    async def permission_catalog_update(self, catalog):
        pass

    async def runtime_capabilities_update(self, capabilities):
        pass

    async def runtime_health_update(self, *args, **kwargs):
        pass

    async def runtime_error(self, *args, **kwargs):
        pass

    async def attachment_download(self, *args, **kwargs):
        raise AssertionError("not expected")

    async def sync_state_read(self, key):
        return None

    async def sync_state_write(self, key, value):
        pass

    async def sync_state_delete(self, key):
        pass


def _runtime() -> tuple[AcpRuntime, _Host, _FakeClient]:
    config = RuntimeConfig(
        "acp",
        1,
        {
            "command": "hermes-acp",
            "agentHome": "/tmp/home",
            "cwd": "/tmp/home",
            "startupTimeoutMs": 30000,
            "requestTimeoutMs": 600000,
            "permissionTimeoutMs": 300000,
            "idleTimeoutSeconds": 0,
            "maxRestartAttempts": 3,
            "restartBackoffMs": 1000,
        },
    )
    host = _Host()
    runtime = AcpRuntime(config, host)
    client = _FakeClient()
    runtime._client = client
    return runtime, host, client


def test_create_and_start_session_wires_native_id() -> None:
    async def run() -> None:
        runtime, host, client = _runtime()
        result = await runtime.create_and_start_session("sess_1", "hello", client_message_id="cm1")
        assert result.ok is True
        assert host.meta_upserts[0][0] == "sess_1"
        assert host.meta_upserts[0][1] == "ext-1"
        methods = [method for method, _ in client.requests]
        assert methods == ["session/new", "session/prompt"]
        # The prompt carries the native (external) session id and a text block.
        prompt = next(params for method, params in client.requests if method == "session/prompt")
        assert prompt["sessionId"] == "ext-1"
        assert prompt["prompt"] == [{"type": "text", "text": "hello"}]
        # Turn boundaries and meta were published.
        assert any(item.type == "turn.start" for item in host.timeline)
        assert any(item.type == "turn.end" for item in host.timeline)
        assert host.turn_ends and host.turn_ends[0][1] is not None
        # Idle after the turn.
        assert host.state_updates[-1][1] == "idle"
    asyncio.run(run())


def test_list_sessions_and_interrupt_resolve_native_id() -> None:
    async def run() -> None:
        runtime, _host, client = _runtime()
        sessions = await runtime.list_sessions()
        assert sessions[0].external_session_id == "ext-1"
        # Interrupt resolves the platform id through the inventory-registered mapping.
        result = await runtime.interrupt_session(sessions[0].session_id)
        assert result.ok is True
        assert any(method == "session/cancel" and params["sessionId"] == "ext-1" for method, params in client.requests)
    asyncio.run(run())


def test_update_session_selections_sets_model() -> None:
    async def run() -> None:
        runtime, _host, client = _runtime()
        # Seed the catalog selection id registry from a session response.
        await runtime.create_and_start_session("sess_1", "hi", client_message_id="cm1")
        from connector.runtimes.acp import catalogs

        sel = catalogs.model_selection_id("m1")
        result = await runtime.update_session_selections("sess_1", "ext-1", {"model": sel})
        assert result.ok is True
        assert any(method == "session/set_model" and params["modelId"] == "m1" for method, params in client.requests)
    asyncio.run(run())


def test_start_turn_requires_stable_message_id() -> None:
    async def run() -> None:
        runtime, _host, client = _runtime()
        from connector.runtime_protocol import RuntimeInvalidRequestError

        try:
            await runtime.start_turn("sess_1", "ext-1", "hello")
            raise AssertionError("expected RuntimeInvalidRequestError")
        except RuntimeInvalidRequestError:
            pass
        assert client.requests == []
    asyncio.run(run())
