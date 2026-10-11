from __future__ import annotations

import asyncio
from typing import Any

import pytest

from connector.runtimes.acp.client import AcpClient, AcpRpcError


def _client(on_notification=None, on_request=None):
    async def no_notification(method: str, params: Any) -> None:
        pass

    async def no_exit(code: int | None) -> None:
        pass

    return AcpClient(
        command="acp-agent",
        client_version="1.0",
        startup_timeout=1.0,
        request_timeout=1.0,
        notification_handler=on_notification or no_notification,
        request_handler=on_request,
        exit_handler=no_exit,
    )


def test_response_frame_resolves_pending_future() -> None:
    async def run() -> None:
        client = _client()
        loop = asyncio.get_running_loop()
        future = loop.create_future()
        client._pending["1"] = future
        client._handle_frame(b'{"jsonrpc":"2.0","id":"1","result":{"ok":true}}')
        assert await future == {"ok": True}
    asyncio.run(run())


def test_error_frame_sets_exception() -> None:
    async def run() -> None:
        client = _client()
        loop = asyncio.get_running_loop()
        future = loop.create_future()
        client._pending["1"] = future
        client._handle_frame(b'{"jsonrpc":"2.0","id":"1","error":{"code":-32602,"message":"bad params"}}')
        error = future.exception()
        assert isinstance(error, AcpRpcError)
        assert error.rpc_code == -32602
    asyncio.run(run())


def test_notification_dispatch() -> None:
    async def run() -> None:
        seen: list[tuple[str, dict]] = []

        async def on_notification(method: str, params: Any) -> None:
            seen.append((method, params))

        client = _client(on_notification=on_notification)
        client.initialize_result = {}
        client._handle_frame(b'{"jsonrpc":"2.0","method":"session/update","params":{"sessionId":"s1","update":{"sessionUpdate":"usage_update"}}}')
        await asyncio.sleep(0)
        assert seen == [("session/update", {"sessionId": "s1", "update": {"sessionUpdate": "usage_update"}})]
    asyncio.run(run())


def test_early_notifications_are_buffered_until_initialize() -> None:
    async def run() -> None:
        seen: list[str] = []

        async def on_notification(method: str, params: Any) -> None:
            seen.append(method)

        client = _client(on_notification=on_notification)
        client._handle_frame(b'{"jsonrpc":"2.0","method":"session/update","params":{"sessionId":"s1","update":{"sessionUpdate":"usage_update"}}}')
        assert client._early_notifications == [("session/update", {"sessionId": "s1", "update": {"sessionUpdate": "usage_update"}})]
        assert seen == []
    asyncio.run(run())


def test_invalid_frame_raises() -> None:
    client = _client()
    with pytest.raises(RuntimeError):
        client._handle_frame(b'{"jsonrpc":"1.0"}')
    with pytest.raises(ValueError):
        client._handle_frame(b'not json')
