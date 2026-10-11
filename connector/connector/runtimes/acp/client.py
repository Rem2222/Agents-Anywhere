from __future__ import annotations

import asyncio
import json
from collections.abc import Awaitable, Callable, Mapping
from contextlib import suppress
from typing import Any

from connector.logging import logger

MAX_FRAME_BYTES = 8 * 1024 * 1024
PROTOCOL_VERSION = 1

NotificationHandler = Callable[[str, Mapping[str, Any]], Awaitable[None]]
RequestHandler = Callable[[str, Mapping[str, Any]], Awaitable[Any]]
ExitHandler = Callable[[int | None], Awaitable[None]]


class AcpRpcError(RuntimeError):
    """JSON-RPC error returned by the ACP agent."""

    def __init__(
        self,
        code: int,
        message: str,
        data: Mapping[str, Any] | None = None,
    ) -> None:
        super().__init__(message)
        self.rpc_code = code
        self.data = dict(data or {})
        self.method_code = self.data.get("method") if isinstance(self.data.get("method"), str) else None
        self.retryable = self.data.get("retryable") is True


class AcpNoResponse(Exception):
    """Signal that the connector will not answer an inbound agent request."""


class AcpClient:
    """Connect one ACP agent child process over stdio (NDJSON JSON-RPC)."""

    def __init__(
        self,
        *,
        command: str,
        args: tuple[str, ...] = (),
        env: Mapping[str, str] | None = None,
        cwd: str | None = None,
        client_version: str,
        startup_timeout: float,
        request_timeout: float,
        notification_handler: NotificationHandler,
        request_handler: RequestHandler | None = None,
        exit_handler: ExitHandler,
    ) -> None:
        self.command = command
        self.args = args
        self.env = dict(env or {})
        self.cwd = cwd
        self.client_version = client_version
        self.startup_timeout = startup_timeout
        self.request_timeout = request_timeout
        self.notification_handler = notification_handler
        self.request_handler = request_handler
        self.exit_handler = exit_handler

        self.process: asyncio.subprocess.Process | None = None
        self.initialize_result: dict[str, Any] | None = None
        self._pending: dict[str | int, asyncio.Future[Any]] = {}
        self._request_id = 0
        self._write_lock = asyncio.Lock()
        self._reader_task: asyncio.Task[None] | None = None
        self._stderr_task: asyncio.Task[None] | None = None
        self._inbound_tasks: set[asyncio.Task[None]] = set()
        self._early_notifications: list[tuple[str, dict[str, Any]]] = []
        self._closing = False

    @property
    def connected(self) -> bool:
        return (
            not self._closing
            and self.process is not None
            and self.process.returncode is None
            and self._reader_task is not None
            and not self._reader_task.done()
        )

    async def start(self) -> dict[str, Any]:
        if self.process is not None:
            raise RuntimeError("ACP agent connection is already started")
        self._closing = False
        try:
            self.process = await asyncio.wait_for(
                asyncio.create_subprocess_exec(
                    self.command,
                    *self.args,
                    stdin=asyncio.subprocess.PIPE,
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE,
                    cwd=self.cwd,
                    env=self.env,
                ),
                self.startup_timeout,
            )
        except (OSError, TimeoutError) as exc:
            raise ConnectionError(f"cannot start ACP agent {self.command!r}") from exc
        self._reader_task = asyncio.create_task(self._read_frames(), name="acp-agent-stdout")
        self._stderr_task = asyncio.create_task(self._drain_stderr(), name="acp-agent-stderr")
        try:
            result = await self.request(
                "initialize",
                {
                    "protocolVersion": PROTOCOL_VERSION,
                    "clientInfo": {
                        "name": "agents-anywhere-connector",
                        "title": "Agents Anywhere Connector",
                        "version": self.client_version,
                    },
                    "clientCapabilities": {
                        "fs": {"readTextFile": False, "writeTextFile": False},
                        "terminal": False,
                    },
                },
                timeout=self.startup_timeout,
            )
        except BaseException:
            await self.close()
            raise
        if not isinstance(result, dict):
            await self.close()
            raise RuntimeError("ACP initialize result must be an object")  # noqa: TRY004 - protocol violation, not a type error
        protocol_version = result.get("protocolVersion")
        if isinstance(protocol_version, bool) or not isinstance(protocol_version, int):
            await self.close()
            raise RuntimeError("ACP agent returned an invalid protocol version")  # noqa: TRY004
        self.initialize_result = result
        for method, params in self._early_notifications:
            self._dispatch_notification(method, params)
        self._early_notifications.clear()
        return result

    async def request(
        self,
        method: str,
        params: Mapping[str, Any] | None = None,
        *,
        timeout: float | None = None,
    ) -> Any:
        if self.process is None or self.process.returncode is not None:
            raise RuntimeError("ACP agent connection is not running")
        if self._closing:
            raise RuntimeError("ACP agent connection is closing")
        self._request_id += 1
        request_id = f"aa-{self._request_id}"
        loop = asyncio.get_running_loop()
        future: asyncio.Future[Any] = loop.create_future()
        self._pending[request_id] = future
        try:
            await self._send(
                {"jsonrpc": "2.0", "id": request_id, "method": method, "params": dict(params or {})}
            )
            return await asyncio.wait_for(future, timeout or self.request_timeout)
        except asyncio.CancelledError:
            with suppress(Exception):
                await self.notify("$/cancelRequest", {"requestId": request_id})
            raise
        except TimeoutError as exc:
            with suppress(Exception):
                await self.notify("$/cancelRequest", {"requestId": request_id})
            raise TimeoutError(f"ACP agent request timed out: {method}") from exc
        finally:
            self._pending.pop(request_id, None)

    async def notify(self, method: str, params: Mapping[str, Any] | None = None) -> None:
        await self._send({"jsonrpc": "2.0", "method": method, "params": dict(params or {})})

    async def respond(self, request_id: str | int, result: Any) -> None:
        await self._send({"jsonrpc": "2.0", "id": request_id, "result": result})

    async def close(self) -> None:
        process = self.process
        if process is None:
            return
        self._closing = True
        for future in tuple(self._pending.values()):
            if not future.done():
                future.set_exception(RuntimeError("ACP agent connection stopped"))
        self._pending.clear()
        reader_task, self._reader_task = self._reader_task, None
        stderr_task, self._stderr_task = self._stderr_task, None
        if process.returncode is None and process.stdin is not None:
            with suppress(OSError):
                process.stdin.close()
        try:
            await asyncio.wait_for(process.wait(), 5)
        except TimeoutError:
            with suppress(ProcessLookupError):
                process.kill()
            await process.wait()
        if reader_task is not None and reader_task is not asyncio.current_task():
            reader_task.cancel()
            await asyncio.gather(reader_task, return_exceptions=True)
        if stderr_task is not None and stderr_task is not asyncio.current_task():
            stderr_task.cancel()
            await asyncio.gather(stderr_task, return_exceptions=True)
        if self._inbound_tasks:
            await asyncio.gather(*self._inbound_tasks, return_exceptions=True)
        self.process = None

    async def _send(self, payload: Mapping[str, Any]) -> None:
        process = self.process
        if process is None or process.returncode is not None or process.stdin is None:
            raise RuntimeError("ACP agent connection is not running")
        encoded = (
            json.dumps(payload, ensure_ascii=False, separators=(",", ":"), allow_nan=False).encode("utf-8")
            + b"\n"
        )
        if len(encoded) > MAX_FRAME_BYTES:
            raise ValueError("ACP request frame exceeds 8 MiB")
        async with self._write_lock:
            process.stdin.write(encoded)
            await process.stdin.drain()

    async def _read_frames(self) -> None:
        process = self.process
        assert process is not None and process.stdout is not None
        try:
            while True:
                line = await process.stdout.readline()
                if not line:
                    return
                if len(line) > MAX_FRAME_BYTES:
                    raise RuntimeError("ACP response frame exceeds 8 MiB")
                self._handle_frame(line)
        except (ValueError, UnicodeError, json.JSONDecodeError, RuntimeError) as exc:
            logger.error("ACP protocol failure error_type={}", exc.__class__.__name__)
        finally:
            if process.returncode is None:
                with suppress(ProcessLookupError):
                    process.kill()
            for future in tuple(self._pending.values()):
                if not future.done():
                    future.set_exception(RuntimeError("ACP agent connection closed"))
            if not self._closing:
                await self.exit_handler(process.returncode)

    async def _drain_stderr(self) -> None:
        process = self.process
        assert process is not None and process.stderr is not None
        try:
            while True:
                line = await process.stderr.readline()
                if not line:
                    return
                text = line.decode("utf-8", errors="replace").rstrip("\n")
                if text:
                    logger.debug("ACP agent stderr: {}", text)
        except (ValueError, UnicodeError):
            return

    def _handle_frame(self, line: bytes) -> None:
        value = json.loads(line.decode("utf-8"))
        if not isinstance(value, dict) or value.get("jsonrpc") != "2.0":
            raise RuntimeError("ACP agent emitted an invalid JSON-RPC frame")
        if "id" in value and ("result" in value or "error" in value) and "method" not in value:
            self._handle_response(value)
            return
        method = value.get("method")
        params = value.get("params", {})
        if not isinstance(method, str) or not isinstance(params, dict):
            raise RuntimeError("ACP agent emitted an invalid request or notification")  # noqa: TRY004
        if "id" in value:
            self._handle_request(value["id"], method, params)
            return
        if self.initialize_result is None:
            if len(self._early_notifications) >= 64:
                raise RuntimeError("ACP agent emitted too many notifications during initialize")
            self._early_notifications.append((method, params))
            return
        self._dispatch_notification(method, params)

    def _handle_response(self, value: dict[str, Any]) -> None:
        request_id = value.get("id")
        future = self._pending.get(request_id)
        if future is None or future.done():
            return
        error = value.get("error")
        if error is not None:
            if not isinstance(error, dict) or not isinstance(error.get("code"), int):
                future.set_exception(RuntimeError("ACP agent returned an invalid error response"))
                return
            future.set_exception(
                AcpRpcError(
                    error["code"],
                    str(error.get("message") or "ACP agent request failed"),
                    error.get("data") if isinstance(error.get("data"), dict) else None,
                )
            )
            return
        future.set_result(value.get("result"))

    def _handle_request(self, request_id: str | int, method: str, params: dict[str, Any]) -> None:
        task = asyncio.create_task(self._answer_request(request_id, method, params))
        self._inbound_tasks.add(task)
        task.add_done_callback(self._inbound_tasks.discard)

    async def _answer_request(self, request_id: str | int, method: str, params: dict[str, Any]) -> None:
        if self.request_handler is None:
            await self.respond(
                request_id,
                None,
            )
            return
        try:
            result = await self.request_handler(method, params)
        except AcpNoResponse:
            return
        except Exception as exc:  # noqa: BLE001 - isolate a failed inbound handler from the read loop
            with suppress(Exception):
                await self._send(
                    {
                        "jsonrpc": "2.0",
                        "id": request_id,
                        "error": {"code": -32603, "message": str(exc) or "Connector handler failed"},
                    }
                )
            return
        with suppress(Exception):
            await self.respond(request_id, result)

    def _dispatch_notification(self, method: str, params: dict[str, Any]) -> None:
        task = asyncio.create_task(self.notification_handler(method, params))
        self._inbound_tasks.add(task)
        task.add_done_callback(self._notification_done)

    def _notification_done(self, task: asyncio.Task[None]) -> None:
        self._inbound_tasks.discard(task)
        if task.cancelled():
            return
        error = task.exception()
        if error is not None:
            logger.error("ACP notification handler failed error_type={}", error.__class__.__name__)
