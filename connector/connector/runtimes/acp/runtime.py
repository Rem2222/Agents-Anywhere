from __future__ import annotations

import asyncio
import uuid
from collections.abc import Mapping
from contextlib import suppress
from typing import Any

from connector.logging import logger
from connector.runtime_protocol import (
    AgentRuntime,
    RuntimeAttachment,
    RuntimeCapability,
    RuntimeCapabilitySet,
    RuntimeCommand,
    RuntimeCommandResult,
    RuntimeConfig,
    RuntimeIdentity,
    RuntimeInvalidRequestError,
    RuntimeModelCatalog,
    RuntimeOperationResult,
    RuntimePermissionCatalog,
    RuntimeTimelineSnapshot,
    RuntimeUnavailableError,
    RuntimeUnsupportedError,
    RuntimeUpstreamError,
    SessionMeta,
    SessionNotice,
    SessionState,
)
from connector.runtime_protocol.host import RuntimeHostClient
from connector.runtimes.acp import catalogs, discovery, provider_config
from connector.runtimes.acp.attachments import image_blocks, text_block
from connector.runtimes.acp.client import AcpClient, AcpNoResponse, AcpRpcError
from connector.runtimes.acp.interactions import PermissionBridge
from connector.runtimes.acp.models import (
    AcpSessionState,
    HostCall,
    NoticeCall,
    SessionMetaCall,
    SessionStateCall,
    TimelineItemCall,
    TimelineSyncCall,
    TurnEndedCall,
    close_open_items,
    reduce_session_update,
    turn_end_call,
    turn_start_call,
)
from connector.runtimes.session_identity import stable_runtime_session_id

POLL_INTERVAL_SECONDS = 5.0
CATALOG_CACHE_KEY = "acp/catalog"


class _LoadCapture:
    """Buffers history updates streamed before a ``session/load`` response."""

    def __init__(self) -> None:
        self.calls: list[HostCall] = []


class AcpRuntime(AgentRuntime):
    """One ACP agent child process, multiplexing every session it owns."""

    def __init__(
        self,
        config: RuntimeConfig,
        host: RuntimeHostClient,
        client_version: str = "1.0",
    ) -> None:
        self.config = config
        self.host = host
        self.client_version = client_version
        self._identity = RuntimeIdentity("acp", "unknown", "ACP agent")
        self._client: AcpClient | None = None
        self._connect_lock = asyncio.Lock()
        self._stopping = False
        self._restart_task: asyncio.Task[None] | None = None
        self._idle_task: asyncio.Task[None] | None = None
        self._sessions: dict[str, AcpSessionState] = {}
        self._loads: dict[str, _LoadCapture] = {}
        self._active_turns: set[str] = set()
        self._session_status: dict[str, str] = {}
        self._session_selections: dict[str, dict[str, str | None]] = {}
        self._session_metadata: dict[str, dict[str, Any]] = {}
        self._catalog_revision = 0
        self._model_catalog: RuntimeModelCatalog | None = None
        self._permission_catalog: RuntimePermissionCatalog | None = None
        self._model_selection_ids: dict[str, str] = {}
        self._mode_selection_ids: dict[str, str] = {}
        self._agent_info: discovery.AcpAgentInfo | None = None
        self._bridge = PermissionBridge(
            permission_timeout=0.0,
            notice_callback=self._notice_callback,
        )

    # -- identity / lifecycle -------------------------------------------------

    @property
    def sync_mode(self) -> str:
        return "events"

    @property
    def identity(self) -> RuntimeIdentity:
        return self._identity

    async def start(self) -> None:
        self._stopping = False
        values = provider_config.normalized_config_values(dict(self.config.values))
        self._bridge.permission_timeout = int(values["permissionTimeoutMs"]) / 1000
        await self.host.runtime_health_update("starting", {
            "code": "runtime_initializing", "message": "Starting ACP agent…", "retryable": True,
        })
        try:
            await self._ensure_client()
            if values["idleTimeoutSeconds"]:
                self._idle_task = asyncio.create_task(self._idle_loop())
        except (OSError, RuntimeError, ValueError, TimeoutError):
            with suppress(Exception):
                await self.host.runtime_health_update(
                    "starting",
                    {"code": "runtime_unavailable", "message": "Waiting for the ACP agent to start…", "retryable": True},
                )
            self._schedule_restart()

    async def stop(self) -> None:
        self._stopping = True
        if getattr(self, "_idle_task", None) is not None:
            self._idle_task.cancel()
            await asyncio.gather(self._idle_task, return_exceptions=True)
            self._idle_task = None
        if self._restart_task is not None:
            self._restart_task.cancel()
            await asyncio.gather(self._restart_task, return_exceptions=True)
            self._restart_task = None
        await self._bridge.cancel_all()
        async with self._connect_lock:
            client, self._client = self._client, None
            if client is not None:
                await client.close()

    async def get_config(self) -> RuntimeConfig:
        return self.config

    # -- capabilities / catalogs ---------------------------------------------

    async def get_runtime_capabilities(self) -> RuntimeCapabilitySet:
        return self._runtime_capability_set()

    def _runtime_capability_set(self, session_id: str | None = None) -> RuntimeCapabilitySet:
        capabilities: list[RuntimeCapability] = []
        image = self._agent_info.image_attachments if self._agent_info is not None else True
        permission = self._permission_catalog is not None and bool(self._permission_catalog.permissions)
        entries = (
            ("session.send_message", True),
            ("session.interrupt", True),
            ("session.steer", False),
            ("session.commands", True),
            ("session.interaction.approval", True),
            ("catalog.model", True),
            ("catalog.permission", permission),
            ("runtime.attachment", image),
        )
        for capability_id, supported in entries:
            capabilities.append(
                RuntimeCapability(
                    capability_id=capability_id,
                    scope="session" if session_id else "runtime",
                    runtime="acp",
                    session_id=session_id,
                    connector_id=self.host.connector_id,
                    supported=supported,
                    available=supported,
                    allowed=True,
                )
            )
        return RuntimeCapabilitySet(
            runtime="acp",
            revision=1,
            capabilities=tuple(capabilities),
            session_id=session_id,
            connector_id=self.host.connector_id,
        )

    async def list_model_catalog(self, query: str | None = None, limit: int = 100) -> RuntimeModelCatalog:
        await self._restore_catalog_cache()
        if self._model_catalog is not None:
            return self._model_catalog
        custom = self._custom_model_items()
        return RuntimeModelCatalog(runtime="acp", revision=max(1, self._catalog_revision), models=custom)

    async def list_permission_catalog(self, query: str | None = None, limit: int = 100) -> RuntimePermissionCatalog:
        await self._restore_catalog_cache()
        if self._permission_catalog is not None:
            return self._permission_catalog
        raise RuntimeUnsupportedError("list_permission_catalog")

    def _custom_model_items(self):
        from connector.runtimes.custom_models import custom_model_items

        return custom_model_items("acp", self.config.values.get("customModels"))

    # -- sessions -------------------------------------------------------------

    def _platform_session_id(self, external_session_id: str) -> str:
        return stable_runtime_session_id(self.host.connector_id, "acp", external_session_id)

    def _session_state(self, external_session_id: str) -> AcpSessionState:
        state = self._sessions.get(external_session_id)
        if state is None:
            state = AcpSessionState(self._platform_session_id(external_session_id), external_session_id)
            self._sessions[external_session_id] = state
        return state

    async def list_sessions(
        self,
        limit: int = 100,
        cursor: str | None = None,
        force: bool = False,
    ) -> tuple[SessionMeta, ...]:
        result = await self._request("session/list", {"cursor": cursor} if cursor else None)
        return self._session_metas(result)

    async def list_complete_session_inventory(
        self,
        page_size: int = 100,
        force: bool = False,
    ) -> tuple[SessionMeta, ...]:
        output: list[SessionMeta] = []
        seen: set[str] = set()
        cursor: str | None = None
        while True:
            result = await self._request("session/list", {"cursor": cursor} if cursor else None)
            metas = self._session_metas(result)
            for meta in metas:
                if meta.session_id in seen:
                    raise RuntimeUpstreamError("ACP inventory repeated a session")
                seen.add(meta.session_id)
                output.append(meta)
            next_cursor = result.get("nextCursor")
            if not next_cursor:
                return tuple(output)
            cursor = next_cursor

    def _session_metas(self, result: dict[str, Any]) -> tuple[SessionMeta, ...]:
        sessions = result.get("sessions") if isinstance(result.get("sessions"), list) else []
        metas: list[SessionMeta] = []
        for raw in sessions:
            if not isinstance(raw, dict):
                continue
            ext_id = raw.get("sessionId")
            if not isinstance(ext_id, str) or not ext_id:
                continue
            platform_id = self._platform_session_id(ext_id)
            self._session_state(ext_id)  # keep the platform->native mapping for interrupt
            metas.append(
                SessionMeta(
                    session_id=platform_id,
                    external_session_id=ext_id,
                    runtime="acp",
                    title=raw.get("title") if isinstance(raw.get("title"), str) else None,
                    cwd=raw.get("cwd") if isinstance(raw.get("cwd"), str) else None,
                    ordering_time=raw.get("updatedAt") if isinstance(raw.get("updatedAt"), str) else None,
                )
            )
        return tuple(metas)

    async def get_session_snapshot(
        self,
        session_id: str,
        external_session_id: str | None = None,
        limit: int | None = None,
    ) -> RuntimeTimelineSnapshot:
        ext_id = external_session_id or self._resolve_external_id(session_id)
        if ext_id is None:
            return RuntimeTimelineSnapshot(session_id=session_id, external_session_id=None, runtime="acp", items=(), complete=True)
        await self._ensure_client()
        state = self._session_state(ext_id)
        capture = _LoadCapture()
        self._loads[ext_id] = capture
        try:
            response = await self._request(
                "session/load",
                {"sessionId": ext_id, "cwd": self._session_cwd(), "mcpServers": []},
            )
        finally:
            self._loads.pop(ext_id, None)
        await self._seed_catalogs(response)
        items = tuple(call.item for call in capture.calls if isinstance(call, TimelineItemCall))
        return RuntimeTimelineSnapshot(
            session_id=state.session_id,
            external_session_id=ext_id,
            runtime="acp",
            items=items,
            complete=True,
        )

    def _resolve_external_id(self, session_id: str) -> str | None:
        for ext_id, state in self._sessions.items():
            if state.session_id == session_id:
                return ext_id
        return None

    async def get_session_state(
        self,
        session_id: str,
        external_session_id: str | None = None,
    ) -> SessionState | None:
        ext_id = external_session_id or self._resolve_external_id(session_id)
        if ext_id is None:
            return None
        platform_id = self._platform_session_id(ext_id)
        status = self._session_status.get(ext_id, "idle")
        return SessionState(
            session_id=platform_id,
            external_session_id=ext_id,
            runtime="acp",
            status=status,  # type: ignore[arg-type]
            selections=dict(self._session_selections.get(ext_id, {})),
            metadata=dict(self._session_metadata.get(ext_id, {})),
        )

    async def get_session_notices(
        self,
        session_id: str,
        external_session_id: str | None = None,
    ) -> tuple[SessionNotice, ...]:
        return self._bridge.notices(session_id)

    async def get_session_capabilities(
        self,
        session_id: str,
        external_session_id: str | None = None,
    ) -> RuntimeCapabilitySet:
        return self._runtime_capability_set(session_id)

    # -- turns -----------------------------------------------------------------

    async def create_and_start_session(
        self,
        session_id: str,
        content: str,
        title: str | None = None,
        cwd: str | None = None,
        selections: Mapping[str, str | None] | None = None,
        attachments: tuple[RuntimeAttachment, ...] = (),
        client_message_id: str | None = None,
        runtime_options: Mapping[str, Any] | None = None,
    ) -> RuntimeOperationResult:
        if runtime_options:
            raise RuntimeInvalidRequestError("Unsupported ACP creation option")
        cwd = self._normalize_cwd(cwd)
        response = await self._request("session/new", {"cwd": cwd, "mcpServers": []})
        ext_id = response.get("sessionId")
        if not isinstance(ext_id, str) or not ext_id:
            raise RuntimeUpstreamError("ACP session/new returned no session id")
        await self.host.session_meta_upsert(
            session_id=session_id,
            runtime="acp",
            external_session_id=ext_id,
            title=title,
            cwd=cwd,
        )
        await self.host.session_state_update(session_id=session_id, runtime="acp", status="idle", external_session_id=ext_id)
        await self._seed_catalogs(response)
        return await self.start_turn(
            session_id,
            ext_id,
            content,
            selections=selections,
            attachments=attachments,
            client_message_id=client_message_id,
            cwd=cwd,
        )

    async def start_turn(
        self,
        session_id: str,
        external_session_id: str | None,
        content: str,
        selections: Mapping[str, str | None] | None = None,
        attachments: tuple[RuntimeAttachment, ...] = (),
        client_message_id: str | None = None,
        cwd: str | None = None,
    ) -> RuntimeOperationResult:
        if external_session_id is None:
            raise RuntimeInvalidRequestError("ACP start_turn requires an external session id")
        if not content.strip() and not attachments:
            raise RuntimeInvalidRequestError("A message and a stable clientMessageId are required")
        if not client_message_id:
            raise RuntimeInvalidRequestError("A stable clientMessageId is required")
        if await self._already_sent(external_session_id, client_message_id):
            return RuntimeOperationResult(ok=True, code="already_sent", message="Message was already delivered")
        await self._ensure_client()
        # Mark as sent only once transport is up; a lost response must not cause a
        # re-send, but a failed connect must leave the message retryable.
        await self._mark_sent(external_session_id, client_message_id)

        blocks = [text_block(content)]
        if attachments:
            blocks.extend(await image_blocks(self.host, session_id, attachments))

        state = self._session_state(external_session_id)
        turn_id = state.next_turn_id()
        self._active_turns.add(external_session_id)
        self._session_status[external_session_id] = "running"
        await self.host.session_state_update(session_id, "acp", status="running", external_session_id=external_session_id)
        await self._apply(state.session_id, external_session_id, [turn_start_call(state, turn_id)])
        try:
            response = await self._request(
                "session/prompt",
                {"sessionId": external_session_id, "prompt": blocks, "messageId": str(uuid.uuid4())},
                timeout=self._request_timeout(),
            )
        finally:
            self._active_turns.discard(external_session_id)
            self._session_status[external_session_id] = "idle"
        calls = close_open_items(state)
        calls.append(turn_end_call(state, turn_id))
        calls.append(TurnEndedCall(turn_id=turn_id, outcome="completed"))
        await self._apply(state.session_id, external_session_id, calls)
        await self.host.session_state_update(session_id, "acp", status="idle", external_session_id=external_session_id)
        return RuntimeOperationResult(result=dict(response) if isinstance(response, dict) else {})

    async def interrupt_session(
        self,
        session_id: str,
        reason: str | None = None,
    ) -> RuntimeOperationResult:
        ext_id = self._resolve_external_id(session_id)
        if ext_id is None:
            return RuntimeOperationResult(ok=False, code="session_not_found", message="Unknown ACP session")
        result = await self._request("session/cancel", {"sessionId": ext_id})
        return RuntimeOperationResult(result=dict(result) if isinstance(result, dict) else {})

    async def update_session_selections(
        self,
        session_id: str,
        external_session_id: str | None,
        selections: Mapping[str, str | None],
    ) -> RuntimeOperationResult:
        ext_id = external_session_id or self._resolve_external_id(session_id)
        if ext_id is None:
            return RuntimeOperationResult(ok=False, code="session_not_found", message="Unknown ACP session")
        if ext_id in self._active_turns and "model" in selections:
            return RuntimeOperationResult(ok=False, code="busy", message="Model can only be changed while the session is idle")
        for scope, method, registry in (
            ("model", "session/set_model", self._model_selection_ids),
            ("permission", "session/set_mode", self._mode_selection_ids),
        ):
            selection_id = selections.get(scope)
            if selection_id is None:
                continue
            native_id = registry.get(selection_id)
            if native_id is None:
                return RuntimeOperationResult(ok=False, code="invalid_selection", message=f"Unknown {scope} selection")
            key = "modelId" if scope == "model" else "modeId"
            await self._request(method, {"sessionId": ext_id, key: native_id})
            current = dict(self._session_selections.get(ext_id, {}))
            current[scope] = selection_id
            self._session_selections[ext_id] = current
            await self.host.session_state_update(
                session_id, "acp", selections={scope: selection_id}, external_session_id=ext_id
            )
        return RuntimeOperationResult()

    async def list_commands(
        self,
        session_id: str,
        external_session_id: str | None = None,
        query: str | None = None,
        limit: int = 50,
    ) -> tuple[RuntimeCommand, ...]:
        ext_id = external_session_id or self._resolve_external_id(session_id)
        state = self._sessions.get(ext_id) if ext_id else None
        commands = state.commands if state is not None else []
        return tuple(
            RuntimeCommand(
                id=command.get("name") or "",
                title=command.get("name") or "",
                description=command.get("description") if isinstance(command.get("description"), str) else None,
                accepts_args=isinstance(command.get("input"), dict),
                metadata={"source": "acp.available_commands_update"},
            )
            for command in commands
            if isinstance(command.get("name"), str) and command["name"]
        )

    async def execute_command(
        self,
        session_id: str,
        command: str,
        external_session_id: str | None = None,
        raw: str | None = None,
        args: tuple[str, ...] = (),
    ) -> RuntimeCommandResult:
        # ACP has no command RPC; a slash command is an ordinary prompt the agent parses itself.
        text = raw if raw is not None else (f"/{command}" + (" " + " ".join(args) if args else ""))
        result = await self.start_turn(session_id, external_session_id, text, client_message_id=str(uuid.uuid4()))
        return RuntimeCommandResult(command=command, ok=result.ok, code=result.code, message=result.message, result=result.result)

    async def respond_interaction(
        self,
        session_id: str,
        notice_id: str,
        action_id: str,
        input_data: Mapping[str, Any] | None = None,
    ) -> RuntimeOperationResult:
        outcome = await self._bridge.respond(notice_id, action_id)
        if outcome is None:
            return RuntimeOperationResult(ok=False, code="not_pending", message="No pending approval for this notice")
        return RuntimeOperationResult(result={"outcome": outcome})

    # -- transport -------------------------------------------------------------

    async def _ensure_client(self) -> None:
        if self._client is not None and self._client.connected:
            return
        async with self._connect_lock:
            if self._stopping:
                raise RuntimeUnavailableError("ACP agent is stopping")
            if self._client is not None and not self._client.connected:
                client, self._client = self._client, None
                await client.close()
            if self._client is None:
                await self._start_client()

    async def _start_client(self) -> None:
        values = provider_config.normalized_config_values(dict(self.config.values))
        command, args = provider_config.command_and_args(values)
        env = dict(values.get("environment") or {})
        preset_id = values.get("agentPreset")
        from connector.runtimes.acp import presets

        preset = presets.preset_by_id(preset_id) if isinstance(preset_id, str) else None
        if preset and preset.home_env:
            env.setdefault(preset.home_env, str(values.get("agentHome") or ""))
        client = AcpClient(
            command=command,
            args=args,
            env=env,
            cwd=values.get("cwd") or None,
            client_version=self.client_version,
            startup_timeout=int(values["startupTimeoutMs"]) / 1000,
            request_timeout=int(values["requestTimeoutMs"]) / 1000,
            notification_handler=self._handle_notification,
            request_handler=self._handle_request,
            exit_handler=self._handle_exit,
        )
        try:
            result = await client.start()
        except BaseException:
            await client.close()
            raise
        self._agent_info = discovery._agent_info(result)
        self._identity = RuntimeIdentity(
            runtime="acp",
            runtime_version=self._agent_info.version,
            display_name=self._agent_info.name or "ACP agent",
            protocol_version=str(self._agent_info.protocol_version),
        )
        self._client = client
        await self.host.runtime_capabilities_update(self._runtime_capability_set())
        await self._seed_catalogs(result)
        await self.host.runtime_health_update("running")

    async def _request(
        self,
        method: str,
        params: Mapping[str, Any] | None = None,
        *,
        timeout: float | None = None,
    ) -> dict[str, Any]:
        try:
            await self._ensure_client()
            if self._client is None:
                raise RuntimeUnavailableError("ACP agent is not running")
            result = await self._client.request(method, params, timeout=timeout)
        except AcpRpcError as exc:
            if exc.rpc_code == -32601:
                raise RuntimeUnsupportedError(method) from exc
            if exc.rpc_code in {-32600, -32602}:
                raise RuntimeInvalidRequestError(str(exc)) from exc
            if exc.retryable:
                raise RuntimeUnavailableError(str(exc)) from exc
            raise RuntimeUpstreamError(str(exc)) from exc
        except (OSError, TimeoutError, ConnectionError, RuntimeError) as exc:
            logger.warning("ACP agent request unavailable method={} error_type={}", method, type(exc).__name__)
            raise RuntimeUnavailableError("ACP agent is unavailable") from exc
        if not isinstance(result, dict):
            raise RuntimeUpstreamError("ACP response must be an object")
        return result

    async def _handle_notification(self, method: str, params: Mapping[str, Any]) -> None:
        if method != "session/update":
            return
        session_update = params.get("update") if isinstance(params, Mapping) else None
        if not isinstance(session_update, Mapping):
            return
        ext_id = params.get("sessionId")
        if not isinstance(ext_id, str) or not ext_id:
            return
        state = self._session_state(ext_id)
        capture = self._loads.get(ext_id)
        origin = "history" if capture is not None else "live"
        calls = reduce_session_update(state, session_update, origin=origin)
        if capture is not None:
            capture.calls.extend(calls)
            return
        await self._apply(state.session_id, ext_id, calls)

    async def _handle_request(self, method: str, params: Mapping[str, Any]) -> Any:
        if method != "session/request_permission":
            raise AcpNoResponse()
        ext_id = params.get("sessionId")
        tool_call = params.get("toolCall")
        options = params.get("options")
        if not isinstance(ext_id, str) or not isinstance(tool_call, dict):
            raise AcpNoResponse()
        tool_call_id = tool_call.get("toolCallId")
        if not isinstance(tool_call_id, str) or not tool_call_id:
            raise AcpNoResponse()
        state = self._session_state(ext_id)
        # Project the pending tool call as waiting_approval, then wait for a user.
        pending_update = {**tool_call, "sessionUpdate": "tool_call_update", "status": "pending"}
        calls = reduce_session_update(state, pending_update, origin="live")
        await self._apply(state.session_id, ext_id, calls)
        option_tuple = tuple(option for option in options if isinstance(option, dict)) if isinstance(options, list) else ()
        title = tool_call.get("title") if isinstance(tool_call.get("title"), str) and tool_call["title"] else "Approve tool call"
        return await self._bridge.handle_request(
            self._next_request_id(),
            state.session_id,
            tool_call_id,
            option_tuple,
            title,
        )

    async def _notice_callback(self, notice: SessionNotice) -> None:
        await self.host.notice_upsert(notice)

    async def _handle_exit(self, return_code: int | None) -> None:
        self._client = None
        await self._bridge.cancel_all()
        for ext_id in list(self._active_turns):
            self._active_turns.discard(ext_id)
            state = self._sessions.get(ext_id)
            if state is not None:
                await self._apply(state.session_id, ext_id, close_open_items(state))
        for ext_id in list(self._session_status):
            self._session_status[ext_id] = "disconnected"
        if self._stopping:
            return
        with suppress(Exception):
            await self.host.runtime_error("acp", "ACP_AGENT_EXITED", "ACP agent process exited", details={"retryable": True})
        with suppress(Exception):
            await self.host.runtime_health_update(
                "error",
                {"code": "runtime_unavailable", "message": "ACP agent exited; restarting…", "retryable": True},
            )
        self._schedule_restart()

    def _schedule_restart(self) -> None:
        if not self._stopping and (self._restart_task is None or self._restart_task.done()):
            self._restart_task = asyncio.create_task(self._restart_loop())

    async def _restart_loop(self) -> None:
        values = provider_config.normalized_config_values(dict(self.config.values))
        fast_attempts = int(values["maxRestartAttempts"])
        attempt = 0
        while not self._stopping:
            if self._client is not None and self._client.connected:
                return
            delay = POLL_INTERVAL_SECONDS
            if attempt < fast_attempts:
                delay = min(int(values["restartBackoffMs"]) / 1000 * 2**attempt, delay)
            await asyncio.sleep(delay)
            if self._stopping:
                return
            try:
                await self._ensure_client()
                return
            except Exception:  # noqa: BLE001 - recoverable reconnect failure, keep polling
                attempt = min(attempt + 1, fast_attempts)
                continue

    async def _idle_loop(self) -> None:
        values = provider_config.normalized_config_values(dict(self.config.values))
        idle_seconds = int(values["idleTimeoutSeconds"])
        while not self._stopping:
            await asyncio.sleep(max(30.0, idle_seconds / 2))
            if self._stopping:
                return
            if self._active_turns or not self._sessions:
                continue
            if self._client is not None and self._client.connected:
                logger.debug("Recycling idle ACP agent process")
                client, self._client = self._client, None
                await client.close()

    # -- reduction -------------------------------------------------------------

    async def _apply(self, platform_session_id: str, external_session_id: str, calls: list[HostCall]) -> None:
        for call in calls:
            if isinstance(call, TimelineItemCall):
                await self.host.timeline_item_upsert(call.item)
            elif isinstance(call, TimelineSyncCall):
                await self.host.timeline_sync(platform_session_id, "acp", call.items, external_session_id, complete=call.complete)
            elif isinstance(call, SessionStateCall):
                if call.selections is not None:
                    current = dict(self._session_selections.get(external_session_id, {}))
                    current.update({key: value for key, value in call.selections.items() if value is not None})
                    self._session_selections[external_session_id] = current
                if call.metadata is not None:
                    meta = dict(self._session_metadata.get(external_session_id, {}))
                    meta.update(call.metadata)
                    self._session_metadata[external_session_id] = meta
                await self.host.session_state_update(
                    platform_session_id, "acp",
                    status=call.status,
                    selections=call.selections,
                    external_session_id=external_session_id,
                    metadata=call.metadata,
                )
            elif isinstance(call, SessionMetaCall):
                await self.host.session_meta_upsert(
                    platform_session_id, "acp",
                    external_session_id=external_session_id,
                    title=call.title,
                    ordering_time=call.ordering_time,
                )
            elif isinstance(call, TurnEndedCall):
                await self.host.session_turn_ended(
                    platform_session_id, "acp",
                    external_session_id=external_session_id,
                    turn_id=call.turn_id,
                    outcome=call.outcome,
                )
            elif isinstance(call, NoticeCall):
                await self.host.notice_upsert(call.notice)

    # -- catalogs / dedup -------------------------------------------------------

    async def _seed_catalogs(self, response: Mapping[str, Any]) -> None:
        models = response.get("models")
        available_models = models.get("availableModels") if isinstance(models, dict) else None
        current_model = models.get("currentModelId") if isinstance(models, dict) else None
        modes = response.get("modes")
        available_modes = modes.get("availableModes") if isinstance(modes, dict) else None
        current_mode = modes.get("currentModeId") if isinstance(modes, dict) else None

        if isinstance(available_models, list) and available_models:
            catalog = catalogs.model_catalog_from_state(
                available_models,
                current_model if isinstance(current_model, str) else None,
                max(1, self._catalog_revision),
                custom_models=self._custom_model_items(),
            )
            self._model_selection_ids = {
                item.selection_id: str(item.metadata.get("acpModelId") or item.id)
                for item in catalog.models
                if item.selection_id
            }
            if self._model_catalog is None or catalog.models != self._model_catalog.models:
                self._catalog_revision += 1
                catalog = RuntimeModelCatalog(runtime="acp", revision=self._catalog_revision, models=catalog.models)
                self._model_catalog = catalog
                await self.host.model_catalog_update(catalog)
                await self._save_catalog_cache()
        elif self._model_catalog is None:
            custom = self._custom_model_items()
            if custom:
                self._catalog_revision = max(1, self._catalog_revision)
                catalog = RuntimeModelCatalog(runtime="acp", revision=self._catalog_revision, models=custom)
                self._model_catalog = catalog
                self._model_selection_ids = {
                    item.selection_id: str(item.metadata.get("acpModelId") or item.id)
                    for item in catalog.models
                    if item.selection_id
                }
                await self.host.model_catalog_update(catalog)

        if isinstance(available_modes, list) and available_modes:
            catalog = catalogs.permission_catalog_from_state(
                available_modes,
                current_mode if isinstance(current_mode, str) else None,
                max(1, self._catalog_revision),
            )
            if catalog is not None:
                self._mode_selection_ids = {
                    item.selection_id: str(item.metadata.get("acpModeId") or item.id)
                    for item in catalog.permissions
                    if item.selection_id
                }
                if self._permission_catalog is None or catalog.permissions != self._permission_catalog.permissions:
                    self._catalog_revision += 1
                    catalog = RuntimePermissionCatalog(runtime="acp", revision=self._catalog_revision, permissions=catalog.permissions)
                    self._permission_catalog = catalog
                    await self.host.permission_catalog_update(catalog)

    async def _restore_catalog_cache(self) -> None:
        if self._model_catalog is not None:
            return
        cached = await self.host.sync_state_read(CATALOG_CACHE_KEY)
        if not isinstance(cached, dict) or not isinstance(cached.get("revision"), int):
            return
        self._catalog_revision = cached["revision"]

    async def _save_catalog_cache(self) -> None:
        await self.host.sync_state_write(CATALOG_CACHE_KEY, {"revision": self._catalog_revision})

    async def _already_sent(self, external_session_id: str, client_message_id: str) -> bool:
        cached = await self.host.sync_state_read(self._pending_key(external_session_id, client_message_id))
        return isinstance(cached, dict) and cached.get("status") == "sent"

    async def _mark_sent(self, external_session_id: str, client_message_id: str) -> None:
        await self.host.sync_state_write(self._pending_key(external_session_id, client_message_id), {"status": "sent"})

    @staticmethod
    def _pending_key(external_session_id: str, client_message_id: str) -> str:
        return f"acp/pending/{external_session_id}/{client_message_id}"

    # -- helpers ---------------------------------------------------------------

    def _next_request_id(self) -> str:
        return f"perm-{uuid.uuid4().hex[:12]}"

    def _request_timeout(self) -> float:
        values = provider_config.normalized_config_values(dict(self.config.values))
        return int(values["requestTimeoutMs"]) / 1000

    def _session_cwd(self) -> str:
        values = provider_config.normalized_config_values(dict(self.config.values))
        return str(values.get("cwd") or values.get("agentHome") or "")

    def _normalize_cwd(self, cwd: str | None) -> str:
        values = provider_config.normalized_config_values(dict(self.config.values))
        if cwd and str(cwd).strip():
            from pathlib import Path

            if Path(cwd).expanduser().is_absolute():
                return str(cwd)
        return str(values.get("cwd") or values.get("agentHome") or "")
