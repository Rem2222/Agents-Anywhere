from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import Any

from connector.runtime_protocol.models import SessionNotice
from connector.runtimes.acp.client import AcpNoResponse

NoticeCallback = Callable[[SessionNotice], Awaitable[None]]


@dataclass(slots=True)
class PendingPermission:
    request_id: str | int
    platform_session_id: str
    tool_call_id: str
    notice_id: str
    options: tuple[dict[str, Any], ...]
    notice: SessionNotice
    future: asyncio.Future[dict[str, Any]]


def permission_notice(
    platform_session_id: str,
    tool_call_id: str,
    options: tuple[dict[str, Any], ...],
    title: str,
) -> SessionNotice:
    """Build the blocking approval interaction for one ACP permission request."""
    actions: list[dict[str, Any]] = []
    for option in options:
        if not isinstance(option, dict) or not option.get("optionId"):
            continue
        kind = option.get("kind") if isinstance(option.get("kind"), str) else ""
        actions.append(
            {
                "actionId": option["optionId"],
                "label": option.get("name") if isinstance(option.get("name"), str) and option["name"] else option["optionId"],
                "style": "primary" if kind.startswith("allow") else "danger",
            }
        )
    return SessionNotice(
        notice_id=f"acp:perm:{tool_call_id}",
        session_id=platform_session_id,
        runtime="acp",
        type="interaction",
        title=title or "Approve tool call",
        severity="warning",
        status="open",
        interaction_type="tool_call_approval",
        blocking={"scope": "tool_call", "targetId": tool_call_id},
        response_required=True,
        actions=tuple(actions),
        source={"runtime": "acp", "approvalId": tool_call_id},
    )


def outcome_for(action_id: str, options: tuple[dict[str, Any], ...]) -> dict[str, Any] | None:
    """Map an action (option) id to the ACP ``RequestPermissionResponse.outcome``."""
    option = next((option for option in options if option.get("optionId") == action_id), None)
    if option is None:
        return None
    kind = option.get("kind") if isinstance(option.get("kind"), str) else ""
    if kind.startswith("allow"):
        return {"outcome": "selected", "optionId": action_id}
    return {"outcome": "cancelled"}


class PermissionBridge:
    """Holds pending ACP permission requests until a user answers or time runs out.

    The connector never answers the agent on its own timeout: Hermes self-denies
    after its own ``approvals.timeout`` and closes the tool call itself. We only
    expire the notice so the approval card does not hang forever.
    """

    def __init__(
        self,
        *,
        permission_timeout: float,
        notice_callback: NoticeCallback,
    ) -> None:
        self.permission_timeout = permission_timeout
        self.notice_callback = notice_callback
        self._pending: dict[str, PendingPermission] = {}
        self._by_request: dict[str | int, str] = {}

    async def handle_request(
        self,
        request_id: str | int,
        platform_session_id: str,
        tool_call_id: str,
        options: tuple[dict[str, Any], ...],
        title: str,
    ) -> dict[str, Any]:
        """Register a request, publish its notice, and await the user's answer."""
        notice = permission_notice(platform_session_id, tool_call_id, options, title)
        loop = asyncio.get_running_loop()
        future: asyncio.Future[dict[str, Any]] = loop.create_future()
        pending = PendingPermission(
            request_id=request_id,
            platform_session_id=platform_session_id,
            tool_call_id=tool_call_id,
            notice_id=notice.notice_id,
            options=options,
            notice=notice,
            future=future,
        )
        self._pending[notice.notice_id] = pending
        self._by_request[request_id] = notice.notice_id
        await self.notice_callback(notice)
        try:
            return await asyncio.wait_for(future, self.permission_timeout)
        except TimeoutError:
            await self.expire(notice.notice_id)
            raise AcpNoResponse() from None

    async def expire(self, notice_id: str) -> None:
        pending = self._pending.pop(notice_id, None)
        if pending is None:
            return
        self._by_request.pop(pending.request_id, None)
        if not pending.future.done():
            pending.future.set_result({"outcome": "cancelled"})
        expired = _with_status(permission_notice(
            pending.platform_session_id, pending.tool_call_id, pending.options, "Approve tool call"
        ), "expired")
        await self.notice_callback(expired)

    async def respond(self, notice_id: str, action_id: str) -> dict[str, Any] | None:
        pending = self._pending.pop(notice_id, None)
        if pending is None:
            return None
        self._by_request.pop(pending.request_id, None)
        outcome = outcome_for(action_id, pending.options)
        if outcome is None:
            outcome = {"outcome": "cancelled"}
        if not pending.future.done():
            pending.future.set_result(outcome)
        resolved = _with_status(permission_notice(
            pending.platform_session_id, pending.tool_call_id, pending.options, "Approve tool call"
        ), "resolved")
        await self.notice_callback(resolved)
        return outcome

    async def cancel_all(self) -> None:
        for notice_id in list(self._pending):
            await self.expire(notice_id)

    def notices(self, platform_session_id: str | None = None) -> tuple[SessionNotice, ...]:
        """Open approval notices, optionally filtered by platform session id."""
        return tuple(
            pending.notice
            for pending in self._pending.values()
            if platform_session_id is None or pending.platform_session_id == platform_session_id
        )


def _with_status(notice: SessionNotice, status: str) -> SessionNotice:
    return SessionNotice(
        notice_id=notice.notice_id,
        session_id=notice.session_id,
        runtime=notice.runtime,
        type=notice.type,
        title=notice.title,
        message=notice.message,
        severity=notice.severity,
        status=status,
        interaction_type=notice.interaction_type,
        blocking=notice.blocking,
        response_required=notice.response_required,
        actions=notice.actions,
        source=notice.source,
        context=notice.context,
        metadata=notice.metadata,
    )
