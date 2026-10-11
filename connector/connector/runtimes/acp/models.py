from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from connector.runtime_protocol import (
    CommandToolContent,
    FileChangeToolContent,
    MarkdownMessageContent,
    MessageTimelineItem,
    ReasoningSystemContent,
    RuntimeTimelineItem,
    SessionNotice,
    SystemTimelineItem,
    TimelineSource,
    ToolCallContent,
    ToolTimelineItem,
    TurnEndSystemContent,
    TurnEndTimelineItem,
    TurnStartSystemContent,
    TurnStartTimelineItem,
)
from connector.runtimes.acp.catalogs import permission_selection_id


@dataclass(frozen=True, slots=True)
class TimelineItemCall:
    item: RuntimeTimelineItem


@dataclass(frozen=True, slots=True)
class TimelineSyncCall:
    items: tuple[RuntimeTimelineItem, ...]
    complete: bool


@dataclass(frozen=True, slots=True)
class SessionStateCall:
    status: str | None = None
    selections: Mapping[str, str | None] | None = None
    metadata: Mapping[str, Any] | None = None


@dataclass(frozen=True, slots=True)
class SessionMetaCall:
    title: str | None = None
    ordering_time: str | None = None


@dataclass(frozen=True, slots=True)
class NoticeCall:
    notice: SessionNotice


@dataclass(frozen=True, slots=True)
class TurnEndedCall:
    turn_id: str | None
    outcome: str


HostCall = (
    TimelineItemCall
    | TimelineSyncCall
    | SessionStateCall
    | SessionMetaCall
    | NoticeCall
    | TurnEndedCall
)


def _text(content: Any) -> str:
    if isinstance(content, dict) and content.get("type") == "text":
        text = content.get("text")
        return text if isinstance(text, str) else ""
    if isinstance(content, str):
        return content
    return ""


class AcpSessionState:
    """Per-session monotonic order and open item bookkeeping for the reducer."""

    def __init__(self, session_id: str, external_session_id: str) -> None:
        self.session_id = session_id
        self.external_session_id = external_session_id
        self._next_order_seq = 0
        self._order_by_id: dict[str, int] = {}
        self._revision: dict[str, int] = {}
        self._open_message_id: str | None = None
        self._message_text = ""
        self._open_thought_id: str | None = None
        self._thought_text = ""
        self._tools: dict[str, dict[str, Any]] = {}
        self.commands: list[dict[str, Any]] = []
        self.turn_counter = 0

    def _order_for(self, item_id: str) -> int:
        order_seq = self._order_by_id.get(item_id)
        if order_seq is None:
            self._next_order_seq += 1
            order_seq = self._next_order_seq
            self._order_by_id[item_id] = order_seq
        return order_seq

    def _revision_for(self, item_id: str) -> int:
        revision = self._revision.get(item_id, 0) + 1
        self._revision[item_id] = revision
        return revision

    def _source(self, event: str, native_id: str | None, origin: str) -> TimelineSource:
        return TimelineSource(
            runtime="acp",
            external_session_id=self.external_session_id,
            native_item_id=native_id,
            event=event,
            metadata={"origin": origin},
        )

    def next_turn_id(self) -> str:
        self.turn_counter += 1
        return f"acp:turn:{self.turn_counter}"


def _message_item_call(
    state: AcpSessionState,
    *,
    item_id: str,
    message_id: str,
    text: str,
    status: str,
    event: str,
    origin: str,
) -> TimelineItemCall:
    item = MessageTimelineItem(
        id=item_id,
        type="message",
        status=status,  # type: ignore[arg-type]
        role="assistant",
        content=MarkdownMessageContent(text=text),
        source=state._source(event, message_id, origin),
        revision=state._revision_for(item_id),
    ).to_platform_item(session_id=state.session_id, order_seq=state._order_for(item_id))
    return TimelineItemCall(item=item)


def _thought_item_call(
    state: AcpSessionState,
    *,
    item_id: str,
    message_id: str,
    text: str,
    status: str,
    event: str,
    origin: str,
) -> TimelineItemCall:
    item = SystemTimelineItem(
        id=item_id,
        type="system",
        status=status,  # type: ignore[arg-type]
        role="system",
        content=ReasoningSystemContent(text=text),
        source=state._source(event, message_id, origin),
        revision=state._revision_for(item_id),
    ).to_platform_item(session_id=state.session_id, order_seq=state._order_for(item_id))
    return TimelineItemCall(item=item)


def _tool_content(kind: str, title: str, raw_input: Any, raw_output: Any):
    if kind == "execute" and isinstance(raw_input, dict) and raw_input.get("command"):
        return CommandToolContent(
            title=title,
            command=str(raw_input["command"]),
            input=raw_input,
            output=raw_output,
        )
    if kind in {"edit", "delete", "move"}:
        return FileChangeToolContent(title=title, input=raw_input, output=raw_output)
    return ToolCallContent(title=title, input=raw_input, output=raw_output)


def _tool_status(status: str) -> str:
    if status == "completed":
        return "done"
    if status == "failed":
        return "failed"
    if status == "pending":
        return "waiting_approval"
    return "running"


def _tool_item_call(
    state: AcpSessionState,
    *,
    tool_call_id: str,
    kind: str,
    title: str,
    status: str,
    raw_input: Any,
    raw_output: Any,
    event: str,
    origin: str,
) -> TimelineItemCall:
    item_id = f"acp:tool:{tool_call_id}"
    content = _tool_content(kind, title, raw_input, raw_output)
    item = ToolTimelineItem(
        id=item_id,
        type="tool",
        status=status,  # type: ignore[arg-type]
        role="tool",
        content=content,
        source=state._source(event, tool_call_id, origin),
        revision=state._revision_for(item_id),
    ).to_platform_item(session_id=state.session_id, order_seq=state._order_for(item_id))
    return TimelineItemCall(item=item)


def reduce_session_update(
    state: AcpSessionState,
    session_update: Mapping[str, Any],
    *,
    origin: str = "live",
) -> list[HostCall]:
    """Project one ACP ``session/update`` payload into host calls.

    Pure with respect to I/O: the caller applies the returned calls. ``state``
    is mutated to keep order/sequence and open-item bookkeeping per session.
    """
    variant = session_update.get("sessionUpdate")
    calls: list[HostCall] = []

    if variant == "agent_message_chunk":
        message_id = session_update.get("messageId")
        message_id = message_id if isinstance(message_id, str) and message_id else "assistant"
        text = _text(session_update.get("content"))
        if state._open_message_id is not None and state._open_message_id != message_id:
            calls.append(
                _message_item_call(
                    state,
                    item_id=f"acp:msg:{state._open_message_id}",
                    message_id=state._open_message_id,
                    text=state._message_text,
                    status="done",
                    event="agent_message_chunk",
                    origin=origin,
                )
            )
            state._open_message_id = None
            state._message_text = ""
        if state._open_message_id is None:
            state._open_message_id = message_id
            state._message_text = ""
        state._message_text += text
        calls.append(
            _message_item_call(
                state,
                item_id=f"acp:msg:{message_id}",
                message_id=message_id,
                text=state._message_text,
                status="running",
                event="agent_message_chunk",
                origin=origin,
            )
        )
        return calls

    if variant == "agent_thought_chunk":
        message_id = session_update.get("messageId")
        message_id = message_id if isinstance(message_id, str) and message_id else "assistant"
        text = _text(session_update.get("content"))
        if state._open_thought_id is not None and state._open_thought_id != message_id:
            calls.append(
                _thought_item_call(
                    state,
                    item_id=f"acp:thought:{state._open_thought_id}",
                    message_id=state._open_thought_id,
                    text=state._thought_text,
                    status="done",
                    event="agent_thought_chunk",
                    origin=origin,
                )
            )
            state._open_thought_id = None
            state._thought_text = ""
        if state._open_thought_id is None:
            state._open_thought_id = message_id
            state._thought_text = ""
        state._thought_text += text
        calls.append(
            _thought_item_call(
                state,
                item_id=f"acp:thought:{message_id}",
                message_id=message_id,
                text=state._thought_text,
                status="running",
                event="agent_thought_chunk",
                origin=origin,
            )
        )
        return calls

    if variant == "user_message_chunk":
        content = session_update.get("content")
        text = _text(content)
        message_id = session_update.get("messageId")
        key = message_id if isinstance(message_id, str) and message_id else text
        item_id = f"acp:umsg:{hash(key) & 0xFFFFFFFFFFFFFFFF:x}"
        item = MessageTimelineItem(
            id=item_id,
            type="message",
            status="done",
            role="user",
            content=MarkdownMessageContent(text=text),
            source=state._source("user_message_chunk", message_id if isinstance(message_id, str) else None, origin),
            revision=state._revision_for(item_id),
        ).to_platform_item(session_id=state.session_id, order_seq=state._order_for(item_id))
        calls.append(TimelineItemCall(item=item))
        return calls

    if variant == "tool_call":
        tool_call_id = session_update.get("toolCallId")
        if not isinstance(tool_call_id, str) or not tool_call_id:
            return calls
        kind = session_update.get("kind") if isinstance(session_update.get("kind"), str) else "other"
        title = session_update.get("title") if isinstance(session_update.get("title"), str) else kind
        raw_input = session_update.get("rawInput")
        raw_output = session_update.get("rawOutput")
        status = session_update.get("status")
        status = status if isinstance(status, str) else "in_progress"
        if kind == "think":
            text = str(title) if title else ""
            calls.append(
                _thought_item_call(
                    state,
                    item_id=f"acp:tool:{tool_call_id}",
                    message_id=tool_call_id,
                    text=text,
                    status="done",
                    event="tool_call",
                    origin=origin,
                )
            )
            return calls
        state._tools[tool_call_id] = {"kind": kind, "title": title, "raw_input": raw_input}
        calls.append(
            _tool_item_call(
                state,
                tool_call_id=tool_call_id,
                kind=kind,
                title=title,
                status=_tool_status(status),
                raw_input=raw_input,
                raw_output=raw_output,
                event="tool_call",
                origin=origin,
            )
        )
        return calls

    if variant == "tool_call_update":
        tool_call_id = session_update.get("toolCallId")
        if not isinstance(tool_call_id, str) or not tool_call_id:
            return calls
        meta = state._tools.get(tool_call_id, {})
        kind = session_update.get("kind")
        if not isinstance(kind, str):
            kind = meta.get("kind") if isinstance(meta.get("kind"), str) else "other"
        title = session_update.get("title")
        if not isinstance(title, str):
            title = meta.get("title") if isinstance(meta.get("title"), str) else kind
        raw_input = session_update.get("rawInput") if session_update.get("rawInput") is not None else meta.get("raw_input")
        raw_output = session_update.get("rawOutput")
        status = session_update.get("status")
        status = status if isinstance(status, str) else "completed"
        state._tools[tool_call_id] = {"kind": kind, "title": title, "raw_input": raw_input}
        calls.append(
            _tool_item_call(
                state,
                tool_call_id=tool_call_id,
                kind=kind,
                title=title,
                status=_tool_status(status),
                raw_input=raw_input,
                raw_output=raw_output,
                event="tool_call_update",
                origin=origin,
            )
        )
        return calls

    if variant == "plan":
        entries = session_update.get("entries")
        calls.append(SessionStateCall(metadata={"plan": entries}))
        return calls

    if variant == "available_commands_update":
        commands = session_update.get("availableCommands")
        if isinstance(commands, list):
            state.commands = [command for command in commands if isinstance(command, dict)]
        return calls

    if variant == "current_mode_update":
        current_mode_id = session_update.get("currentModeId")
        if isinstance(current_mode_id, str) and current_mode_id:
            calls.append(
                SessionStateCall(selections={"permission": permission_selection_id(current_mode_id)})
            )
        return calls

    if variant == "config_option_update":
        calls.append(SessionStateCall(metadata={"configOptions": session_update.get("configOptions")}))
        return calls

    if variant == "session_info_update":
        title = session_update.get("title")
        updated_at = session_update.get("updatedAt")
        calls.append(
            SessionMetaCall(
                title=title if isinstance(title, str) and title else None,
                ordering_time=updated_at if isinstance(updated_at, str) else None,
            )
        )
        return calls

    if variant == "usage_update":
        usage = {key: session_update.get(key) for key in ("used", "size", "cost") if session_update.get(key) is not None}
        if usage:
            calls.append(SessionStateCall(metadata={"usage": usage}))
        return calls

    # Unknown variants are ignored so newer schema revisions never break an instance.
    return calls


def close_open_items(state: AcpSessionState, *, origin: str = "live") -> list[HostCall]:
    """Close the open assistant message and thought at turn end."""
    calls: list[HostCall] = []
    if state._open_message_id is not None:
        calls.append(
            _message_item_call(
                state,
                item_id=f"acp:msg:{state._open_message_id}",
                message_id=state._open_message_id,
                text=state._message_text,
                status="done",
                event="agent_message_chunk",
                origin=origin,
            )
        )
        state._open_message_id = None
        state._message_text = ""
    if state._open_thought_id is not None:
        calls.append(
            _thought_item_call(
                state,
                item_id=f"acp:thought:{state._open_thought_id}",
                message_id=state._open_thought_id,
                text=state._thought_text,
                status="done",
                event="agent_thought_chunk",
                origin=origin,
            )
        )
        state._open_thought_id = None
        state._thought_text = ""
    return calls


def turn_start_call(state: AcpSessionState, turn_id: str, *, origin: str = "live") -> TimelineItemCall:
    item_id = turn_id + ":start"
    item = TurnStartTimelineItem(
        id=item_id,
        type="turn.start",
        status="running",
        role="system",
        content=TurnStartSystemContent(),
        source=state._source("session.prompt", turn_id, origin),
        revision=state._revision_for(item_id),
    ).to_platform_item(session_id=state.session_id, order_seq=state._order_for(item_id))
    return TimelineItemCall(item=item)


def turn_end_call(state: AcpSessionState, turn_id: str, *, origin: str = "live") -> TimelineItemCall:
    item_id = turn_id + ":end"
    item = TurnEndTimelineItem(
        id=item_id,
        type="turn.end",
        status="done",
        role="system",
        content=TurnEndSystemContent(),
        source=state._source("session.prompt", turn_id, origin),
        revision=state._revision_for(item_id),
    ).to_platform_item(session_id=state.session_id, order_seq=state._order_for(item_id))
    return TimelineItemCall(item=item)
