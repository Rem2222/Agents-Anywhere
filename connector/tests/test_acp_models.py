from __future__ import annotations

from connector.runtime_protocol import RuntimeTimelineItem
from connector.runtimes.acp.models import (
    AcpSessionState,
    SessionMetaCall,
    SessionStateCall,
    TimelineItemCall,
    TurnEndedCall,
    close_open_items,
    reduce_session_update,
    turn_end_call,
    turn_start_call,
)


def _state() -> AcpSessionState:
    return AcpSessionState("sess_platform", "ext-1")


def _items(calls):
    return [call.item for call in calls if isinstance(call, TimelineItemCall)]


def test_agent_message_stream_accumulates_text() -> None:
    state = _state()
    calls = reduce_session_update(state, {"sessionUpdate": "agent_message_chunk", "messageId": "m1", "content": {"type": "text", "text": "Hel"}})
    items = _items(calls)
    assert len(items) == 1
    assert items[0].type == "message"
    assert items[0].status == "running"
    assert items[0].content["text"] == "Hel"
    calls = reduce_session_update(state, {"sessionUpdate": "agent_message_chunk", "messageId": "m1", "content": {"type": "text", "text": "lo"}})
    items = _items(calls)
    assert items[0].content["text"] == "Hello"
    assert items[0].revision == 2


def test_message_id_change_closes_previous() -> None:
    state = _state()
    reduce_session_update(state, {"sessionUpdate": "agent_message_chunk", "messageId": "m1", "content": {"type": "text", "text": "a"}})
    calls = reduce_session_update(state, {"sessionUpdate": "agent_message_chunk", "messageId": "m2", "content": {"type": "text", "text": "b"}})
    items = _items(calls)
    assert [item.content["text"] for item in items] == ["a", "b"]
    assert [item.status for item in items] == ["done", "running"]


def test_tool_call_lifecycle() -> None:
    state = _state()
    start = reduce_session_update(
        state,
        {"sessionUpdate": "tool_call", "toolCallId": "tc1", "kind": "execute", "title": "ls", "rawInput": {"command": "ls"}},
    )
    start_item = _items(start)[0]
    assert start_item.type == "tool"
    assert start_item.status == "running"
    assert start_item.content["kind"] == "command"
    assert start_item.content["command"] == "ls"

    done = reduce_session_update(state, {"sessionUpdate": "tool_call_update", "toolCallId": "tc1", "status": "completed", "rawOutput": "ok"})
    done_item = _items(done)[0]
    assert done_item.status == "done"
    assert done_item.id == start_item.id
    assert done_item.content["output"] == "ok"


def test_tool_call_pending_is_waiting_approval() -> None:
    state = _state()
    calls = reduce_session_update(
        state,
        {"sessionUpdate": "tool_call_update", "toolCallId": "tc1", "kind": "execute", "title": "rm", "status": "pending"},
    )
    item = _items(calls)[0]
    assert item.status == "waiting_approval"


def test_session_info_update_produces_meta_call() -> None:
    state = _state()
    calls = reduce_session_update(state, {"sessionUpdate": "session_info_update", "title": "New title"})
    assert any(isinstance(call, SessionMetaCall) and call.title == "New title" for call in calls)


def test_usage_and_mode_updates() -> None:
    state = _state()
    usage = reduce_session_update(state, {"sessionUpdate": "usage_update", "used": 100, "size": 1000})
    assert any(isinstance(call, SessionStateCall) and call.metadata == {"usage": {"used": 100, "size": 1000}} for call in usage)
    mode = reduce_session_update(state, {"sessionUpdate": "current_mode_update", "currentModeId": "accept_edits"})
    state_call = next(call for call in mode if isinstance(call, SessionStateCall))
    assert state_call.selections is not None
    assert "permission" in state_call.selections


def test_unknown_variant_is_ignored() -> None:
    state = _state()
    assert reduce_session_update(state, {"sessionUpdate": "some_future_variant", "x": 1}) == []


def test_close_open_items_and_turn_boundaries() -> None:
    state = _state()
    reduce_session_update(state, {"sessionUpdate": "agent_message_chunk", "messageId": "m1", "content": {"type": "text", "text": "hi"}})
    closed = close_open_items(state)
    assert _items(closed)[0].status == "done"

    turn_id = state.next_turn_id()
    start = turn_start_call(state, turn_id)
    assert start.item.type == "turn.start"
    end = turn_end_call(state, turn_id)
    assert end.item.type == "turn.end"
    assert isinstance(end.item, RuntimeTimelineItem)


def test_user_message_chunk_produces_user_message() -> None:
    state = _state()
    calls = reduce_session_update(state, {"sessionUpdate": "user_message_chunk", "content": {"type": "text", "text": "hello"}})
    item = _items(calls)[0]
    assert item.type == "message"
    assert item.role == "user"
    assert item.status == "done"


def test_turn_ended_call_type() -> None:
    state = _state()
    calls = close_open_items(state)
    calls.append(TurnEndedCall(turn_id="t1", outcome="completed"))
    assert any(isinstance(call, TurnEndedCall) for call in calls)
