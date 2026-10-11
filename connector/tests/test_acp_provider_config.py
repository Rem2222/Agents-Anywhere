from __future__ import annotations

import asyncio
from pathlib import Path

import pytest

from connector.runtime_protocol import RuntimeInvalidRequestError
from connector.runtimes.acp import presets, provider_config
from connector.runtimes.acp.discovery import AcpDiscovery


def test_default_config_values_and_schema() -> None:
    schema = provider_config.acp_config_schema()
    assert schema["type"] == "object"
    assert set(provider_config.default_config_values()) >= {
        "agentPreset",
        "command",
        "args",
        "cwd",
        "agentHome",
        "environment",
        "startupTimeoutMs",
        "requestTimeoutMs",
        "permissionTimeoutMs",
        "idleTimeoutSeconds",
        "maxRestartAttempts",
        "restartBackoffMs",
        "customModels",
    }
    assert "customModels" in schema["properties"]
    assert schema["properties"]["permissionTimeoutMs"]["minimum"] == 60_000


def test_normalization_requires_command_and_absolute_home(tmp_path: Path) -> None:
    with pytest.raises(RuntimeInvalidRequestError, match="command"):
        provider_config.normalized_config_values({"command": "", "agentHome": str(tmp_path), "cwd": str(tmp_path)})
    with pytest.raises(RuntimeInvalidRequestError, match="agentHome"):
        provider_config.normalized_config_values(
            {"command": "hermes-acp", "agentHome": "relative/home", "cwd": str(tmp_path)}
        )


def test_normalization_canonicalizes_paths_and_custom_models(tmp_path: Path) -> None:
    values = provider_config.normalized_config_values(
        {
            "command": "hermes-acp",
            "agentHome": str(tmp_path / "sub" / ".." / "home"),
            "cwd": str(tmp_path / "work"),
            "customModels": [{"modelId": "m1", "displayName": "Model One"}],
        }
    )
    assert values["agentHome"] == str((tmp_path / "home").resolve())
    assert values["customModels"] == [{"modelId": "m1", "displayName": "Model One"}]


def test_environment_guard_blocks_managed_names() -> None:
    with pytest.raises(RuntimeInvalidRequestError, match="managed"):
        provider_config.normalized_config_values(
            {"command": "hermes-acp", "agentHome": "/tmp", "cwd": "/tmp", "environment": {"AGENT_CONNECTOR_TOKEN": "x"}}
        )


def test_command_and_args_resolve_hermes_preset(monkeypatch) -> None:
    values = provider_config.default_config_values()
    values["agentHome"] = "/tmp"
    values["cwd"] = "/tmp"
    command, args = provider_config.command_and_args(values)
    # Falls back to the first hermes candidate even when not on PATH.
    assert command == "hermes-acp"
    assert args == ()


def test_capabilities_are_static_and_complete() -> None:
    capabilities = provider_config.acp_capabilities()
    assert capabilities["startTurn"] is True
    assert capabilities["steerTurn"] is False
    assert capabilities["interactions"] is True
    assert capabilities["ipc"] is False


def test_preset_catalog_marks_custom_as_enabled() -> None:
    catalog = presets.preset_catalog()
    ids = {row["id"] for row in catalog["presets"]}
    assert "hermes" in ids and "custom" in ids
    custom = next(row for row in catalog["presets"] if row["id"] == "custom")
    assert custom["enabled"] is True


def test_agent_info_extraction() -> None:
    from connector.runtimes.acp import discovery

    info = discovery._agent_info(
        {
            "protocolVersion": 1,
            "agentInfo": {"name": "hermes-agent", "version": "0.21.6"},
            "agentCapabilities": {
                "loadSession": True,
                "promptCapabilities": {"image": True},
                "sessionCapabilities": {"fork": {}, "list": {}, "resume": {}},
            },
            "authMethods": [{"type": "terminal"}],
        }
    )
    assert info.name == "hermes-agent"
    assert info.load_session is True
    assert info.image_attachments is True
    assert info.list_sessions is True


def test_probe_failure_returns_unavailable() -> None:
    from connector.runtimes.acp import discovery

    async def run() -> None:
        result = await discovery.probe({"command": "/definitely/not/a/binary", "agentHome": "/tmp", "cwd": "/tmp"})
        assert isinstance(result, AcpDiscovery)
        assert result.available is False
        assert result.reason

    asyncio.run(run())
