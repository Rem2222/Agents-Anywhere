from __future__ import annotations

import asyncio
from pathlib import Path

import pytest

from connector.runtime_protocol import RuntimeConfig, RuntimeInvalidRequestError
from connector.runtimes.acp.discovery import AcpAgentInfo, AcpDiscovery
from connector.runtimes.acp.provider import AcpProvider
from connector.runtimes.providers import default_runtime_providers


def test_acp_is_fourth_default_provider() -> None:
    assert [provider.runtime for provider in default_runtime_providers()] == ["codex", "claude", "dsh", "acp"]


class _Host:
    connector_id = "test"


def test_provider_identity_and_descriptor() -> None:
    async def run() -> None:
        provider = AcpProvider(discoverer=lambda _values: _ok_discovery())
        assert provider.runtime_type == "acp"
        assert provider.display_name == "ACP agent"
        assert provider.instance_policy == "multiple"
        descriptor = await provider.discover()
        assert descriptor.instance_policy == "multiple"
        assert descriptor.max_instances is None
        assert descriptor.capabilities["startTurn"] is True
        schema = await provider.get_config_schema()
        assert "agentPreset" in schema.schema["properties"]
        assert schema.metadata["presetCatalog"]["presets"]
    asyncio.run(run())


async def _ok_discovery() -> AcpDiscovery:
    return AcpDiscovery(
        True,
        True,
        AcpAgentInfo("hermes-agent", "0.21.6", 1),
        None,
        {
            "presetCatalog": {},
            "agentInfo": {"name": "hermes-agent", "version": "0.21.6", "protocolVersion": 1},
        },
    )


async def _down_discovery() -> AcpDiscovery:
    return AcpDiscovery(False, False, None, "no agent", {})


def test_validate_config_accepts_a_live_agent(tmp_path: Path) -> None:
    async def run() -> None:
        provider = AcpProvider(discoverer=lambda _v: _ok_discovery(), prober=lambda _v: _ok_discovery())
        config = await provider.validate_config(
            {"command": "hermes-acp", "agentHome": str(tmp_path / "home"), "cwd": str(tmp_path)}
        )
        assert config.runtime == "acp"
        assert config.metadata["sameSessionWriterLimit"] == 1
        assert config.metadata["agentInfo"]["name"] == "hermes-agent"
    asyncio.run(run())


def test_validate_config_rejects_a_broken_binary() -> None:
    async def run() -> None:
        provider = AcpProvider(prober=lambda _v: _down_discovery())
        with pytest.raises(RuntimeInvalidRequestError):
            await provider.validate_config({"command": "broken", "agentHome": "/tmp", "cwd": "/tmp"})
    asyncio.run(run())


def test_resource_claims_and_session_source_key(tmp_path: Path) -> None:
    config = RuntimeConfig("acp", 1, {"agentHome": str(tmp_path / "home"), "command": "hermes-acp"})
    provider = AcpProvider()
    claims = provider.resource_claims(config)
    assert any(claim.kind == "acp_agent_home" for claim in claims)
    source = provider.session_source_key(config)
    assert source.kind == "acp_agent_home"
    assert "secret" not in source.key


def test_custom_preset_claims_the_command(tmp_path: Path) -> None:
    config = RuntimeConfig("acp", 1, {"agentHome": str(tmp_path / "home"), "agentPreset": "custom", "command": "/opt/my-agent"})
    provider = AcpProvider()
    claims = provider.resource_claims(config)
    assert any(claim.kind == "acp_command" for claim in claims)
