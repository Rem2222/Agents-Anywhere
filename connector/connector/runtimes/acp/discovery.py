from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from connector.runtimes.acp import presets, provider_config
from connector.runtimes.acp.client import AcpClient, AcpNoResponse


@dataclass(frozen=True, slots=True)
class AcpAgentInfo:
    name: str
    version: str
    protocol_version: int
    load_session: bool = False
    resume: bool = False
    fork: bool = False
    list_sessions: bool = False
    image_attachments: bool = False
    auth_methods: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class AcpDiscovery:
    available: bool
    configured: bool
    agent: AcpAgentInfo | None = None
    reason: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


def _static_metadata(values: dict[str, Any]) -> dict[str, Any]:
    return {
        "platform": "acp",
        "presetCatalog": presets.preset_catalog(),
        "sameSessionWriterLimit": 1,
        "crossProcessWriterExclusion": True,
    }


async def discover(values: dict[str, Any]) -> AcpDiscovery:
    """Report that this connector supports the ``acp`` runtime type.

    Discovery never touches a process; whether an agent binary is actually
    reachable is a configuration/start concern handled by ``probe``.
    """
    return AcpDiscovery(True, False, None, None, _static_metadata(values))


def _agent_info(result: dict[str, Any]) -> AcpAgentInfo:
    agent_info = result.get("agentInfo") if isinstance(result.get("agentInfo"), dict) else {}
    capabilities = result.get("agentCapabilities") if isinstance(result.get("agentCapabilities"), dict) else {}
    prompt_capabilities = (
        capabilities.get("promptCapabilities") if isinstance(capabilities.get("promptCapabilities"), dict) else {}
    )
    session_capabilities = (
        capabilities.get("sessionCapabilities") if isinstance(capabilities.get("sessionCapabilities"), dict) else {}
    )
    auth_methods = tuple(
        str(item.get("type")) for item in result.get("authMethods", []) if isinstance(item, dict) and item.get("type")
    )
    return AcpAgentInfo(
        name=str(agent_info.get("name") or agent_info.get("title") or "acp-agent"),
        version=str(agent_info.get("version") or "unknown"),
        protocol_version=int(result.get("protocolVersion") or 1),
        load_session=capabilities.get("loadSession") is True,
        resume=session_capabilities.get("resume") is not None,
        fork=session_capabilities.get("fork") is not None,
        list_sessions=session_capabilities.get("list") is not None,
        image_attachments=prompt_capabilities.get("image") is True,
        auth_methods=auth_methods,
    )


async def probe(values: dict[str, Any]) -> AcpDiscovery:
    """Spawn the configured agent and complete an ``initialize`` handshake.

    A broken binary or a non-ACP process is an invalid configuration, not an
    offline runtime: the caller maps this to ``RuntimeInvalidRequestError``.
    """
    command, args = provider_config.command_and_args(values)
    env = dict(values.get("environment") or {})
    preset_id = values.get("agentPreset")
    preset = presets.preset_by_id(preset_id) if isinstance(preset_id, str) else None
    if preset and preset.home_env:
        env.setdefault(preset.home_env, str(values.get("agentHome") or ""))
    cwd = values.get("cwd") or None
    startup_timeout = int(values.get("startupTimeoutMs", 30_000)) / 1000

    async def no_notification(_method: str, _params: Any) -> None:
        return None

    async def no_exit(_code: int | None) -> None:
        return None

    async def no_request(_method: str, _params: Any) -> Any:
        raise AcpNoResponse()

    client = AcpClient(
        command=command,
        args=args,
        env=env,
        cwd=cwd,
        client_version="1.0",
        startup_timeout=startup_timeout,
        request_timeout=startup_timeout,
        notification_handler=no_notification,
        request_handler=no_request,
        exit_handler=no_exit,
    )
    try:
        result = await client.start()
    except (OSError, RuntimeError, ValueError, TimeoutError) as exc:
        return AcpDiscovery(
            False,
            False,
            None,
            f"ACP agent {command!r} did not complete the initialize handshake: {exc}",
            _static_metadata(values),
        )
    finally:
        await client.close()
    agent = _agent_info(result)
    metadata = {
        **_static_metadata(values),
        "agentInfo": {
            "name": agent.name,
            "version": agent.version,
            "protocolVersion": agent.protocol_version,
            "loadSession": agent.load_session,
            "resume": agent.resume,
            "fork": agent.fork,
            "list": agent.list_sessions,
            "imageAttachments": agent.image_attachments,
        },
    }
    return AcpDiscovery(True, True, agent, None, metadata)
