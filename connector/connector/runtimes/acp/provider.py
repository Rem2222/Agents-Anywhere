from __future__ import annotations

from collections.abc import Awaitable, Callable, Mapping
from typing import Any

from jsonschema import Draft202012Validator

from connector.runtime_protocol import (
    AgentRuntime,
    RuntimeConfig,
    RuntimeConfigSchema,
    RuntimeInvalidRequestError,
    RuntimeProvider,
    RuntimeResourceClaim,
    RuntimeSourceKey,
    RuntimeTypeDescriptor,
)
from connector.runtime_protocol.filesystem import (
    filesystem_resource_key,
)
from connector.runtime_protocol.host import RuntimeHostClient
from connector.runtimes.acp import discovery, presets, provider_config
from connector.runtimes.acp.runtime import AcpRuntime

ACP_CONFIG_SCHEMA_REVISION = 1
Discovery = Callable[[dict[str, Any]], Awaitable[discovery.AcpDiscovery]]


class AcpProvider(RuntimeProvider):
    def __init__(
        self,
        discoverer: Discovery | None = None,
        prober: Discovery | None = None,
    ) -> None:
        self._discoverer = discoverer or discovery.discover
        self._prober = prober or discovery.probe
        self._last_discovery: discovery.AcpDiscovery | None = None

    @property
    def runtime_type(self) -> str:
        return "acp"

    @property
    def implementation_type(self) -> str:
        return "acp"

    @property
    def display_name(self) -> str:
        return "ACP agent"

    @property
    def description(self) -> str:
        return "Agent Client Protocol agent runtime (Hermes or any ACP agent)"

    @property
    def instance_policy(self):
        return "multiple"

    async def discover(self) -> RuntimeTypeDescriptor:
        values = provider_config.default_config_values()
        result = await self._discoverer(values)
        self._last_discovery = result
        metadata = dict(result.metadata or {})
        metadata.update(
            {
                "protocolVersion": "1",
                "sameSessionWriterLimit": 1,
                "crossProcessWriterExclusion": True,
            }
        )
        return RuntimeTypeDescriptor(
            runtime_type=self.runtime_type,
            display_name=self.display_name,
            description=self.description,
            implementation_type=self.implementation_type,
            available=result.available,
            capabilities=provider_config.acp_capabilities(),
            reason=None if result.available else (result.reason or "ACP agent is unavailable"),
            config_schema=self._config_schema(),
            instance_policy="multiple",
            max_instances=None,
            recommended=False,
            metadata=metadata,
        )

    async def get_config_schema(self) -> RuntimeConfigSchema:
        return self._config_schema()

    def _config_schema(self) -> RuntimeConfigSchema:
        schema = provider_config.acp_config_schema()
        defaults = provider_config.default_config_values()
        catalog = presets.preset_catalog()
        field = catalog.get("configField")
        if isinstance(field, dict):
            schema["properties"]["agentPreset"] = dict(field)
            if isinstance(field.get("default"), str):
                defaults["agentPreset"] = field["default"]
        return RuntimeConfigSchema(
            runtime=self.runtime_type,
            revision=ACP_CONFIG_SCHEMA_REVISION,
            schema=schema,
            ui_schema={
                "order": [
                    "agentPreset",
                    "command",
                    "args",
                    "agentHome",
                    "cwd",
                    "environment",
                    "startupTimeoutMs",
                    "requestTimeoutMs",
                    "permissionTimeoutMs",
                    "idleTimeoutSeconds",
                    "maxRestartAttempts",
                    "restartBackoffMs",
                    "customModels",
                ],
                "command": {"component": "path"},
                "agentHome": {"component": "path"},
                "cwd": {"component": "path"},
                "environment": {"component": "keyValue"},
                "agentPreset": catalog.get("uiField", {"component": "select", "options": []}),
            },
            defaults=defaults,
            metadata={
                "presetCatalog": catalog,
                "sameSessionWriterLimit": 1,
                "crossProcessWriterExclusion": True,
            },
        )

    async def validate_config(self, values: Mapping[str, Any]) -> RuntimeConfig:
        raw = {key: value for key, value in values.items()}
        errors = sorted(
            Draft202012Validator(provider_config.acp_config_schema()).iter_errors(raw),
            key=lambda error: list(error.absolute_path),
        )
        if errors:
            path = "/" + "/".join(str(part) for part in errors[0].absolute_path)
            raise RuntimeInvalidRequestError(f"acp config is invalid at {path or '/'}: {errors[0].message}")
        normalized = provider_config.normalized_config_values(raw)
        result = await self._prober(normalized)
        self._last_discovery = result
        if not result.available:
            raise RuntimeInvalidRequestError(result.reason or "ACP agent did not answer the initialize handshake")
        metadata = dict(result.metadata or {})
        metadata.update(
            {
                "protocolVersion": "1",
                "sameSessionWriterLimit": 1,
                "crossProcessWriterExclusion": True,
            }
        )
        schema_info = self._config_schema()
        return RuntimeConfig(
            runtime=self.runtime_type,
            revision=ACP_CONFIG_SCHEMA_REVISION,
            values=normalized,
            schema=schema_info.schema,
            ui_schema=schema_info.ui_schema,
            metadata=metadata,
        )

    async def create_runtime(self, config: RuntimeConfig, host: RuntimeHostClient) -> AgentRuntime:
        return AcpRuntime(config=config, host=host)

    def resource_claims(self, config: RuntimeConfig) -> tuple[RuntimeResourceClaim, ...]:
        values = dict(config.values)
        home = str(values.get("agentHome") or provider_config.agent_home(values))
        claims = [
            RuntimeResourceClaim(
                kind="acp_agent_home",
                key=filesystem_resource_key(home),
                label=f"ACP agent home {home!r}",
            )
        ]
        preset_id = values.get("agentPreset")
        preset = presets.preset_by_id(preset_id) if isinstance(preset_id, str) else None
        if preset is None or preset.home_env is None:
            try:
                command, _args = provider_config.command_and_args(values)
            except RuntimeInvalidRequestError:
                command = ""
            if command:
                claims.append(
                    RuntimeResourceClaim(
                        kind="acp_command",
                        key=filesystem_resource_key(command),
                        label=f"ACP command {command!r}",
                    )
                )
        return tuple(claims)

    def session_source_key(self, config: RuntimeConfig) -> RuntimeSourceKey:
        values = dict(config.values)
        home = str(values.get("agentHome") or provider_config.agent_home(values))
        return RuntimeSourceKey(
            kind="acp_agent_home",
            key=filesystem_resource_key(home),
        )
