from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from connector.runtime_protocol import RuntimeInvalidRequestError
from connector.runtime_protocol.filesystem import canonical_path
from connector.runtimes.acp import presets
from connector.runtimes.custom_models import (
    custom_models_schema,
    normalize_custom_models,
)

DEFAULT_STARTUP_TIMEOUT_MS = 30_000
DEFAULT_REQUEST_TIMEOUT_MS = 600_000
DEFAULT_PERMISSION_TIMEOUT_MS = 300_000
DEFAULT_IDLE_TIMEOUT_SECONDS = 900
DEFAULT_MAX_RESTART_ATTEMPTS = 3
DEFAULT_RESTART_BACKOFF_MS = 1_000

PROTECTED_ENV_PREFIXES = ("AGENT_CONNECTOR_", "AGENT_SERVER_")
PROTECTED_ENV_NAMES = {
    "AGENT_CONNECTOR_ID",
    "AGENT_CONNECTOR_TOKEN",
    "AGENT_CONNECTOR_CONFIG",
    "AGENT_CONNECTOR_DATA_DIR",
    "AGENT_CONNECTOR_STATE_FILE",
    "AGENT_SERVER_URL",
}


def acp_config_schema() -> dict[str, Any]:
    return {
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "type": "object",
        "properties": {
            "agentPreset": {
                "type": "string",
                "title": "ACP agent preset",
                "description": "Which ACP agent this runtime instance drives.",
                "enum": [preset.id for preset in presets.PRESETS],
                "default": "hermes",
            },
            "command": {
                "type": "string",
                "title": "Command",
                "description": "Executable (or path) that starts the ACP stdio server.",
                "minLength": 1,
                "maxLength": 2048,
            },
            "args": {
                "type": "array",
                "title": "Arguments",
                "description": "Extra arguments passed to the command.",
                "items": {"type": "string", "minLength": 1, "maxLength": 2048},
                "default": [],
            },
            "cwd": {
                "type": "string",
                "title": "Working directory",
                "description": "Absolute working directory for the agent process.",
                "minLength": 1,
            },
            "agentHome": {
                "type": "string",
                "title": "Agent home",
                "description": (
                    "Storage directory for the agent's sessions. Passed as the preset's "
                    "home environment variable and used to key session identity."
                ),
                "minLength": 1,
            },
            "environment": {
                "type": "object",
                "title": "Environment variables",
                "description": "Environment overrides for the ACP agent process.",
                "propertyNames": {"pattern": "^[^=\\u0000]+$"},
                "additionalProperties": {"anyOf": [{"type": "string"}, {"type": "null"}]},
                "default": {},
            },
            "startupTimeoutMs": {
                "type": "integer",
                "minimum": 100,
                "maximum": 600_000,
                "default": DEFAULT_STARTUP_TIMEOUT_MS,
                "title": "Startup timeout (ms)",
            },
            "requestTimeoutMs": {
                "type": "integer",
                "minimum": 1_000,
                "maximum": 3_600_000,
                "default": DEFAULT_REQUEST_TIMEOUT_MS,
                "title": "Request timeout (ms)",
                "description": "Long turn operations (`session/prompt`) can run for many minutes.",
            },
            "permissionTimeoutMs": {
                "type": "integer",
                "minimum": 60_000,
                "maximum": 3_600_000,
                "default": DEFAULT_PERMISSION_TIMEOUT_MS,
                "title": "Permission timeout (ms)",
                "description": "Must be at least the agent's own approval timeout (Hermes default 300 s).",
            },
            "idleTimeoutSeconds": {
                "type": "integer",
                "minimum": 0,
                "maximum": 86_400,
                "default": DEFAULT_IDLE_TIMEOUT_SECONDS,
                "title": "Idle recycle timeout (seconds)",
                "description": "Recycle the agent process after this long with no active turns; 0 disables.",
            },
            "maxRestartAttempts": {
                "type": "integer",
                "minimum": 0,
                "maximum": 10,
                "default": DEFAULT_MAX_RESTART_ATTEMPTS,
                "description": "Fast reconnect attempts before polling every 5 seconds.",
            },
            "restartBackoffMs": {
                "type": "integer",
                "minimum": 100,
                "maximum": 600_000,
                "default": DEFAULT_RESTART_BACKOFF_MS,
            },
            "customModels": custom_models_schema(),
        },
        "additionalProperties": False,
    }


def default_config_values() -> dict[str, Any]:
    return {
        "agentPreset": "hermes",
        "command": "",
        "args": [],
        "cwd": "",
        "agentHome": "",
        "environment": {},
        "startupTimeoutMs": DEFAULT_STARTUP_TIMEOUT_MS,
        "requestTimeoutMs": DEFAULT_REQUEST_TIMEOUT_MS,
        "permissionTimeoutMs": DEFAULT_PERMISSION_TIMEOUT_MS,
        "idleTimeoutSeconds": DEFAULT_IDLE_TIMEOUT_SECONDS,
        "maxRestartAttempts": DEFAULT_MAX_RESTART_ATTEMPTS,
        "restartBackoffMs": DEFAULT_RESTART_BACKOFF_MS,
        "customModels": [],
    }


def normalized_config_values(raw: dict[str, Any]) -> dict[str, Any]:
    values = {**default_config_values(), **raw}
    preset_id = values.get("agentPreset")
    preset = presets.preset_by_id(preset_id) if isinstance(preset_id, str) else None
    if preset is None:
        raise RuntimeInvalidRequestError(f"agentPreset must be one of {[p.id for p in presets.PRESETS]}")

    command = values.get("command")
    if not isinstance(command, str) or "\x00" in command or not command.strip():
        raise RuntimeInvalidRequestError("command must be a non-empty string without NUL")
    values["command"] = command.strip()

    args = values.get("args", [])
    if not isinstance(args, list) or any(
        not isinstance(item, str) or not item or "\x00" in item for item in args
    ):
        raise RuntimeInvalidRequestError("args must be an array of non-empty strings without NUL")
    values["args"] = list(args)

    agent_home = values.get("agentHome")
    if not isinstance(agent_home, str) or not agent_home:
        raise RuntimeInvalidRequestError("agentHome must be a non-empty absolute path")
    if not Path(agent_home).expanduser().is_absolute():
        raise RuntimeInvalidRequestError("agentHome must be an absolute path")
    values["agentHome"] = canonical_path(agent_home)
    if preset.home_env:
        values.setdefault("_homeEnv", preset.home_env)

    cwd = values.get("cwd")
    if not isinstance(cwd, str) or not cwd:
        raise RuntimeInvalidRequestError("cwd must be a non-empty absolute path")
    if not Path(cwd).expanduser().is_absolute():
        raise RuntimeInvalidRequestError("cwd must be an absolute path")
    values["cwd"] = canonical_path(cwd)

    values["environment"] = merge_environment(values.get("environment"))

    for key, minimum, maximum in (
        ("startupTimeoutMs", 100, 600_000),
        ("requestTimeoutMs", 1_000, 3_600_000),
        ("permissionTimeoutMs", 60_000, 3_600_000),
        ("restartBackoffMs", 100, 600_000),
        ("idleTimeoutSeconds", 0, 86_400),
        ("maxRestartAttempts", 0, 10),
    ):
        value = values.get(key)
        if not isinstance(value, int) or isinstance(value, bool) or not minimum <= value <= maximum:
            raise RuntimeInvalidRequestError(f"{key} must be an integer between {minimum} and {maximum}")

    if isinstance(values.get("customModels"), list):
        values["customModels"] = normalize_custom_models(values["customModels"])
    return values


def merge_environment(raw: Any) -> dict[str, str]:
    """Merge user overrides over the connector environment, guarding managed names."""
    if raw is None:
        overrides: dict[str, Any] = {}
    elif isinstance(raw, dict):
        overrides = raw
    else:
        raise RuntimeInvalidRequestError("environment must be an object")

    environment = dict(os.environ)
    for key, value in overrides.items():
        if not isinstance(key, str) or not key or "=" in key or "\x00" in key:
            raise RuntimeInvalidRequestError("environment contains an invalid variable name")
        if key in PROTECTED_ENV_NAMES or key.startswith(PROTECTED_ENV_PREFIXES):
            raise RuntimeInvalidRequestError(f"environment variable {key!r} is managed by the connector")
        if value is None:
            environment.pop(key, None)
            continue
        if not isinstance(value, str) or "\x00" in value:
            raise RuntimeInvalidRequestError(f"environment variable {key!r} must be a string or null")
        environment[key] = value
    return environment


def agent_home(values: dict[str, Any]) -> Path:
    configured = values.get("agentHome")
    if isinstance(configured, str) and configured:
        return Path(configured)
    preset_id = values.get("agentPreset")
    preset = presets.preset_by_id(preset_id) if isinstance(preset_id, str) else None
    default_home = presets.default_home_path(preset) if preset else None
    if default_home:
        return Path(default_home)
    home_env = preset.home_env if preset else None
    if home_env and os.environ.get(home_env):
        return Path(canonical_path(os.environ[home_env]))
    return Path.home() / ".hermes"


def command_and_args(values: dict[str, Any]) -> tuple[str, tuple[str, ...]]:
    """Resolve the concrete command + args from config (preset defaults fall back to config)."""
    command = values.get("command")
    args = values.get("args") or []
    if isinstance(command, str) and command:
        return command, tuple(args)
    preset_id = values.get("agentPreset")
    preset = presets.preset_by_id(preset_id) if isinstance(preset_id, str) else None
    if preset is None:
        raise RuntimeInvalidRequestError("agentPreset is invalid")
    resolved = presets.resolve_candidate(preset)
    if resolved is None:
        raise RuntimeInvalidRequestError("custom ACP agent requires a command")
    return resolved[0], tuple(args) or resolved[1]


def acp_capabilities() -> dict[str, bool]:
    """Static inventory capabilities for the ``acp`` runtime type.

    Reaches are refined at runtime start from the agent's own ``initialize``
    capabilities (image attachments, permission catalog from modes).
    """
    return {
        "modelCatalog": True,
        "permissionCatalog": True,
        "sessionDiscovery": True,
        "sessionSnapshot": True,
        "sessionState": True,
        "sessionNotices": True,
        "createAndStartSession": True,
        "startTurn": True,
        "steerTurn": False,
        "interruptTurn": True,
        "commands": True,
        "interactions": True,
        "attachments": True,
        "ipc": False,
    }
