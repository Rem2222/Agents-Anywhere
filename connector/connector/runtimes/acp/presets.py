from __future__ import annotations

import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from connector.runtime_protocol.filesystem import canonical_path


@dataclass(frozen=True, slots=True)
class AcpCommandCandidate:
    """One way to launch an ACP agent; ``args`` only for the bare ``hermes`` binary."""

    command: str
    args: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class AcpPreset:
    """Static preset describing how to run one ACP agent as a child process."""

    id: str
    display_name: str
    candidates: tuple[AcpCommandCandidate, ...] = ()
    home_env: str | None = None
    default_home: str | None = None
    description: str = ""
    custom: bool = False


HERMES_PRESET = AcpPreset(
    id="hermes",
    display_name="Hermes Agent",
    candidates=(
        AcpCommandCandidate("hermes-acp"),
        AcpCommandCandidate("hermes", ("acp",)),
    ),
    home_env="HERMES_HOME",
    default_home="~/.hermes",
    description="Run Hermes Agent through its ACP stdio server (`hermes acp`).",
)

CUSTOM_PRESET = AcpPreset(
    id="custom",
    display_name="Custom ACP agent",
    description="Run any Agent Client Protocol agent by command line.",
    custom=True,
)

PRESETS: tuple[AcpPreset, ...] = (HERMES_PRESET, CUSTOM_PRESET)


def preset_by_id(preset_id: str) -> AcpPreset | None:
    return next((preset for preset in PRESETS if preset.id == preset_id), None)


def resolve_candidate(preset: AcpPreset) -> tuple[str, tuple[str, ...]] | None:
    """First candidate whose command is on PATH, else the first candidate (for a typed path)."""
    for candidate in preset.candidates:
        if shutil.which(candidate.command) is not None:
            return candidate.command, candidate.args
    if preset.candidates:
        candidate = preset.candidates[0]
        return candidate.command, candidate.args
    return None


def preset_enabled(preset: AcpPreset) -> bool:
    """Custom presets are always selectable; a binary preset needs its command on PATH."""
    if preset.custom:
        return True
    return any(shutil.which(candidate.command) is not None for candidate in preset.candidates)


def default_home_path(preset: AcpPreset) -> str | None:
    if preset.default_home is None:
        return None
    return canonical_path(Path(preset.default_home).expanduser())


def preset_catalog() -> dict[str, Any]:
    """Metadata the connector runtime-config dialog renders as a preset select."""
    presets: list[dict[str, Any]] = []
    for preset in PRESETS:
        resolved = resolve_candidate(preset)
        row: dict[str, Any] = {
            "id": preset.id,
            "displayName": preset.display_name,
            "description": preset.description,
            "enabled": preset_enabled(preset),
        }
        row["defaultCommand"] = resolved[0] if resolved is not None else ""
        row["defaultArgs"] = list(resolved[1]) if resolved is not None else []
        if preset.home_env:
            row["homeEnv"] = preset.home_env
        home = default_home_path(preset)
        if home is not None:
            row["defaultHome"] = home
        if not preset_enabled(preset):
            row["disabledReason"] = (
                f"{preset.candidates[0].command} not found on PATH"
                if preset.candidates
                else "Provide a command line"
            )
        presets.append(row)
    default_id = "hermes" if preset_enabled(HERMES_PRESET) else "custom"
    return {
        "configField": {
            "type": "string",
            "enum": [preset.id for preset in PRESETS],
            "default": default_id,
            "title": "ACP agent preset",
            "description": "Which ACP agent this runtime instance drives.",
        },
        "uiField": {"component": "select", "options": presets},
        "presets": presets,
        "defaultPreset": default_id,
    }
