from __future__ import annotations

from typing import Any

from connector.runtime_protocol import (
    RuntimeModelCatalog,
    RuntimeModelItem,
    RuntimePermissionCatalog,
    RuntimePermissionItem,
)
from connector.server.protocol import protocol_selection_id


def model_selection_id(model_id: str) -> str:
    return protocol_selection_id("acp", "model", {"model_id": model_id})


def permission_selection_id(mode_id: str) -> str:
    return protocol_selection_id("acp", "permission", {"mode_id": mode_id})


def model_catalog_from_state(
    models: list[dict[str, Any]],
    current_model_id: str | None,
    revision: int,
    *,
    custom_models: tuple[RuntimeModelItem, ...] = (),
) -> RuntimeModelCatalog:
    """Build the model catalog from a session response's ``availableModels``.

    Custom models seeded from config stay present so the selector is not empty
    before the first session; live session models win on ``model_id``.
    """
    items: list[RuntimeModelItem] = []
    seen: set[str] = set()
    for raw in models:
        if not isinstance(raw, dict):
            continue
        model_id = raw.get("modelId")
        if not isinstance(model_id, str) or not model_id or model_id in seen:
            continue
        seen.add(model_id)
        name = raw.get("name") if isinstance(raw.get("name"), str) and raw["name"] else model_id
        items.append(
            RuntimeModelItem(
                id=model_id,
                title=name,
                selection_id=model_selection_id(model_id),
                description=raw.get("description") if isinstance(raw.get("description"), str) else None,
                metadata={"acpModelId": model_id, "source": "acp.session.models"},
            )
        )
    for item in custom_models:
        if item.id in seen:
            continue
        seen.add(item.id)
        items.append(
            RuntimeModelItem(
                id=item.id,
                title=item.title,
                selection_id=model_selection_id(item.id),
                description=item.description,
                metadata={"acpModelId": item.id, "source": "runtime.config.customModels"},
            )
        )
    if current_model_id and current_model_id not in seen:
        items.append(
            RuntimeModelItem(
                id=current_model_id,
                title=current_model_id,
                selection_id=model_selection_id(current_model_id),
                metadata={"acpModelId": current_model_id, "source": "acp.session.currentModel"},
            )
        )
    return RuntimeModelCatalog(runtime="acp", revision=revision, models=tuple(items))


def permission_catalog_from_state(
    modes: list[dict[str, Any]],
    current_mode_id: str | None,
    revision: int,
) -> RuntimePermissionCatalog | None:
    """Build the permission catalog from a session response's ``availableModes``.

    Returns ``None`` when the agent reports no modes (permission catalog is then
    advertised as unsupported at runtime).
    """
    if not modes:
        return None
    items: list[RuntimePermissionItem] = []
    seen: set[str] = set()
    for raw in modes:
        if not isinstance(raw, dict):
            continue
        mode_id = raw.get("id")
        if not isinstance(mode_id, str) or not mode_id or mode_id in seen:
            continue
        seen.add(mode_id)
        name = raw.get("name") if isinstance(raw.get("name"), str) and raw["name"] else mode_id
        items.append(
            RuntimePermissionItem(
                id=mode_id,
                title=name,
                selection_id=permission_selection_id(mode_id),
                description=raw.get("description") if isinstance(raw.get("description"), str) else None,
                metadata={"acpModeId": mode_id, "source": "acp.session.modes"},
            )
        )
    if current_mode_id and current_mode_id not in seen:
        items.append(
            RuntimePermissionItem(
                id=current_mode_id,
                title=current_mode_id,
                selection_id=permission_selection_id(current_mode_id),
                metadata={"acpModeId": current_mode_id, "source": "acp.session.currentMode"},
            )
        )
    return RuntimePermissionCatalog(runtime="acp", revision=revision, permissions=tuple(items))
