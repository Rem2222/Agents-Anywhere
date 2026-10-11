from __future__ import annotations

from connector.runtimes.acp import catalogs


def test_model_selection_ids_are_stable_and_distinct() -> None:
    a = catalogs.model_selection_id("anthropic:claude-sonnet")
    b = catalogs.model_selection_id("anthropic:claude-sonnet")
    c = catalogs.model_selection_id("openai:gpt-5")
    assert a == b
    assert a != c
    assert a.startswith("sel_model_")


def test_model_catalog_from_state() -> None:
    catalog = catalogs.model_catalog_from_state(
        [
            {"modelId": "anthropic:claude-sonnet", "name": "Claude Sonnet"},
            {"modelId": "openai:gpt-5", "name": "GPT-5"},
        ],
        "anthropic:claude-sonnet",
        3,
    )
    assert catalog.runtime == "acp"
    assert [model.id for model in catalog.models] == ["anthropic:claude-sonnet", "openai:gpt-5"]
    assert catalog.models[0].selection_id == catalogs.model_selection_id("anthropic:claude-sonnet")
    assert catalog.models[0].metadata["acpModelId"] == "anthropic:claude-sonnet"


def test_custom_models_fill_empty_catalog() -> None:
    from connector.runtimes.custom_models import custom_model_items

    custom = custom_model_items("acp", [{"modelId": "custom-1", "displayName": "Custom"}])
    catalog = catalogs.model_catalog_from_state([], None, 1, custom_models=custom)
    assert [model.id for model in catalog.models] == ["custom-1"]


def test_permission_catalog_from_state() -> None:
    catalog = catalogs.permission_catalog_from_state(
        [{"id": "default", "name": "Default"}, {"id": "accept_edits", "name": "Accept edits"}],
        "default",
        2,
    )
    assert catalog is not None
    assert [permission.id for permission in catalog.permissions] == ["default", "accept_edits"]
    assert catalog.permissions[0].selection_id == catalogs.permission_selection_id("default")


def test_permission_catalog_none_without_modes() -> None:
    assert catalogs.permission_catalog_from_state([], None, 1) is None
