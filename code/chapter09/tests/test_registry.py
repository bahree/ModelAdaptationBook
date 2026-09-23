"""Unit tests for the model registry (Listing 9.1)."""
from __future__ import annotations

import pytest

from chapter09.model_registry import (
    get_active_version,
    get_latest_version,
    get_version,
    list_versions,
    make_version_tag,
    promote_version,
    register_model,
    rollback,
)


@pytest.fixture()
def registry_dir(tmp_path):
    """Provide a fresh temporary directory for each test."""
    return tmp_path / "registry"


def _register_v1(registry_dir):
    return register_model(
        registry_dir=registry_dir,
        version_tag="model-v1",
        technique="sft",
        base_model="Qwen/Qwen3-4B-Instruct-2507",
        data_hash="abc123",
        checkpoint_path="chapter06/runs/sft_run1",
        eval_metrics={"overall_f1": 0.72},
        notes="First version",
    )


def _register_v2(registry_dir):
    return register_model(
        registry_dir=registry_dir,
        version_tag="model-v2",
        technique="dpo",
        base_model="Qwen/Qwen3-4B-Instruct-2507",
        data_hash="def456",
        checkpoint_path="chapter08/runs/dpo_run1",
        eval_metrics={"overall_f1": 0.58},
        notes="Second version",
    )


def test_register_model(registry_dir):
    entry = _register_v1(registry_dir)
    assert entry["version_tag"] == "model-v1"
    assert entry["status"] == "registered"
    assert entry["technique"] == "sft"
    assert entry["eval_metrics"]["overall_f1"] == 0.72

    # Verify it's in the registry
    found = get_version(registry_dir, "model-v1")
    assert found is not None
    assert found["version_tag"] == "model-v1"


def test_promote_model(registry_dir):
    _register_v1(registry_dir)
    _register_v2(registry_dir)

    # Promote v1
    promoted = promote_version(registry_dir, "model-v1")
    assert promoted["status"] == "active"

    # Promote v2 -- v1 should be retired
    promoted = promote_version(registry_dir, "model-v2")
    assert promoted["status"] == "active"

    v1 = get_version(registry_dir, "model-v1")
    assert v1["status"] == "retired"


def test_rollback(registry_dir):
    _register_v1(registry_dir)
    _register_v2(registry_dir)

    promote_version(registry_dir, "model-v1")
    promote_version(registry_dir, "model-v2")

    # v2 is active, v1 is retired
    assert get_active_version(registry_dir)["version_tag"] == "model-v2"

    # Rollback should re-activate v1
    rolled_back = rollback(registry_dir)
    assert rolled_back["version_tag"] == "model-v1"
    assert rolled_back["status"] == "active"

    # v2 should now be retired
    v2 = get_version(registry_dir, "model-v2")
    assert v2["status"] == "retired"


def test_list_versions(registry_dir):
    assert list_versions(registry_dir) == []

    _register_v1(registry_dir)
    assert len(list_versions(registry_dir)) == 1

    _register_v2(registry_dir)
    assert len(list_versions(registry_dir)) == 2


def test_duplicate_version_tag(registry_dir):
    _register_v1(registry_dir)
    with pytest.raises(ValueError, match="already exists"):
        _register_v1(registry_dir)


def test_get_latest_version(registry_dir):
    assert get_latest_version(registry_dir) is None
    _register_v1(registry_dir)
    _register_v2(registry_dir)
    latest = get_latest_version(registry_dir)
    assert latest["version_tag"] == "model-v2"


def test_get_active_version_none_when_empty(registry_dir):
    assert get_active_version(registry_dir) is None
    _register_v1(registry_dir)
    # Registered but not promoted -- no active version
    assert get_active_version(registry_dir) is None


def test_promote_nonexistent_raises(registry_dir):
    _register_v1(registry_dir)
    with pytest.raises(ValueError, match="not found"):
        promote_version(registry_dir, "nonexistent-version")


def test_rollback_no_retired_raises(registry_dir):
    _register_v1(registry_dir)
    promote_version(registry_dir, "model-v1")
    # Only one version promoted, none retired -- rollback should fail
    with pytest.raises(ValueError, match="No retired version"):
        rollback(registry_dir)


def test_make_version_tag():
    tag = make_version_tag("it-support", 3, "lora", "Qwen/Qwen3-4B-Instruct-2507")
    assert "it-support-v3-" in tag
    assert "lora" in tag
    assert "Qwen3-4B-Instruct" in tag
