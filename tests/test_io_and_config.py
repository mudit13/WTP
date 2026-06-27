"""Placeholder resolution + that the committed config.yaml is well-formed."""
import os

import pytest
import yaml

from lib import io_utils

CONFIG = os.path.join(io_utils.repo_root(), "configs", "config.yaml")


def _load_raw():
    with open(CONFIG, "r", encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def test_resolve_placeholders_strings_lists_dicts():
    env = {"WTP_ROOT": "/root", "X": "y"}
    assert io_utils.resolve_placeholders("${WTP_ROOT}/data", env) == "/root/data"
    assert io_utils.resolve_placeholders(["${X}", 1], env) == ["y", 1]
    assert io_utils.resolve_placeholders({"a": "${X}"}, env) == {"a": "y"}


def test_resolve_placeholders_missing_raises():
    with pytest.raises(KeyError):
        io_utils.resolve_placeholders("${NOPE}", {})


def test_repo_root_has_scripts():
    assert (io_utils.repo_root() / "scripts").is_dir()


def test_config_parses_with_expected_top_level_keys():
    cfg = _load_raw()
    for key in ("datasets", "attribution", "common_size", "augmentation", "seed"):
        assert key in cfg, "config.yaml missing top-level key: %s" % key
    assert cfg["datasets"], "datasets list is empty"


def test_config_datasets_have_required_fields():
    cfg = _load_raw()
    for d in cfg["datasets"]:
        for key in ("name", "dir", "label", "generator", "category"):
            assert key in d, "dataset %r missing %s" % (d.get("name"), key)
        assert d["label"] in ("real", "fake")


def test_config_has_both_classes():
    labels = {d["label"] for d in _load_raw()["datasets"]}
    assert "real" in labels and "fake" in labels
