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


def test_load_group_map_round_trip(tmp_path):
    csv_path = tmp_path / "openforensics_groups.csv"
    csv_path.write_text(
        "full_path,source_image_id\n"
        "/data/real1.jpg,Val:7\n"
        "/data/fake1.jpg,Val:7\n"
        "/data/real2.jpg,Val:8\n",
        encoding="utf-8",
    )
    gm = io_utils.load_group_map(str(csv_path))
    assert gm == {"/data/real1.jpg": "Val:7", "/data/fake1.jpg": "Val:7",
                  "/data/real2.jpg": "Val:8"}
    groups = io_utils.apply_group_map(
        ["/data/real1.jpg", "/data/fake1.jpg", "/data/unseen.jpg"], gm)
    assert list(groups) == ["Val:7", "Val:7", "/data/unseen.jpg"]


def test_load_group_map_missing_file_is_empty():
    assert io_utils.load_group_map("/does/not/exist.csv") == {}
    assert io_utils.load_group_map(None) == {}


def test_default_group_map_paths_uses_dataset_root():
    paths = io_utils.default_group_map_paths({"dataset_root": "/x/dataset"})
    assert paths == [os.path.join("/x/dataset", "openforensics", "openforensics_groups.csv")]
    assert io_utils.default_group_map_paths({}) == []
