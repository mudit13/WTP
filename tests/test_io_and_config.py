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


class _RecordingLogger:
    def __init__(self):
        self.warnings = []

    def warning(self, msg, *args, **kwargs):
        self.warnings.append(msg % args if args else msg)

    def info(self, *args, **kwargs):
        pass


def test_apply_group_map_exact_match_no_warning():
    """Regression: when full_path values in the group map and the query paths use the SAME
    prefix, matching works and no mismatch warning should fire."""
    group_map = {"/pitsec_sose26_topic8/dataset/openforensics/real/a.jpg": "Val:1"}
    log = _RecordingLogger()
    groups = io_utils.apply_group_map(
        ["/pitsec_sose26_topic8/dataset/openforensics/real/a.jpg"], group_map, logger=log)
    assert list(groups) == ["Val:1"]
    assert log.warnings == []


def test_apply_group_map_prefix_mismatch_warns_but_does_not_fix():
    """Regression for the real bug found on the server: extract_openforensics.py recorded the
    sidecar's full_path with a HOST prefix (/vol2/.../sharedDockerDir/...), while
    build_master_index.py (run inside the container) builds full_path with a CONTAINER prefix
    (/pitsec_sose26_topic8/...) for the SAME physical file - apply_group_map must still fail to
    match (no silent basename-based fix, which would risk false matches elsewhere) but MUST
    loudly warn that a same-filename, different-prefix near-miss occurred."""
    group_map = {
        "/vol2/pitsec_sose26_topic8/sharedDockerDir/dataset/openforensics/real/a.jpg": "Val:1",
    }
    query_paths = ["/pitsec_sose26_topic8/dataset/openforensics/real/a.jpg"]
    log = _RecordingLogger()
    groups = io_utils.apply_group_map(query_paths, group_map, logger=log)

    # No silent fix: falls back to the query path itself (singleton), same as any other miss.
    assert list(groups) == query_paths
    # But it MUST have warned loudly about the near-miss.
    assert len(log.warnings) == 1
    assert "PREFIX MISMATCH" in log.warnings[0]


def test_apply_group_map_no_logger_is_silent_and_safe():
    """Passing no logger (the default) must not raise, even with a prefix mismatch present."""
    group_map = {"/vol2/host/path/a.jpg": "Val:1"}
    groups = io_utils.apply_group_map(["/container/path/a.jpg"], group_map)
    assert list(groups) == ["/container/path/a.jpg"]


def test_group_lookup_map_from_df_prefers_source_path():
    """Regression for the real bug found on the server: prepare_variants.py rewrites full_path
    to a NEW derived-variant file, keeping the ORIGINAL path only in source_path. A group-aware
    sidecar is written against the ORIGINAL path, so the lookup map must resolve via
    source_path, not full_path, whenever source_path is present."""
    import pandas as pd
    df = pd.DataFrame({
        "full_path": ["/dataset/variants/aspect/openforensics_real/a.png",
                     "/dataset/variants/aspect/openforensics_real/b.png"],
        "source_path": ["/dataset/openforensics/real/a.jpg",
                        "/dataset/openforensics/real/b.jpg"],
    })
    lookup = io_utils.group_lookup_map_from_df(df)
    assert lookup["/dataset/variants/aspect/openforensics_real/a.png"] == \
        "/dataset/openforensics/real/a.jpg"
    assert lookup["/dataset/variants/aspect/openforensics_real/b.png"] == \
        "/dataset/openforensics/real/b.jpg"


def test_group_lookup_map_from_df_no_source_path_column_falls_back_to_full_path():
    import pandas as pd
    df = pd.DataFrame({"full_path": ["/a.jpg", "/b.jpg"]})
    lookup = io_utils.group_lookup_map_from_df(df)
    assert lookup == {"/a.jpg": "/a.jpg", "/b.jpg": "/b.jpg"}


def test_apply_group_map_with_lookup_end_to_end_variant_index_scenario():
    """End-to-end reproduction of the exact server scenario: a variant index (full_path points
    at a derived file; source_path points at the original), matched against a sidecar keyed by
    the ORIGINAL (source_path) path. Two coupled crops (pair via group id "Val:1") must land in
    the SAME group despite their full_path values sharing no relationship to each other at all."""
    lookup_map = {
        "/dataset/variants/aspect/openforensics_real/a.png": "/dataset/openforensics/real/a.jpg",
        "/dataset/variants/aspect/openforensics_fake/a_f.png": "/dataset/openforensics/fake/a_f.jpg",
        "/dataset/variants/aspect/openforensics_real/b.png": "/dataset/openforensics/real/b.jpg",
    }
    group_map = {
        "/dataset/openforensics/real/a.jpg": "Val:1",
        "/dataset/openforensics/fake/a_f.jpg": "Val:1",  # same source photo as a.jpg
        # b.jpg intentionally absent from group_map (no coupling for it)
    }
    variant_paths = [
        "/dataset/variants/aspect/openforensics_real/a.png",
        "/dataset/variants/aspect/openforensics_fake/a_f.png",
        "/dataset/variants/aspect/openforensics_real/b.png",
    ]
    groups = io_utils.apply_group_map_with_lookup(variant_paths, lookup_map, group_map)
    assert groups[0] == groups[1] == "Val:1"  # the coupled pair, correctly grouped
    # b.png has no group_map hit -> must fall back to ITS OWN full_path (not its source_path,
    # and not some other sentinel), so it is correctly treated as an ungrouped singleton.
    assert groups[2] == variant_paths[2]
