"""create_data_manifest.py produces the byte-level manifest verify_handover.py checks mounted
files against - it must record sha256/size for files that exist, flag missing files instead of
crashing, and resolve group ids the same way the split/audit tooling does."""
import hashlib
import json
import logging

import pandas as pd
import pytest

import create_data_manifest as cdm
from lib import schema


class _NullLogger:
    def info(self, *a, **k):
        pass

    def warning(self, *a, **k):
        pass


def _sha256_of(text):
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def test_build_data_manifest_hashes_existing_files_and_flags_missing(tmp_path):
    real_dir = tmp_path / "dataset" / "real"
    real_dir.mkdir(parents=True)
    f1 = real_dir / "a.png"
    f1.write_text("hello-a", encoding="utf-8")
    missing_path = str(real_dir / "does_not_exist.png")

    index_csv = tmp_path / "index.csv"
    pd.DataFrame({
        schema.PATH: [str(f1), missing_path],
        schema.LABEL: ["real", "fake"],
        schema.GENERATOR: ["London-DB", "SD1.5"],
        schema.DATASET: ["londondb", "sd15_txt2img"],
    }).to_csv(index_csv, index=False)

    config = {"dataset_root": str(tmp_path / "dataset")}
    manifest, summary = cdm.build_data_manifest(str(index_csv), config, None, _NullLogger())

    assert len(manifest) == 2
    row_a = manifest[manifest["path"] == str(f1)].iloc[0]
    assert row_a["sha256"] == _sha256_of("hello-a")
    assert row_a["missing"] == False  # noqa: E712
    assert row_a["relative_path"] == "real/a.png"

    row_missing = manifest[manifest["path"] == missing_path].iloc[0]
    assert row_missing["missing"] == True  # noqa: E712
    assert pd.isna(row_missing["sha256"])

    assert summary["n_rows"] == 2
    assert summary["n_missing"] == 1
    assert summary["total_bytes"] == len("hello-a")


def test_build_data_manifest_resolves_group_ids_via_group_map(tmp_path, monkeypatch):
    d = tmp_path / "dataset"
    d.mkdir()
    real_img = d / "real.jpg"
    real_img.write_text("real", encoding="utf-8")
    fake_img = d / "fake.png"
    fake_img.write_text("fake", encoding="utf-8")
    unrelated_img = d / "solo.png"
    unrelated_img.write_text("solo", encoding="utf-8")

    group_csv = tmp_path / "groups.csv"
    group_csv.write_text(
        "full_path,source_image_id\n"
        f"{real_img},Val:1\n"
        f"{fake_img},Val:1\n",
        encoding="utf-8")

    index_csv = tmp_path / "index.csv"
    pd.DataFrame({
        schema.PATH: [str(real_img), str(fake_img), str(unrelated_img)],
        schema.LABEL: ["real", "fake", "real"],
        schema.GENERATOR: ["OpenForensics", "OpenForensics-fake", "FFHQ"],
        schema.DATASET: ["openforensics_real", "openforensics_fake", "dffd_ffhq"],
    }).to_csv(index_csv, index=False)

    config = {"dataset_root": str(d)}
    manifest, _ = cdm.build_data_manifest(str(index_csv), config, [str(group_csv)], _NullLogger())

    grouped = dict(zip(manifest["path"], manifest["group_id"]))
    assert grouped[str(real_img)] == "Val:1"
    assert grouped[str(fake_img)] == "Val:1"
    assert grouped[str(unrelated_img)] == ""  # no coupling recorded -> empty, not its own path


def test_build_checkpoint_manifest_hashes_and_flags_missing(tmp_path):
    ckpt = tmp_path / "clip_linear.pt"
    ckpt.write_text("weights", encoding="utf-8")
    missing = str(tmp_path / "nope.pt")

    manifest, summary = cdm.build_checkpoint_manifest([str(ckpt), missing], _NullLogger())

    assert summary["n_rows"] == 2
    assert summary["n_missing"] == 1
    row = manifest[manifest["path"] == str(ckpt)].iloc[0]
    assert row["sha256"] == _sha256_of("weights")


def test_main_writes_manifest_csv_and_summary_json(tmp_path):
    from types import SimpleNamespace
    ckpt = tmp_path / "clip_linear.pt"
    ckpt.write_text("weights", encoding="utf-8")
    out = tmp_path / "checkpoint_manifest.csv"

    args = SimpleNamespace(mode="checkpoints", config=None, index=None, group_map=None,
                           paths=[str(ckpt)], out=str(out))
    cdm.main(args)

    assert out.exists()
    summary_path = tmp_path / "checkpoint_manifest_summary.json"
    assert summary_path.exists()
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    assert summary["kind"] == "checkpoints"
    assert summary["n_rows"] == 1
    assert summary["n_missing"] == 0
