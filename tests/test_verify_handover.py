import argparse
import csv
import hashlib
import json
import sys
from pathlib import Path

import pytest
import yaml


SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import verify_handover as vh  # noqa: E402


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


def base_run(tmp_path, metric=0.75, commit="a" * 40, config_hash="b" * 64):
    run = tmp_path / "results" / "run-one"
    run.mkdir(parents=True)
    write_json(
        run / "run_manifest.json",
        {
            "run_id": "run-one",
            "core_commit": commit,
            "config_sha256": config_hash,
        },
    )
    write_json(
        run / "leakage_audit_8way.json",
        {
            "group_straddle": {"n_groups_straddling": 0},
            "exact_cross_split_duplicates": {"count": 0},
        },
    )
    write_json(run / "metrics.json", {"test": {"balanced_accuracy": metric}})
    return run


def release_data(run, expected=0.75, commit="a" * 40, config_hash="b" * 64):
    return {
        "schema_version": 1,
        "release_id": "release-one",
        "run_id": "run-one",
        "results_dir": str(run),
        "git_commit": commit,
        "config_sha256": config_hash,
        "variant": "aspect",
        "jpeg_aug": "on",
        "tolerance": 1e-9,
        "require_data_manifest": False,
        "require_checkpoint_manifest": False,
        "required_artifacts": [
            "run_manifest.json",
            "leakage_audit_8way.json",
            "metrics.json",
        ],
        "headline_checks": [
            {
                "file": "metrics.json",
                "keys": ["test", "balanced_accuracy"],
                "expected": expected,
            }
        ],
    }


def namespace(release=None, results_dir=None, verify_mounts=False):
    return argparse.Namespace(
        release=str(release) if release else None,
        results_dir=str(results_dir) if results_dir else None,
        run_id=None,
        variant=None,
        jpeg_aug=None,
        tolerance=None,
        require_data_manifest=None,
        require_checkpoint_manifest=None,
        verify_mounts=verify_mounts,
    )


def verifier_from_settings(settings):
    return vh.Verifier(
        results_dir=settings["results_dir"],
        run_id=settings["run_id"],
        variant=settings["variant"],
        jpeg_aug=settings["jpeg_aug"],
        tolerance=settings["tolerance"],
        require_data_manifest=settings["require_data_manifest"],
        require_checkpoint_manifest=settings["require_checkpoint_manifest"],
        verify_mounts=settings["verify_mounts"],
        required_artifacts=settings["required_artifacts"],
        headline_checks=settings["headline_checks"],
        expected_commit=settings["expected_commit"],
        expected_config_sha256=settings["expected_config_sha256"],
        release_id=settings["release_id"],
    )


def test_release_manifest_drives_spot_checks(tmp_path):
    run = base_run(tmp_path)
    release = tmp_path / "release.yaml"
    release.write_text(yaml.safe_dump(release_data(run)), encoding="utf-8")

    settings = vh.build_settings(namespace(release=release))
    verifier = verifier_from_settings(settings)
    manifest = verifier.run()

    assert manifest["run_id"] == "run-one"
    assert verifier.errors == []
    assert vh.HEADLINE_SPOT_CHECKS == {}


def test_placeholder_release_is_rejected(tmp_path):
    release = release_data(base_run(tmp_path))
    release["git_commit"] = "REPLACE_WITH_40_CHARACTER_COMMIT_SHA"

    with pytest.raises(ValueError, match="placeholder"):
        vh.validate_release_config(release)


def test_headline_mismatch_fails(tmp_path):
    run = base_run(tmp_path, metric=0.70)
    release = release_data(run, expected=0.75)
    verifier = vh.Verifier(
        run,
        run_id=release["run_id"],
        required_artifacts=release["required_artifacts"],
        headline_checks=release["headline_checks"],
        expected_commit=release["git_commit"],
        expected_config_sha256=release["config_sha256"],
        tolerance=release["tolerance"],
    )

    verifier.run()
    assert any("Headline check" in error for error in verifier.errors)


def test_release_commit_mismatch_fails(tmp_path):
    run = base_run(tmp_path, commit="c" * 40)
    release = release_data(run, commit="a" * 40)
    verifier = vh.Verifier(
        run,
        run_id=release["run_id"],
        required_artifacts=release["required_artifacts"],
        headline_checks=release["headline_checks"],
        expected_commit=release["git_commit"],
        expected_config_sha256=release["config_sha256"],
    )

    verifier.run()
    assert any("does not match release git_commit" in error for error in verifier.errors)


def test_required_manifest_rehashes_files(tmp_path):
    run = base_run(tmp_path)
    asset = tmp_path / "asset.bin"
    asset.write_bytes(b"verified")
    digest = hashlib.sha256(asset.read_bytes()).hexdigest()

    with open(run / "data_manifest.csv", "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["path", "sha256", "missing"])
        writer.writeheader()
        writer.writerow({"path": str(asset), "sha256": digest, "missing": "false"})

    verifier = vh.Verifier(
        run,
        run_id="run-one",
        required_artifacts=["run_manifest.json", "leakage_audit_8way.json"],
        require_data_manifest=True,
        verify_mounts=True,
    )
    verifier.run()
    assert verifier.errors == []
    assert any("re-hashed 1" in warning for warning in verifier.warnings)

    asset.write_bytes(b"changed")
    verifier = vh.Verifier(
        run,
        run_id="run-one",
        required_artifacts=["run_manifest.json", "leakage_audit_8way.json"],
        require_data_manifest=True,
        verify_mounts=True,
    )
    verifier.run()
    assert any("do not match" in error for error in verifier.errors)


def test_direct_mode_has_no_hardcoded_run_metrics(tmp_path):
    run = base_run(tmp_path, metric=0.01)
    settings = vh.build_settings(namespace(results_dir=run))
    assert settings["headline_checks"] == []
    assert settings["expected_commit"] is None
