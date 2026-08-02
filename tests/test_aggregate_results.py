import json
from types import SimpleNamespace

import pytest

import aggregate_results


def _write(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data), encoding="utf-8")


def _write_manifest(tmp_path, **overrides):
    manifest = {
        "run_id": "test-run",
        "core_commit": "abc123def456",
        "config_sha256": "deadbeef" * 4,
    }
    manifest.update(overrides)
    _write(tmp_path / "run_manifest.json", manifest)


def test_aggregate_rejects_directory_without_run_manifest(tmp_path):
    """Refuse flat/unscoped aggregation (e.g. the bare `results/` base dir) that could mix
    metrics from more than one run."""
    with pytest.raises(SystemExit, match="run_manifest.json"):
        aggregate_results.main(SimpleNamespace(
            results_dir=str(tmp_path), out=str(tmp_path / "REPORT_SUMMARY.md")))


def test_aggregate_writes_run_provenance_header(tmp_path):
    _write_manifest(tmp_path)
    out = tmp_path / "REPORT_SUMMARY.md"
    aggregate_results.main(SimpleNamespace(results_dir=str(tmp_path), out=str(out)))
    text = out.read_text(encoding="utf-8")
    assert "test-run" in text
    assert "abc123def456"[:12] in text


def test_aggregate_notes_analysis_history_without_changing_core_commit(tmp_path):
    _write_manifest(tmp_path, analysis_history=[
        {"at": "2026-08-02T00:00:00", "commit": "zzz999", "reason": "reporting fix"}])
    out = tmp_path / "REPORT_SUMMARY.md"
    aggregate_results.main(SimpleNamespace(results_dir=str(tmp_path), out=str(out)))
    text = out.read_text(encoding="utf-8")
    assert "1 post-creation analysis_history" in text
    assert "abc123def456"[:12] in text  # core_commit unchanged in the header


def test_aggregate_includes_rigor_and_suppresses_meaningless_oos_accuracy(tmp_path):
    _write_manifest(tmp_path)
    metric = {
        "top1_accuracy": 0.8,
        "macro_f1": 0.8,
        "balanced_accuracy": 0.8,
        "cohen_kappa": 0.75,
    }
    _write(tmp_path / "attr" / "attribution_metrics.json", {
        "in_set": metric,
        "out_of_set": dict(metric, top1_accuracy=0.0, cohen_kappa=0.0),
        "all_fakes": metric,
    })
    _write(tmp_path / "ci_attr_8way.json", {
        "overall": {
            "top1_accuracy": {"point": 0.8, "lo": 0.7, "hi": 0.9},
            "cohen_kappa": {"point": 0.75, "lo": 0.6, "hi": 0.85},
            "n": 20,
        },
    })
    _write(tmp_path / "seed_sweep_8way.json", {
        "seeds": [42, 43],
        "top1_accuracy": {"mean": 0.78, "std": 0.03},
        "balanced_accuracy": {"mean": 0.77, "std": 0.04},
        "cohen_kappa": {"mean": 0.74, "std": 0.05},
    })
    _write(tmp_path / "defake_vs_dct_significance.json", {
        "n_shared": 20,
        "mcnemar": {"p_value": 0.5},
        "auroc": {"diff_defake_minus_dct": {
            "point": -0.04, "lo": -0.1, "hi": 0.02, "p_bootstrap": 0.2}},
        "balanced_accuracy": {"diff_defake_minus_dct": {
            "point": -0.08, "lo": -0.14, "hi": -0.02, "p_bootstrap": 0.01}},
    })
    _write(tmp_path / "leakage_audit_8way.json", {
        "group_straddle": {"n_groups_straddling": 0, "n_groups_checked": 6},
        "exact_cross_split_duplicates": {"count": 0},
        "near_cross_split_duplicates": {"count": 12},
    })
    out = tmp_path / "REPORT_SUMMARY.md"
    aggregate_results.main(SimpleNamespace(results_dir=str(tmp_path), out=str(out)))
    text = out.read_text(encoding="utf-8")

    assert "[in_set]" in text
    assert "[out_of_set]" not in text
    assert "[all_fakes]" not in text
    assert "Bootstrap 95% confidence intervals" in text
    assert "Attribution seed sensitivity" in text
    assert "Paired DCT vs pretrained DE-FAKE" in text
    assert "Split-integrity audits" in text
