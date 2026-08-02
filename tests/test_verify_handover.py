"""verify_handover.py is the last, read-only gate before a run directory is handed to a
supervisor: it must catch missing artifacts, failed leakage gates, wrong headline numbers, and
stale flat summaries without re-running anything."""
import json

import pytest

import verify_handover as vh


def _write(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    if isinstance(data, str):
        path.write_text(data, encoding="utf-8")
    else:
        path.write_text(json.dumps(data), encoding="utf-8")


def _minimal_run(tmp_path, run_id="test-run", core_commit="abc123", straddle=0, exact=0):
    _write(tmp_path / "run_manifest.json", {
        "run_id": run_id, "core_commit": core_commit, "config_sha256": "deadbeef" * 4,
        "analysis_history": [],
    })
    _write(tmp_path / "REPORT_SUMMARY.md", "# summary")
    _write(tmp_path / "leakage_audit_8way.json", {
        "group_straddle": {"n_groups_straddling": straddle, "n_groups_checked": 6},
        "exact_cross_split_duplicates": {"count": exact},
    })
    for name in ("seed_sweep_8way.json", "seed_sweep_8way_image_only.json",
                "ci_attr_8way.json", "ci_attr_8way_image_only.json", "ci_attr_9way.json",
                "ci_dct_detection.json", "ci_defake_detection.json",
                "ci_cascade_conditional.json", "ci_cascade_end_to_end.json",
                "defake_vs_dct_significance.json"):
        _write(tmp_path / name, {})
    return tmp_path


def test_passes_on_a_complete_well_formed_run(tmp_path):
    _minimal_run(tmp_path)
    v = vh.Verifier(str(tmp_path), run_id="test-run")
    v.run()
    assert v.errors == []


def test_fails_when_run_manifest_missing():
    v = vh.Verifier("/does/not/exist", run_id="test-run")
    v.run()
    assert any("run_manifest.json" in e for e in v.errors)


def test_fails_when_run_id_does_not_match(tmp_path):
    _minimal_run(tmp_path, run_id="other-run")
    v = vh.Verifier(str(tmp_path), run_id="test-run")
    v.run()
    assert any("does not match expected" in e for e in v.errors)


def test_warns_but_does_not_fail_on_analysis_history(tmp_path):
    _minimal_run(tmp_path)
    manifest_path = tmp_path / "run_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["analysis_history"] = [{"at": "x", "commit": "def456", "reason": "reporting fix"}]
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    v = vh.Verifier(str(tmp_path), run_id="test-run")
    v.run()
    assert v.errors == []
    assert any("analysis_history" in w for w in v.warnings)


def test_fails_on_missing_required_artifact(tmp_path):
    _minimal_run(tmp_path)
    (tmp_path / "ci_attr_8way.json").unlink()
    v = vh.Verifier(str(tmp_path), run_id="test-run")
    v.run()
    assert any("ci_attr_8way.json" in e for e in v.errors)


def test_fails_on_nonzero_group_straddle_or_exact_duplicates(tmp_path):
    _minimal_run(tmp_path, straddle=2, exact=1)
    v = vh.Verifier(str(tmp_path), run_id="test-run")
    v.run()
    assert any("n_groups_straddling" in e for e in v.errors)
    assert any("exact_cross_split_duplicates" in e for e in v.errors)


def test_headline_spot_checks_pass_for_matching_authoritative_numbers(tmp_path):
    _minimal_run(tmp_path, run_id="2026-08-01_eightway_v1")
    _write(tmp_path / "dct_svm_aspect" / "metrics.json",
          {"test": {"balanced_accuracy": 0.608}})
    _write(tmp_path / "attr_eval_8way_aspect" / "attribution_metrics.json",
          {"in_set": {"top1_accuracy": 0.873}})
    _write(tmp_path / "seed_sweep_8way.json", {"top1_accuracy": {"mean": 0.814}})

    v = vh.Verifier(str(tmp_path), run_id="2026-08-01_eightway_v1", variant="aspect")
    v.run()
    assert v.errors == []


def test_headline_spot_check_fails_for_mismatched_numbers(tmp_path):
    _minimal_run(tmp_path, run_id="2026-08-01_eightway_v1")
    _write(tmp_path / "dct_svm_aspect" / "metrics.json",
          {"test": {"balanced_accuracy": 0.400}})  # wrong
    _write(tmp_path / "attr_eval_8way_aspect" / "attribution_metrics.json",
          {"in_set": {"top1_accuracy": 0.873}})
    _write(tmp_path / "seed_sweep_8way.json", {"top1_accuracy": {"mean": 0.814}})

    v = vh.Verifier(str(tmp_path), run_id="2026-08-01_eightway_v1", variant="aspect")
    v.run()
    assert any("Headline spot check failed" in e for e in v.errors)


def test_fails_on_stale_flat_report_summary_next_to_run_dir(tmp_path):
    run_dir = tmp_path / "2026-08-01_eightway_v1"
    _minimal_run(run_dir, run_id="2026-08-01_eightway_v1")
    _write(tmp_path / "REPORT_SUMMARY.md", "# stale flat summary")

    v = vh.Verifier(str(run_dir), run_id="2026-08-01_eightway_v1")
    v.run()
    assert any("flat, non-run-scoped" in e for e in v.errors)


def test_fails_when_only_legacy_leakage_audit_present(tmp_path):
    _minimal_run(tmp_path)
    (tmp_path / "leakage_audit_8way.json").unlink()
    _write(tmp_path / "leakage_audit_8way_full.json", {})
    v = vh.Verifier(str(tmp_path), run_id="test-run")
    v.run()
    assert any("historical leakage_audit_8way_full" in e for e in v.errors)


def _args(**overrides):
    from types import SimpleNamespace
    defaults = dict(results_dir=None, run_id="test-run", variant="aspect", jpeg_aug="on",
                    tolerance=0.01, require_data_manifest=False,
                    require_checkpoint_manifest=False, verify_mounts=False)
    defaults.update(overrides)
    return SimpleNamespace(**defaults)


def test_main_exits_nonzero_on_failure(tmp_path, capsys):
    with pytest.raises(SystemExit):
        vh.main(_args(results_dir=str(tmp_path)))


def test_main_succeeds_on_complete_run(tmp_path):
    _minimal_run(tmp_path)
    vh.main(_args(results_dir=str(tmp_path)))  # must not raise


def test_data_manifest_absent_is_a_warning_not_an_error_by_default(tmp_path):
    _minimal_run(tmp_path)
    v = vh.Verifier(str(tmp_path), run_id="test-run")
    v.run()
    assert v.errors == []
    assert any("data_manifest.csv" in w for w in v.warnings)


def test_data_manifest_absent_is_an_error_when_required(tmp_path):
    _minimal_run(tmp_path)
    v = vh.Verifier(str(tmp_path), run_id="test-run", require_data_manifest=True)
    v.run()
    assert any("data_manifest.csv" in e for e in v.errors)


def test_data_manifest_rehash_passes_when_files_match(tmp_path):
    _minimal_run(tmp_path)
    img = tmp_path / "a.png"
    img.write_bytes(b"hello")
    import hashlib
    sha = hashlib.sha256(b"hello").hexdigest()
    (tmp_path / "data_manifest.csv").write_text(
        "path,sha256,missing\n%s,%s,False\n" % (img, sha), encoding="utf-8")

    v = vh.Verifier(str(tmp_path), run_id="test-run")
    v.run()
    assert v.errors == []
    assert any("0 mismatches" in w for w in v.warnings)


def test_data_manifest_rehash_fails_on_sha256_mismatch(tmp_path):
    _minimal_run(tmp_path)
    img = tmp_path / "a.png"
    img.write_bytes(b"hello")
    (tmp_path / "data_manifest.csv").write_text(
        "path,sha256,missing\n%s,%s,False\n" % (img, "deadbeef" * 8), encoding="utf-8")

    v = vh.Verifier(str(tmp_path), run_id="test-run")
    v.run()
    assert any("do not match their recorded sha256" in e for e in v.errors)


def test_data_manifest_unreachable_file_warns_by_default_but_fails_with_verify_mounts(tmp_path):
    _minimal_run(tmp_path)
    (tmp_path / "data_manifest.csv").write_text(
        "path,sha256,missing\n%s,%s,False\n" % (
            str(tmp_path / "does_not_exist.png"), "a" * 64), encoding="utf-8")

    v = vh.Verifier(str(tmp_path), run_id="test-run")
    v.run()
    assert v.errors == []
    assert any("could not be reached" in w for w in v.warnings)

    v_strict = vh.Verifier(str(tmp_path), run_id="test-run", verify_mounts=True)
    v_strict.run()
    assert any("could not be reached" in e for e in v_strict.errors)
