"""run_experiment.py's perturbation list must be DERIVED from configs/config.yaml's
`robustness:` block (via robustness_perturb._perturbations), never a separately hand-maintained
copy - otherwise adding a perturbation to the config silently does not reach the orchestrator."""
import json
import os
from types import SimpleNamespace

import pytest
import yaml

import run_experiment as re
from lib import io_utils

CONFIG = os.path.join(io_utils.repo_root(), "configs", "config.yaml")


def test_perturbation_names_match_config_robustness_block():
    with open(CONFIG, "r", encoding="utf-8") as fh:
        raw = yaml.safe_load(fh)
    rob = raw["robustness"]
    expected = (
        ["jpeg%d" % q for q in rob["jpeg_quality"]]
        + ["blur%g" % s for s in rob["gaussian_blur_sigma"]]
        + ["resize%g" % f for f in rob["resize_factors"]]
        + ["sharpen%g" % a for a in rob["sharpen"]]
    )
    assert re._perturbation_names(CONFIG) == expected


def test_perturbation_names_grows_when_config_gains_an_entry(tmp_path):
    """Adding a perturbation value to the config must add a name, without editing any code."""
    with open(CONFIG, "r", encoding="utf-8") as fh:
        raw = yaml.safe_load(fh)
    before = re._perturbation_names(CONFIG)

    raw["robustness"]["sharpen"] = list(raw["robustness"]["sharpen"]) + [2.0]
    tmp_cfg = tmp_path / "config_with_extra_sharpen.yaml"
    with open(tmp_cfg, "w", encoding="utf-8") as fh:
        yaml.safe_dump(raw, fh)

    after = re._perturbation_names(str(tmp_cfg))
    assert len(after) == len(before) + 1
    assert "sharpen2" in after and "sharpen2" not in before


def test_perturbation_names_works_without_env_placeholders(monkeypatch):
    """Must NOT require configs/paths.env / WTP_ROOT to be set - --dry_run should still work on
    a fresh checkout with no server environment configured."""
    monkeypatch.delenv("WTP_ROOT", raising=False)
    names = re._perturbation_names(CONFIG)
    assert len(names) > 0


def _ctx(tmp_path):
    args = SimpleNamespace(
        python="python", config=CONFIG, variant="aspect", jpeg_aug="on", device="cuda",
        run_id="test-eightway", dry_run=True, results_dir=str(tmp_path),
        dataset_dir="/dataset", captions_csv="/dataset/captions.csv",
    )
    return re.Ctx(args)


def test_declared_attribution_plan_is_eight_fake_classes():
    classes = re._declared_fake_classes(CONFIG)
    assert len(classes) == 8
    assert "SD1.5-img2img" in classes
    assert "OpenForensics-fake" not in classes


def test_attribution_stage_plans_primary_auxiliary_and_strict_logo(tmp_path):
    steps = re.stage_attribution(_ctx(tmp_path))
    descriptions = [step["desc"] for step in steps]
    assert any("primary 8-way" in d for d in descriptions)
    assert any("auxiliary 9-way" in d for d in descriptions)
    logo = next(step for step in steps if step["desc"].startswith("LOGO"))
    assert "--class_mode" in logo["cmd"]
    assert "fake_only" in logo["cmd"]
    assert "--targets" not in logo["cmd"]  # script default is all configured fake classes


def test_cascade_stage_uses_shared_dct_test_predictions(tmp_path):
    c = _ctx(tmp_path)
    steps = re.stage_cascade(c)
    assert len(steps) == 2
    assert c.test_index in steps[0]["cmd"]
    assert "%sdct_per_image.csv" % c.dct_svm_out in steps[1]["cmd"]


def test_ffhq_ablation_trains_matched_source_specific_heads(tmp_path):
    c = _ctx(tmp_path)
    steps = re.stage_ffhq_ablation(c)
    assert len(steps) == 5
    with_train = steps[0]["cmd"]
    without_train = steps[2]["cmd"]
    assert "FFHQ" in with_train
    assert "FFHQ" not in without_train
    assert all(fake in with_train and fake in without_train for fake in c.fake_classes)
    assert steps[-1]["cmd"][1].endswith("compare_ffhq_ablation.py")


def _fake_config(tmp_path, name="config.yaml"):
    """A minimal but valid config.yaml clone so _sha256 has a real file to hash."""
    cfg_path = tmp_path / name
    cfg_path.write_text("attribution:\n  fake_generators: []\n  real_generators: []\n",
                        encoding="utf-8")
    return str(cfg_path)


def _run_dir_ctx(tmp_path, cfg_path, run_id="my-run"):
    return SimpleNamespace(results=str(tmp_path / "run"), cfg=cfg_path, run_id=run_id)


def test_prepare_run_dir_creates_manifest_with_core_commit(tmp_path, monkeypatch):
    monkeypatch.setattr(re, "_current_git_commit", lambda: "commit-aaa")
    cfg_path = _fake_config(tmp_path)
    c = _run_dir_ctx(tmp_path, cfg_path)
    c.variant, c.jpeg_aug, c.device = "aspect", "on", "cpu"
    args = SimpleNamespace(resume=False)

    re._prepare_run_dir(c, args)

    manifest = json.loads((tmp_path / "run" / "run_manifest.json").read_text(encoding="utf-8"))
    assert manifest["core_commit"] == "commit-aaa"
    assert manifest["git_commit"] == "commit-aaa"
    assert manifest["analysis_history"] == []


def test_resume_with_same_commit_succeeds_silently(tmp_path, monkeypatch):
    monkeypatch.setattr(re, "_current_git_commit", lambda: "commit-aaa")
    cfg_path = _fake_config(tmp_path)
    c = _run_dir_ctx(tmp_path, cfg_path)
    c.variant, c.jpeg_aug, c.device = "aspect", "on", "cpu"
    re._prepare_run_dir(c, SimpleNamespace(resume=False))

    resume_args = SimpleNamespace(resume=True, allow_code_drift=False, code_drift_reason=None,
                                  stages="aggregate")
    re._prepare_run_dir(c, resume_args)  # must not raise

    manifest = json.loads((tmp_path / "run" / "run_manifest.json").read_text(encoding="utf-8"))
    assert manifest.get("analysis_history", []) == []


def test_resume_with_different_code_refused_by_default(tmp_path, monkeypatch):
    monkeypatch.setattr(re, "_current_git_commit", lambda: "commit-aaa")
    cfg_path = _fake_config(tmp_path)
    c = _run_dir_ctx(tmp_path, cfg_path)
    c.variant, c.jpeg_aug, c.device = "aspect", "on", "cpu"
    re._prepare_run_dir(c, SimpleNamespace(resume=False))

    monkeypatch.setattr(re, "_current_git_commit", lambda: "commit-bbb")
    resume_args = SimpleNamespace(resume=True, allow_code_drift=False, code_drift_reason=None,
                                  stages="aggregate")
    with pytest.raises(SystemExit, match="core commit"):
        re._prepare_run_dir(c, resume_args)


def test_resume_with_code_drift_requires_reason(tmp_path, monkeypatch):
    monkeypatch.setattr(re, "_current_git_commit", lambda: "commit-aaa")
    cfg_path = _fake_config(tmp_path)
    c = _run_dir_ctx(tmp_path, cfg_path)
    c.variant, c.jpeg_aug, c.device = "aspect", "on", "cpu"
    re._prepare_run_dir(c, SimpleNamespace(resume=False))

    monkeypatch.setattr(re, "_current_git_commit", lambda: "commit-bbb")
    resume_args = SimpleNamespace(resume=True, allow_code_drift=True, code_drift_reason=None,
                                  stages="aggregate")
    with pytest.raises(SystemExit, match="code_drift_reason"):
        re._prepare_run_dir(c, resume_args)


def test_resume_with_explicit_code_drift_reason_appends_history_without_changing_core(
        tmp_path, monkeypatch):
    monkeypatch.setattr(re, "_current_git_commit", lambda: "commit-aaa")
    cfg_path = _fake_config(tmp_path)
    c = _run_dir_ctx(tmp_path, cfg_path)
    c.variant, c.jpeg_aug, c.device = "aspect", "on", "cpu"
    re._prepare_run_dir(c, SimpleNamespace(resume=False))

    monkeypatch.setattr(re, "_current_git_commit", lambda: "commit-bbb")
    resume_args = SimpleNamespace(resume=True, allow_code_drift=True,
                                  code_drift_reason="reporting-only aggregate re-run",
                                  stages="aggregate")
    re._prepare_run_dir(c, resume_args)  # must not raise

    manifest = json.loads((tmp_path / "run" / "run_manifest.json").read_text(encoding="utf-8"))
    assert manifest["core_commit"] == "commit-aaa"  # immutable
    assert len(manifest["analysis_history"]) == 1
    entry = manifest["analysis_history"][0]
    assert entry["commit"] == "commit-bbb"
    assert entry["reason"] == "reporting-only aggregate re-run"


def test_resume_with_changed_config_still_refused_regardless_of_code_drift_flags(
        tmp_path, monkeypatch):
    monkeypatch.setattr(re, "_current_git_commit", lambda: "commit-aaa")
    cfg_path = _fake_config(tmp_path)
    c = _run_dir_ctx(tmp_path, cfg_path)
    c.variant, c.jpeg_aug, c.device = "aspect", "on", "cpu"
    re._prepare_run_dir(c, SimpleNamespace(resume=False))

    # Mutate the config on disk -> hash changes.
    with open(cfg_path, "a", encoding="utf-8") as fh:
        fh.write("extra_key: true\n")
    resume_args = SimpleNamespace(resume=True, allow_code_drift=True,
                                  code_drift_reason="anything", stages="aggregate")
    with pytest.raises(SystemExit, match="config hash"):
        re._prepare_run_dir(c, resume_args)


def test_rigor_is_default_and_enforces_leakage_gates_before_aggregate(tmp_path):
    c = _ctx(tmp_path)
    steps = re.stage_rigor(c)
    assert len(steps) == 9
    audit = steps[0]["cmd"]
    assert audit[1].endswith("audit_split_leakage.py")
    assert "--fail_on_exact" in audit
    assert "--fail_on_group_straddle" in audit
    assert "rigor" in re.DEFAULT_STAGES
    assert re.DEFAULT_STAGES.index("rigor") < re.DEFAULT_STAGES.index("aggregate")
