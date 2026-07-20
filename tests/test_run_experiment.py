"""run_experiment.py's perturbation list must be DERIVED from configs/config.yaml's
`robustness:` block (via robustness_perturb._perturbations), never a separately hand-maintained
copy - otherwise adding a perturbation to the config silently does not reach the orchestrator."""
import os
from types import SimpleNamespace

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
