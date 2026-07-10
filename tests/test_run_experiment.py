"""run_experiment.py's perturbation list must be DERIVED from configs/config.yaml's
`robustness:` block (via robustness_perturb._perturbations), never a separately hand-maintained
copy - otherwise adding a perturbation to the config silently does not reach the orchestrator."""
import os

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
