"""De-Fake-patched/blipmodels/blip.py must resolve its med_config.json relative to its own file,
not a hardcoded /pitsec_sose26_topic8 path, so the vendored package works from any checkout
location (a fresh Magdeburg container clone, a laptop, CI, ...).

The full model-construction smoke test needs torch/transformers/timm (not installed in this
CPU-only dev/CI environment) and is skipped here if they are unavailable; it still runs on any
environment - including the Magdeburg container - that has the DE-FAKE venv's dependencies.
"""
import os

import pytest

from lib import io_utils

DEFAKE_DIR = os.path.join(io_utils.repo_root(), "De-Fake-patched")
MED_CONFIG = os.path.join(DEFAKE_DIR, "blipmodels", "blipconfig", "med_config.json")


def _torch_stack_available():
    try:
        import torch  # noqa: F401
        import transformers  # noqa: F401
        import timm  # noqa: F401
    except ImportError:
        return False
    return True


def test_med_config_json_exists_at_the_expected_relative_location():
    assert os.path.exists(MED_CONFIG), (
        "blipconfig/med_config.json missing next to blip.py; the default med_config path in "
        "blip.py resolves relative to this file and would break if the file moved.")


def test_default_med_config_is_not_hardcoded_to_a_container_path():
    with open(os.path.join(DEFAKE_DIR, "blipmodels", "blip.py"), "r", encoding="utf-8") as fh:
        source = fh.read()
    assert "med_config = '/pitsec_sose26_topic8" not in source, (
        "blip.py must not hardcode the container path as a med_config default; it should "
        "resolve relative to __file__ so the vendored package works from any checkout location.")
    assert "_DEFAULT_MED_CONFIG" in source


@pytest.mark.skipif(not _torch_stack_available(),
                    reason="torch/transformers/timm not installed in this environment")
def test_blip_decoder_constructs_offline_with_default_med_config(tmp_path, monkeypatch):
    """Import/construction smoke test (server-equivalent of the DE-FAKE `--test` inference
    check): BLIP_Decoder() must build successfully with NO pretrained checkpoint and from an
    unrelated working directory, proving med_config resolution does not depend on cwd."""
    import sys
    sys.path.insert(0, DEFAKE_DIR)
    monkeypatch.chdir(tmp_path)  # prove resolution is cwd-independent

    from blipmodels.blip import BLIP_Decoder  # noqa: E402

    model = BLIP_Decoder(image_size=224, vit="base")  # no `pretrained=` -> no network access
    assert model is not None
