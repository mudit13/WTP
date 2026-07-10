"""predict_defake_head.py must remap captions via source_path for perturbed indices - otherwise
every perturbed row's caption lookup misses (keyed by the NEW perturbed full_path, which never
matches anything in a captions CSV built from the ORIGINAL images) and silently falls back to
"", conflating the perturbation's effect with a caption-mismatch artifact in the measured
attribution robustness (label-flip-rate / confidence-drop)."""
import pandas as pd

import predict_defake_head as pdh
from lib import schema


class _StubLogger:
    def info(self, *args, **kwargs):
        pass

    def warning(self, *args, **kwargs):
        pass


def test_remaps_captions_via_source_path_for_perturbed_index(tmp_path):
    captions_csv = tmp_path / "defake_predictions_aspect.csv"
    pd.DataFrame({
        schema.PATH: ["/orig/a.png", "/orig/b.png"],
        schema.BLIP_CAPTION: ["a photo of a smiling person", "a photo of a serious person"],
    }).to_csv(captions_csv, index=False)

    # Perturbation-style index: full_path is the NEW (perturbed) path; source_path is the
    # ORIGINAL image it was perturbed from - exactly robustness_perturb.py's generate() schema.
    index_csv = tmp_path / "index_jpeg30.csv"
    pd.DataFrame({
        schema.PATH: ["/robust/jpeg30/a.png", "/robust/jpeg30/b.png"],
        "source_path": ["/orig/a.png", "/orig/b.png"],
    }).to_csv(index_csv, index=False)

    out = pdh._resolve_captions_csv(str(index_csv), str(captions_csv), str(tmp_path), _StubLogger())

    assert out != str(captions_csv)  # a new, remapped file - not the original
    remapped = pd.read_csv(out)
    lookup = dict(zip(remapped[schema.PATH], remapped[schema.BLIP_CAPTION]))
    assert lookup["/robust/jpeg30/a.png"] == "a photo of a smiling person"
    assert lookup["/robust/jpeg30/b.png"] == "a photo of a serious person"


def test_no_source_path_column_returns_captions_csv_unchanged(tmp_path):
    """A clean (non-perturbed) index like test_index.csv has no source_path column - must be
    passed through untouched, not remapped."""
    captions_csv = tmp_path / "defake_predictions_aspect.csv"
    pd.DataFrame({
        schema.PATH: ["/orig/a.png"], schema.BLIP_CAPTION: ["a caption"],
    }).to_csv(captions_csv, index=False)

    index_csv = tmp_path / "test_index.csv"
    pd.DataFrame({schema.PATH: ["/orig/a.png"], schema.LABEL: ["real"]}).to_csv(
        index_csv, index=False)

    out = pdh._resolve_captions_csv(str(index_csv), str(captions_csv), str(tmp_path), _StubLogger())
    assert out == str(captions_csv)


def test_none_captions_csv_passthrough(tmp_path):
    index_csv = tmp_path / "index.csv"
    pd.DataFrame({schema.PATH: ["/a.png"], "source_path": ["/orig/a.png"]}).to_csv(
        index_csv, index=False)
    assert pdh._resolve_captions_csv(str(index_csv), None, str(tmp_path), _StubLogger()) is None
