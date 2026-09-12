"""predict_defake_head.py must remap captions via source_path for perturbed indices - otherwise
every perturbed row's caption lookup misses (keyed by the new perturbed full_path, which never
matches anything in a captions CSV built from the original images) and silently falls back to
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

    # Perturbation-style index: full_path is the new (perturbed) path; source_path is the
    # original image it was perturbed from - exactly robustness_perturb.py's generate() schema.
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


def test_clean_index_with_source_path_falls_back_to_full_path(tmp_path):
    """The clean test_index.csv carries a source_path pointing at the PRE-VARIANT original
    (prepare_variants.py writes it), which never appears in a variant-keyed captions CSV. A
    source_path-only lookup misses every row and wipes the text half of every embedding, which
    is what made the cascade's Table 6 predictions disagree with the head's own eval. The
    row's own full_path must be used as the fallback key."""
    captions_csv = tmp_path / "defake_predictions_aspect.csv"
    pd.DataFrame({
        schema.PATH: ["/ds/variants/aspect/a.png", "/ds/variants/aspect/b.png"],
        schema.BLIP_CAPTION: ["a photo of a city", "a photo of a face"],
    }).to_csv(captions_csv, index=False)

    index_csv = tmp_path / "test_index.csv"
    pd.DataFrame({
        schema.PATH: ["/ds/variants/aspect/a.png", "/ds/variants/aspect/b.png"],
        "source_path": ["/ds/sd15_txt2img/images/a.png", "/share/DeepFake/DFFD_Images/b.png"],
    }).to_csv(index_csv, index=False)

    out = pdh._resolve_captions_csv(str(index_csv), str(captions_csv), str(tmp_path), _StubLogger())

    remapped = pd.read_csv(out)
    lookup = dict(zip(remapped[schema.PATH], remapped[schema.BLIP_CAPTION]))
    assert lookup["/ds/variants/aspect/a.png"] == "a photo of a city"
    assert lookup["/ds/variants/aspect/b.png"] == "a photo of a face"


def test_zero_caption_matches_is_fatal(tmp_path):
    """Matching nothing at all means the captions CSV and the index disagree on path prefix.
    Running on would score a caption-trained head on caption-less features, so it must fail
    loudly rather than emit an all-empty caption file."""
    captions_csv = tmp_path / "captions.csv"
    pd.DataFrame({
        schema.PATH: ["/somewhere/else/a.png"], schema.BLIP_CAPTION: ["a caption"],
    }).to_csv(captions_csv, index=False)

    index_csv = tmp_path / "test_index.csv"
    pd.DataFrame({
        schema.PATH: ["/ds/variants/aspect/a.png"],
        "source_path": ["/ds/sd15_txt2img/images/a.png"],
    }).to_csv(index_csv, index=False)

    try:
        pdh._resolve_captions_csv(str(index_csv), str(captions_csv), str(tmp_path), _StubLogger())
    except SystemExit as exc:
        assert "0/1 rows" in str(exc)
    else:
        raise AssertionError("expected SystemExit when no caption matches")


def test_no_source_path_column_returns_captions_csv_unchanged(tmp_path):
    """An index with no source_path column at all must be passed through untouched."""
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
