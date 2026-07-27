import pandas as pd

import compare_ffhq_ablation


def _config():
    return {
        "attribution": {
            "fake_generators": ["A", "StyleGAN3-FFHQ"],
            "real_generators": ["London-DB", "FFHQ", "CelebA", "OpenForensics"],
            "out_of_set_generators": ["OpenForensics-fake"],
        }
    }


def _write(path, style_prediction):
    pd.DataFrame({
        "full_path": ["a1", "s1", "s2"],
        "true_generator": ["A", "StyleGAN3-FFHQ", "StyleGAN3-FFHQ"],
        "pred_generator": ["A", style_prediction, "StyleGAN3-FFHQ"],
        "confidence": [0.9, 0.8, 0.9],
        "entropy": [0.1, 0.2, 0.1],
        "in_set": [True, True, True],
    }).to_csv(path, index=False)


def test_ffhq_ablation_uses_identical_fake_rows_and_tracks_stylegan_shift(tmp_path):
    with_path = tmp_path / "with.csv"
    without_path = tmp_path / "without.csv"
    _write(with_path, "FFHQ")
    _write(without_path, "StyleGAN3-FFHQ")
    result, paired = compare_ffhq_ablation.compare(
        str(with_path), str(without_path), _config())

    assert len(paired) == 3
    assert result["with_ffhq"]["stylegan3"]["real_prediction_rate"] == 0.5
    assert result["without_ffhq"]["stylegan3"]["real_prediction_rate"] == 0.0
    assert result["delta_without_minus_with"]["stylegan3_recall"] == 0.5
    assert result["delta_without_minus_with"]["cohen_kappa"] > 0
