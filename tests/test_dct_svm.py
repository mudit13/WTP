from types import SimpleNamespace

import numpy as np
import pandas as pd

import dct_svm


def _write_features(path, X, labels, generators, paths):
    np.savez_compressed(
        path,
        X=np.asarray(X, dtype=np.float32),
        label=np.asarray(labels),
        generator=np.asarray(generators),
        dataset=np.asarray(["d"] * len(labels)),
        paths=np.asarray(paths),
    )


def test_training_features_are_separate_and_oos_generator_is_excluded(tmp_path):
    paths = ["r1", "r2", "r3", "f1", "f2", "f3", "of1", "of2"]
    labels = ["real", "real", "real", "fake", "fake", "fake", "fake", "fake"]
    generators = ["Real", "Real", "Real", "A", "A", "A", "OpenForensics-fake",
                  "OpenForensics-fake"]
    clean = np.arange(16, dtype=float).reshape(8, 2)
    train_aug = clean + 100.0
    clean_path = tmp_path / "clean.npz"
    train_path = tmp_path / "train_aug.npz"
    _write_features(clean_path, clean, labels, generators, paths)
    _write_features(train_path, train_aug, labels, generators, paths)
    test_index = tmp_path / "test.csv"
    pd.DataFrame({"full_path": ["r3", "f3", "of1"]}).to_csv(test_index, index=False)

    args = SimpleNamespace(
        features=str(clean_path), train_features=str(train_path),
        out_dir=str(tmp_path / "out"), mode="random", model=None,
        holdout_generators=None, exclude_train_generators=["OpenForensics-fake"],
        test_size=0.2, test_index=str(test_index), seed=42,
    )
    dct_svm.main(args)
    per_image = pd.read_csv(tmp_path / "out" / "dct_per_image.csv")
    assert set(per_image["full_path"]) == {"r3", "f3", "of1"}


def test_misaligned_training_features_fail(tmp_path):
    clean_path = tmp_path / "clean.npz"
    train_path = tmp_path / "train.npz"
    _write_features(clean_path, [[0], [1]], ["real", "fake"], ["R", "A"], ["r", "f"])
    _write_features(train_path, [[0], [1]], ["real", "fake"], ["R", "A"], ["f", "r"])
    args = SimpleNamespace(
        features=str(clean_path), train_features=str(train_path),
        out_dir=str(tmp_path / "out"), mode="random", model=None,
        holdout_generators=None, exclude_train_generators=None,
        test_size=0.5, test_index=None, seed=42,
    )
    try:
        dct_svm.main(args)
    except SystemExit as exc:
        assert "do not align" in str(exc)
    else:
        raise AssertionError("misaligned training features should fail")


def test_oos_group_overlap_removes_paired_real_from_training():
    paths = np.array(["real_pair", "fake_pair", "other_real", "known_fake"])
    groups = np.array(["source:1", "source:1", "other_real", "known_fake"])
    train = np.array([True, False, True, True])
    heldout = np.array([False, True, False, False])
    cleaned, n_excluded, n_groups = dct_svm._exclude_heldout_group_overlap(
        train, heldout, groups, paths)
    assert list(cleaned) == [False, False, True, True]
    assert n_excluded == 1
    assert n_groups == 1
