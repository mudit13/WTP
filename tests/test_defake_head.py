"""Pure-numpy logic in the DE-FAKE head helpers (no torch needed):
balanced accuracy, label encoding, class weights, and the content-stable split."""
import numpy as np

from lib import defake_head


def test_balanced_accuracy_perfect_and_imbalanced():
    y = np.array([0, 0, 0, 1])
    assert defake_head.balanced_accuracy(y, y) == 1.0
    # Majority class perfect, minority all wrong -> plain acc high, balanced acc = 0.5.
    y_true = np.array([0, 0, 0, 0, 1, 1])
    y_pred = np.array([0, 0, 0, 0, 0, 0])
    assert defake_head.balanced_accuracy(y_true, y_pred) == 0.5


def test_encode_labels_and_class_weights():
    gens = np.array(["real", "SD1.5", "real"])
    classes = ["real", "SD1.5"]
    y = defake_head.encode_labels(gens, classes)
    assert list(y) == [0, 1, 0]
    w = defake_head.compute_class_weights(y, len(classes))
    # Rarer class ("SD1.5", 1 sample) must get a larger weight than the common one.
    assert w[1] > w[0]


def _bucket_map(keys, y, **kw):
    tr, va, te = defake_head.stratified_split(y, keys=np.array(keys), **kw)
    m = {}
    for name, idx in (("train", tr), ("val", va), ("test", te)):
        for i in idx:
            m[keys[i]] = name
    return m


def test_hash_split_is_deterministic_and_keeps_counts():
    keys = ["c0_%d" % i for i in range(20)] + ["c1_%d" % i for i in range(10)]
    y = np.array([0] * 20 + [1] * 10)
    kw = dict(test_size=0.2, val_size=0.1, seed=42)
    a = _bucket_map(keys, y, **kw)
    b = _bucket_map(keys, y, **kw)
    assert a == b  # deterministic
    # Per-class counts: 20 * 0.2 = 4 test, 20 * 0.1 = 2 val for class 0.
    c0_test = sum(1 for k, v in a.items() if k.startswith("c0_") and v == "test")
    c0_val = sum(1 for k, v in a.items() if k.startswith("c0_") and v == "val")
    assert c0_test == 4 and c0_val == 2


def test_hash_split_stable_when_other_class_changes():
    keys = ["c0_%d" % i for i in range(20)] + ["c1_%d" % i for i in range(10)]
    y = np.array([0] * 20 + [1] * 10)
    kw = dict(test_size=0.2, val_size=0.1, seed=42)
    before = _bucket_map(keys, y, **kw)
    # Drop one class-1 sample; class-0 assignments must be unchanged (cross-class stability).
    keys2 = ["c0_%d" % i for i in range(20)] + ["c1_%d" % i for i in range(9)]
    y2 = np.array([0] * 20 + [1] * 9)
    after = _bucket_map(keys2, y2, **kw)
    for k in ["c0_%d" % i for i in range(20)]:
        assert before[k] == after[k]
