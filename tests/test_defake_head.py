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


def test_group_none_is_byte_identical_to_ungrouped():
    """groups=None (or every key its own singleton group) must reproduce the pre-group-aware
    split exactly - the backward-compatibility guarantee the OpenForensics fix relies on."""
    keys = np.array(["c0_%d" % i for i in range(20)] + ["c1_%d" % i for i in range(10)])
    y = np.array([0] * 20 + [1] * 10)
    kw = dict(test_size=0.2, val_size=0.1, seed=42)
    tr1, va1, te1 = defake_head.stratified_split(y, keys=keys, **kw)
    tr2, va2, te2 = defake_head.stratified_split(y, keys=keys, groups=keys, **kw)
    assert list(tr1) == list(tr2)
    assert list(va1) == list(va2)
    assert list(te1) == list(te2)


def test_group_membership_is_id_based_not_call_population_based():
    """Regression: a row's "grouped" status must depend only on whether it has an EXPLICIT
    group id (groups[i] != keys[i]), never on how many OTHER rows of that group happen to be
    present in a given call's arrays. Otherwise two callers that filter the population
    differently before splitting (e.g. finetune_defake_head.py restricting to trained classes,
    vs. make_split.py splitting the whole index) can disagree on a coupled row's split side even
    though both pass the SAME group id for it - exactly the OpenForensics real/out-of-set-fake
    coupling case, where the out-of-set sibling is filtered out before a trained-classes-only
    split ever sees it."""
    # A "pairA" group of 2 (both present) and a "pairB" group whose second member is ABSENT from
    # this call (simulating the sibling being filtered out upstream) - both must use the SAME
    # group-hash decision for their surviving member(s), not fall back to per-class ranking for
    # the row whose sibling is missing.
    keys = np.array(["pairA_1", "pairA_2", "pairB_1", "solo_1", "solo_2", "solo_3", "solo_4"])
    groups = np.array(["pairA", "pairA", "pairB", "solo_1", "solo_2", "solo_3", "solo_4"])
    y = np.array([0, 1, 0, 0, 0, 1, 1])  # pairA/pairB rows mix classes; solo rows do not

    tr, va, te = defake_head.stratified_split(
        y, test_size=0.2, val_size=0.1, seed=7, keys=keys, groups=groups)
    split_of = {}
    for name, idx in (("train", tr), ("val", va), ("test", te)):
        for i in idx:
            split_of[keys[i]] = name

    # pairA's two members (present together) must land on the same side.
    assert split_of["pairA_1"] == split_of["pairA_2"]

    # pairB_1's bucket must be driven by the SAME group-hash rule as pairA (its id is grouped),
    # not by the per-class ranking that "solo" rows use. Verify directly against the group hash.
    from lib.defake_head import _hash_unit
    score_b = _hash_unit("GROUP:pairB", 7)
    expected_b = "test" if score_b < 0.2 else ("val" if score_b < 0.3 else "train")
    assert split_of["pairB_1"] == expected_b


def test_group_decision_matches_across_differently_filtered_calls():
    """End-to-end version of the regression above: simulate finetune_defake_head.py (restricts
    to trained classes BEFORE splitting, so an out-of-set sibling disappears) vs. make_split.py
    (splits the whole index, sibling present) and assert they agree on every trained-class row's
    test-set membership, with zero rows leaking across (a make_split "test" row that was
    finetune "train"/"val")."""
    trained = ["Real", "InSetFake"]
    out_of_set = ["OutOfSetFake"]
    rng_paths, rng_gens, rng_groups = [], [], []
    for c in trained + out_of_set:
        for i in range(30):
            rng_paths.append("%s_%03d" % (c, i))
            rng_gens.append(c)
            rng_groups.append("%s_%03d" % (c, i))  # singleton
    # 8 coupled pairs: Real (trained) + OutOfSetFake (NOT trained) sharing a source photo.
    for i in range(8):
        gid = "src:%d" % i
        rng_paths.append("real_src_%d" % i); rng_gens.append("Real"); rng_groups.append(gid)
        rng_paths.append("fake_src_%d" % i); rng_gens.append("OutOfSetFake"); rng_groups.append(gid)

    paths = np.array(rng_paths)
    generators = np.array(rng_gens)
    groups = np.array(rng_groups)

    in_mask = np.isin(generators, trained)
    y_ft = defake_head.encode_labels(generators[in_mask], sorted(trained))
    _, _, te_ft = defake_head.stratified_split(
        y_ft, test_size=0.2, val_size=0.1, seed=42, keys=paths[in_mask], groups=groups[in_mask])
    test_ft = set(paths[in_mask][te_ft].tolist())
    tr_ft, va_ft, _ = defake_head.stratified_split(
        y_ft, test_size=0.2, val_size=0.1, seed=42, keys=paths[in_mask], groups=groups[in_mask])
    trainval_ft = set(paths[in_mask][tr_ft].tolist()) | set(paths[in_mask][va_ft].tolist())

    y_ms = defake_head.encode_labels(generators, sorted(set(generators.tolist())))
    _, _, te_ms = defake_head.stratified_split(
        y_ms, test_size=0.2, val_size=0.0, seed=42, keys=paths, groups=groups)
    test_ms = set(paths[te_ms].tolist())

    trained_paths = set(paths[in_mask].tolist())
    assert (test_ft & trained_paths) == (test_ms & trained_paths)
    assert not ((test_ms & trained_paths) & trainval_ft)


def test_group_aware_split_keeps_coupled_group_on_one_side():
    """A source photo contributing one real crop + one fake crop (an OpenForensics-style
    same-source-photo pair) must land ENTIRELY on one split side, even though the two crops
    belong to different classes."""
    # 20 singleton reals, 20 singleton fakes, plus 5 coupled (real,fake) pairs sharing a group id.
    keys = (["real_%d" % i for i in range(20)] + ["fake_%d" % i for i in range(20)]
           + ["pair%d_real" % i for i in range(5)] + ["pair%d_fake" % i for i in range(5)])
    y = np.array([0] * 20 + [1] * 20 + [0] * 5 + [1] * 5)
    groups = (["real_%d" % i for i in range(20)] + ["fake_%d" % i for i in range(20)]
             + ["pair%d" % i for i in range(5)] + ["pair%d" % i for i in range(5)])
    tr, va, te = defake_head.stratified_split(
        y, test_size=0.2, val_size=0.1, seed=42,
        keys=np.array(keys), groups=np.array(groups))
    split_of = {}
    for name, idx in (("train", tr), ("val", va), ("test", te)):
        for i in idx:
            split_of[i] = name
    for i in range(5):
        real_idx = keys.index("pair%d_real" % i)
        fake_idx = keys.index("pair%d_fake" % i)
        assert split_of[real_idx] == split_of[fake_idx], (
            "coupled pair %d straddled the split (real=%s, fake=%s)"
            % (i, split_of[real_idx], split_of[fake_idx]))
    # Every index assigned exactly once, and all rows accounted for.
    assert sorted(split_of.keys()) == list(range(len(keys)))
