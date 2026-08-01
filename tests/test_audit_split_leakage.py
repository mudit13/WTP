import pandas as pd

import audit_split_leakage


def test_finetune_audit_distinguishes_excluded_reals_from_true_oos(tmp_path):
    rows = []
    for generator, label in [
        ("A", "fake"),
        ("B", "fake"),
        ("RealSource", "real"),
        ("OOS-fake", "fake"),
    ]:
        for i in range(10):
            rows.append({
                "full_path": "/%s/%d.png" % (generator, i),
                "generator": generator,
                "label": label,
                "source_dataset": generator,
            })
    index = tmp_path / "index.csv"
    pd.DataFrame(rows).to_csv(index, index=False)
    config = {
        "dataset_root": str(tmp_path / "dataset"),
        "seed": 42,
        "test_size": 0.2,
        "val_size": 0.1,
        "attribution": {
            "primary_mode": "fake_only",
            "fake_generators": ["A", "B"],
            "real_generators": ["RealSource"],
            "out_of_set_generators": ["OOS-fake"],
        },
    }

    result = audit_split_leakage._finetune_splits(
        str(index), config, class_mode="fake_only")
    assert set(result.loc[result["generator"] == "RealSource", "split"]) == {"excluded"}
    assert set(result.loc[result["generator"] == "OOS-fake", "split"]) == {"unseen"}
    assert set(result.loc[result["generator"].isin(["A", "B"]), "split"]) == {
        "train", "val", "test"}


def test_leakage_gate_only_fails_enabled_hard_invariants():
    audit = {
        "exact_cross_split_duplicates": {"count": 2},
        "group_straddle": {"n_groups_straddling": 1},
        "near_cross_split_duplicates": {"count": 999},
    }
    assert audit_split_leakage._gate_failures(audit) == []
    assert audit_split_leakage._gate_failures(
        audit, fail_on_exact=True) == ["2 exact cross-split duplicate group(s)"]
    assert audit_split_leakage._gate_failures(
        audit, fail_on_group_straddle=True) == ["1 source group(s) straddling splits"]
