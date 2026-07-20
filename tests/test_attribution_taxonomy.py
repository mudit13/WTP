import csv

import numpy as np
import pandas as pd

import evaluate_cascade
import make_img2img_group_map
import bootstrap_metrics
from lib import attribution_taxonomy, features_cache


def _config():
    return {
        "attribution": {
            "primary_mode": "fake_only",
            "fake_generators": ["A", "B"],
            "in_set_generators": ["A", "B"],
            "finetune_new_classes": [],
            "real_generators": ["R1", "R2"],
            "real_class_name": "real",
            "real_sample_cap": 4,
            "out_of_set_generators": ["OF-fake"],
        }
    }


def test_primary_and_joint_population_are_disjoint_from_oos():
    generators = np.array(["A", "A", "B", "B", "R1", "R1", "R1",
                           "R2", "R2", "R2", "OF-fake"])
    paths = np.array(["p%d" % i for i in range(len(generators))])

    primary = attribution_taxonomy.prepare_population(
        generators, paths, _config(), mode="fake_only")
    assert primary["classes"] == ["A", "B"]
    assert list(generators[primary["train_mask"]]) == ["A", "A", "B", "B"]
    assert list(generators[primary["oos_mask"]]) == ["OF-fake"]
    assert not np.any(primary["train_mask"] & primary["oos_mask"])

    joint = attribution_taxonomy.prepare_population(
        generators, paths, _config(), mode="joint")
    assert joint["classes"] == ["A", "B", "real"]
    assert int(joint["real_mask"].sum()) == 4
    selected_sources = generators[joint["real_mask"]]
    assert int((selected_sources == "R1").sum()) == 2
    assert int((selected_sources == "R2").sum()) == 2
    assert set(joint["mapped_generators"][joint["real_mask"]]) == {"real"}
    assert not np.any(joint["train_mask"] & joint["oos_mask"])


def test_real_cap_is_content_stable_under_row_reordering():
    generators = np.array(["R1"] * 5 + ["R2"] * 5)
    paths = np.array(["/r1/%d" % i for i in range(5)]
                     + ["/r2/%d" % i for i in range(5)])
    first = attribution_taxonomy.balanced_real_mask(
        generators, paths, _config(), cap=4, seed=42)
    order = np.array([7, 2, 9, 0, 5, 1, 8, 3, 6, 4])
    second = attribution_taxonomy.balanced_real_mask(
        generators[order], paths[order], _config(), cap=4, seed=42)
    assert set(paths[first]) == set(paths[order][second])


def test_missing_fake_class_fails_fast():
    with np.testing.assert_raises_regex(ValueError, "Missing configured fake"):
        attribution_taxonomy.prepare_population(
            np.array(["A"]), np.array(["p"]), _config(), mode="fake_only")


def test_training_aug_cache_is_distinct_from_clean_cache():
    assert features_cache.training_aug_cache_path("/x/features.npz") == \
        "/x/features_train_jpegaug.npz"
    assert features_cache.training_aug_cache_path(None) is None


def test_display_name_preserves_canonical_class_ids():
    config = _config()
    config["attribution"]["display_names"] = {
        "B": "Generator B (qualified condition)"
    }
    assert attribution_taxonomy.classes_for_mode(config, "fake_only") == ["A", "B"]
    assert attribution_taxonomy.display_names(config, ["A", "B"]) == [
        "A", "Generator B (qualified condition)"]


def test_group_bootstrap_keeps_repeated_derivatives_together():
    strat = np.array(["A", "A", "A", "B", "B"])
    groups = np.array(["id1", "id1", "id2", "id3", "id4"])
    idx = bootstrap_metrics._strat_group_resample(
        strat, groups, np.random.default_rng(7))
    selected = groups[idx].tolist()
    # Every time id1 is sampled, both of its derivative rows appear together.
    assert selected.count("id1") % 2 == 0


def test_img2img_group_map_contains_source_and_all_derivatives(tmp_path):
    metadata = tmp_path / "metadata.csv"
    fields = ["output_path", "source_image", "source_identity"]
    with open(metadata, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields)
        writer.writeheader()
        writer.writerow({"output_path": "/img/a1.png", "source_image": "/real/a.jpg",
                         "source_identity": "a"})
        writer.writerow({"output_path": "/img/a2.png", "source_image": "/real/a.jpg",
                         "source_identity": "a"})
        writer.writerow({"output_path": "/img/b1.png", "source_image": "/real/b.jpg",
                         "source_identity": "b"})
    rows = make_img2img_group_map.build_rows(str(metadata))
    mapping = {row["full_path"]: row["source_image_id"] for row in rows}
    assert len(mapping) == 5
    assert mapping["/real/a.jpg"] == mapping["/img/a1.png"] == mapping["/img/a2.png"]
    assert mapping["/real/b.jpg"] == mapping["/img/b1.png"]
    assert mapping["/real/a.jpg"] != mapping["/real/b.jpg"]


def test_cascade_counts_detection_and_attribution_errors_separately():
    dct = pd.DataFrame({
        "full_path": ["a1", "a2", "b1", "r1", "of1"],
        "generator": ["A", "A", "B", "R1", "OF-fake"],
        "y_true": [1, 1, 1, 0, 1],
        "score": [2.0, -1.0, 1.0, 0.5, 0.8],
        "pred": [1, 0, 1, 1, 1],
    })
    attr = pd.DataFrame({
        "full_path": ["a1", "a2", "b1", "r1", "of1"],
        "pred_generator": ["A", "A", "A", "B", "B"],
        "confidence": [0.9, 0.8, 0.7, 0.6, 0.55],
    })
    result, per_image = evaluate_cascade.evaluate(dct, attr, _config())
    known = result["known_fake"]
    assert known["n"] == 3
    assert known["n_detected"] == 2
    assert known["n_not_detected"] == 1
    assert known["conditional_attribution"]["top1_accuracy"] == 0.5
    assert known["end_to_end_attribution"]["top1_accuracy"] == 1.0 / 3.0
    assert result["real_false_positives"]["n_predicted_fake"] == 1
    assert result["openforensics_fake_challenge"]["detection_recall"] == 1.0
    assert int(per_image["end_to_end_correct"].sum()) == 1
