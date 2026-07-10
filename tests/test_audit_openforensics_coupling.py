"""audit_openforensics_coupling.py must key annotation->image lookups PER SPLIT, never by a bare
annotation id, since COCO-style ids are only unique WITHIN one split's JSON export - two
different splits (e.g. Val_poly.json and Train_poly.json) can reuse the same small integer id
for completely unrelated annotations. Regression test for a real bug: an earlier version unioned
ann_id -> image_id across every --polygon_json file with a plain dict, so a colliding id from a
later file silently overwrote an earlier split's (correct) mapping."""
import json

import audit_openforensics_coupling as aoc


class _StubLogger:
    def info(self, *args, **kwargs):
        pass

    def warning(self, *args, **kwargs):
        pass


def test_split_from_json_path():
    assert aoc._split_from_json_path("/x/Val_poly.json") == "Val"
    assert aoc._split_from_json_path("Test-Dev_poly.json") == "Test-Dev"
    assert aoc._split_from_json_path("Test-Challenge_poly.json") == "Test-Challenge"


def test_split_and_ann_id_from_path():
    assert aoc._split_and_ann_id_from_path(
        "/d/openforensics_Val_1094.jpg") == ("Val", 1094)
    assert aoc._split_and_ann_id_from_path(
        "/d/openforensics_Test-Dev_42.jpg") == ("Test-Dev", 42)
    assert aoc._split_and_ann_id_from_path("/d/not_a_match.jpg") == (None, None)


def test_classify_coupled_groups_train_fit_leak_vs_straddling():
    """Regression for the real finding on the server: a group where the real crop is in `train`
    and the fake crop is permanently `unseen` (by design, e.g. an out-of-set generator) DOES
    straddle (the broad, less meaningful metric) but that is expected/near-100% and should NOT
    be read as a leak by itself - train_fit_leak is the metric that actually matters, and must
    correctly flag it (real fit on, fake not) while ALSO correctly NOT flagging a
    val/test-only pair (neither side ever fit on) as a leak."""
    both_classes = {
        "Val:1": [  # real in train, fake permanently unseen -> genuine leak
            {"label": "real", "split": "train"},
            {"label": "fake", "split": "unseen"},
        ],
        "Val:2": [  # real in val, fake permanently unseen -> straddles, but NOT a fit leak
            {"label": "real", "split": "val"},
            {"label": "fake", "split": "unseen"},
        ],
        "Val:3": [  # real in test, fake permanently unseen -> straddles, but NOT a fit leak
            {"label": "real", "split": "test"},
            {"label": "fake", "split": "unseen"},
        ],
        "Val:4": [  # both real and fake in train (group-aware split worked as intended)
            {"label": "real", "split": "train"},
            {"label": "fake", "split": "train"},
        ],
    }
    straddling, same_side, train_test_bridge, train_fit_leak = aoc._classify_coupled_groups(
        both_classes)

    assert set(straddling.keys()) == {"Val:1", "Val:2", "Val:3"}  # broad metric: 3/4
    assert set(same_side.keys()) == {"Val:4"}
    assert set(train_fit_leak.keys()) == {"Val:1"}  # narrow metric: only the REAL leak, 1/4


def test_ann_to_image_does_not_collide_across_splits(tmp_path):
    """Val and Train both define an annotation with id=5 (referring to DIFFERENT photos) - both
    mappings must survive, keyed separately by split."""
    val_json = tmp_path / "Val_poly.json"
    train_json = tmp_path / "Train_poly.json"
    val_json.write_text(json.dumps({
        "images": [{"id": 100, "file_name": "Images/val_100.jpg"}],
        "annotations": [{"id": 5, "image_id": 100, "category_id": 0, "bbox": [0, 0, 10, 10]}],
    }))
    train_json.write_text(json.dumps({
        "images": [{"id": 200, "file_name": "Images/train_200.jpg"}],
        "annotations": [{"id": 5, "image_id": 200, "category_id": 1, "bbox": [0, 0, 10, 10]}],
    }))

    ann_to_image = aoc._load_ann_to_image([str(val_json), str(train_json)], _StubLogger())

    assert ann_to_image[("Val", 5)] == 100
    assert ann_to_image[("Train", 5)] == 200
    assert len(ann_to_image) == 2  # NOT collapsed to 1 by an id collision
