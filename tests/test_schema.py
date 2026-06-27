"""Schema helpers must agree with the CSV conventions the server pipeline writes."""
import pandas as pd

from lib import schema


def test_master_columns_order_and_content():
    assert schema.MASTER_COLUMNS[0] == schema.FILENAME
    assert schema.PATH == "full_path"
    for col in (schema.LABEL, schema.GENERATOR, schema.CATEGORY, schema.DATASET,
                schema.WIDTH, schema.HEIGHT):
        assert col in schema.MASTER_COLUMNS


def test_is_fake_label_case_insensitive():
    s = pd.Series(["real", "fake", "FAKE", "Real"])
    assert list(schema.is_fake_label(s)) == [False, True, True, False]


def test_is_fake_predict_numeric_convention():
    # 1 = fake, 0 = real, -1 = error (treated as not-fake)
    s = pd.Series([0, 1, -1, 1])
    assert list(schema.is_fake_predict(s)) == [False, True, False, True]


def test_is_fake_predict_string_fallback():
    s = pd.Series(["real", "fake"])
    assert list(schema.is_fake_predict(s)) == [False, True]
