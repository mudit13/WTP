"""
Canonical schema for the project's CSVs, matching the EXISTING server pipeline
(build_master_index.py / run_defake_batch.py on github.com/mudit13/WTP).

Centralizing the column names here means every analysis script reads/writes the same
schema the team's generation + inference scripts already produce, so nothing has to be
re-plumbed. ASCII-only; Python 3.9.

master_metadata.csv columns:
    filename, full_path, label, generator, category, source_dataset, width, height

defake prediction columns (added by run_defake_batch.py / run_defake_dffd.py):
    defake_predict (0=real, 1=fake), prob_real, prob_fake, blip_caption
"""

# --- master_metadata.csv columns --------------------------------------------
FILENAME = "filename"
PATH = "full_path"            # container path to the image
LABEL = "label"               # "real" | "fake"
GENERATOR = "generator"       # human name, e.g. "SD1.5", "FLUX.1-schnell", "FFHQ"
CATEGORY = "category"         # "real" | "near_in_set" | "out_of_set"
DATASET = "source_dataset"    # e.g. "sd15_txt2img", "dffd_pggan_v1"
WIDTH = "width"
HEIGHT = "height"

MASTER_COLUMNS = [FILENAME, PATH, LABEL, GENERATOR, CATEGORY, DATASET, WIDTH, HEIGHT]

# --- prediction columns (DE-FAKE binary detector) ---------------------------
DEFAKE_PREDICT = "defake_predict"   # int: 0 real, 1 fake (-1 on error)
PROB_REAL = "prob_real"
PROB_FAKE = "prob_fake"
BLIP_CAPTION = "blip_caption"

# --- label constants ---------------------------------------------------------
REAL = "real"
FAKE = "fake"


def is_fake_label(series):
    """Boolean mask: ground-truth label == fake. Accepts a pandas Series of strings."""
    return series.astype(str).str.lower() == FAKE


def is_fake_predict(series):
    """Boolean mask for the DE-FAKE prediction column.

    Handles both the numeric convention (1 = fake, 0 = real, -1 = error) and any legacy
    string form ("fake"/"real"). Error rows (-1) are treated as not-fake.
    """
    import pandas as pd
    s = series
    numeric = pd.to_numeric(s, errors="coerce")
    if numeric.notna().any():
        return numeric == 1
    return s.astype(str).str.lower() == FAKE
