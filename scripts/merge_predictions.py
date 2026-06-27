"""
merge_predictions.py
Concatenate the main DE-FAKE prediction CSV and the DFFD prediction CSV into a single
defake_predictions_all.csv (the file the analysis/scoring scripts consume).

Paths are env-overridable (configs/paths.env) with the original defaults preserved.

Run anywhere with python (stdlib only):
    python scripts/merge_predictions.py
"""

import csv
import os

WTP_ROOT = os.environ.get("WTP_ROOT", "/pitsec_sose26_topic8")
CSV1 = os.environ.get("WTP_PRED_CSV", f"{WTP_ROOT}/dataset/defake_predictions.csv")
CSV2 = os.environ.get("WTP_PRED_DFFD_CSV", f"{WTP_ROOT}/dataset/defake_predictions_dffd.csv")
OUTPUT = os.environ.get("WTP_PRED_ALL_CSV", f"{WTP_ROOT}/dataset/defake_predictions_all.csv")


def main():
    with open(CSV1, newline="") as f:
        reader = csv.DictReader(f)
        rows1 = list(reader)
        fieldnames = reader.fieldnames

    rows2 = []
    if os.path.exists(CSV2):
        with open(CSV2, newline="") as f:
            rows2 = list(csv.DictReader(f))
    else:
        print(f"[warn] {CSV2} not found - merging only the main predictions CSV")

    all_rows = rows1 + rows2

    with open(OUTPUT, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(all_rows)

    print(f"Merged: {len(rows1)} + {len(rows2)} = {len(all_rows)} rows")
    print(f"Output: {OUTPUT}")


if __name__ == "__main__":
    main()
