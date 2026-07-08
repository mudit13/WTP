"""
merge_predictions.py
Concatenate the main DE-FAKE prediction CSV and the DFFD prediction CSV into a single
defake_predictions_all.csv (the file the analysis/scoring scripts consume).

Paths are env-overridable (configs/paths.env) with the original defaults preserved.

By default a missing DFFD CSV only warns (the "all" file is still written from the main CSV).
Pass --require-dffd to make a missing DFFD CSV a hard error, so a partial merge can never
silently masquerade as the full dataset.

Run anywhere with python (stdlib only):
    python scripts/merge_predictions.py [--require-dffd]
"""

import argparse
import csv
import os
import sys

WTP_ROOT = os.environ.get("WTP_ROOT", "/pitsec_sose26_topic8")
CSV1 = os.environ.get("WTP_PRED_CSV", f"{WTP_ROOT}/dataset/defake_predictions.csv")
CSV2 = os.environ.get("WTP_PRED_DFFD_CSV", f"{WTP_ROOT}/dataset/defake_predictions_dffd.csv")
OUTPUT = os.environ.get("WTP_PRED_ALL_CSV", f"{WTP_ROOT}/dataset/defake_predictions_all.csv")


def main(require_dffd: bool = False):
    with open(CSV1, newline="") as f:
        reader = csv.DictReader(f)
        rows1 = list(reader)
        fieldnames = reader.fieldnames

    rows2 = []
    if os.path.exists(CSV2):
        with open(CSV2, newline="") as f:
            rows2 = list(csv.DictReader(f))
    else:
        msg = (f"DFFD predictions not found: {CSV2}")
        if require_dffd:
            print(f"[error] {msg} - aborting (--require-dffd set)", file=sys.stderr)
            raise SystemExit(2)
        print("=" * 72)
        print(f"[WARN] {msg}")
        print("[WARN] Writing a PARTIAL 'all' file from the main predictions ONLY.")
        print("[WARN] Re-run 'run_defake_batch.py --dataset_filter dffd_', or pass "
              "--require-dffd to forbid this.")
        print("=" * 72)

    all_rows = rows1 + rows2

    with open(OUTPUT, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(all_rows)

    print(f"Merged: {len(rows1)} + {len(rows2)} = {len(all_rows)} rows")
    print(f"Output: {OUTPUT}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--require-dffd", action="store_true",
                        help="Fail (exit 2) if the DFFD predictions CSV is missing.")
    main(require_dffd=parser.parse_args().require_dffd)
