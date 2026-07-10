#!/usr/bin/env python3
"""
extract_openforensics.py
Project-owned, self-contained OpenForensics face-crop extractor. No dependency on any external
/ ad-hoc extraction or sidecar CSV: it reads the OpenForensics COCO-style polygon JSONs
(ground truth) and writes crops DIRECTLY into <out_dir>/{real,fake}/, which is exactly what
build_master_index.py consumes via the openforensics_real / openforensics_fake config entries
(label + generator come from the directory + config, not from any CSV).

Why this exists: our pipeline is directory-driven (real/ vs fake/). OpenForensics ships full
scene photos + per-face polygon annotations, so faces must be cropped out first. This is that
step, owned by us and version-controlled, so the dataset is reproducible from the raw source.

Balanced subset by design: only --per_class_limit crops per class are written (seeded random
selection), so we get a size-matched OpenForensics subset (300 real + 300 fake by default)
without extracting all ~150k faces. Selecting a cap also makes it fast enough to run on the
small Val split alone.

GROUP-AWARE SPLITTING (source-photo coupling fix): each OpenForensics scene photo can contain
BOTH a genuine and a manipulated face annotation, so a real crop and a fake crop can share one
source photo (same camera/lighting/JPEG history) - a real<->fake / train<->test bridge that the
project's dHash near-duplicate audit cannot see (the two crops are different face regions). This
extractor now records the source `image_id` for every crop it writes (both in
openforensics_metadata.csv AND in a dedicated `openforensics_groups.csv` sidecar of
`full_path,source_image_id` rows), so downstream splitting (scripts/lib/defake_head.py
stratified_split's `groups=` argument, wired through finetune_defake_head.py / train_ganfp.py /
benchmark_attribution.py / leave_one_generator_out.py / make_split.py / audit_split_leakage.py)
can keep every crop from one source photo on the SAME side of the split, instead of splitting on
the crop's own full_path alone.

RUN ON THE HOST: the OpenForensics source under /vol1 is not mounted inside the container.
Point --out_dir at the host path that the CONTAINER sees as ${WTP_ROOT}/dataset/openforensics
(so the crops land where build_master_index.py, run inside the container, will look).

HOST/CONTAINER PATH MISMATCH (record full_path as the CONTAINER would see it, not as --out_dir
literally reads on the host): build_master_index.py runs INSIDE THE CONTAINER, so every
full_path it writes into master_metadata.csv (and everything derived from it) uses the
CONTAINER-side prefix (e.g. /pitsec_sose26_topic8/dataset/openforensics/...). This script runs
on the HOST, where --out_dir is typically a DIFFERENT absolute path to the SAME bind-mounted
directory (e.g. /vol2/<user>/sharedDockerDir/dataset/openforensics/...). If full_path is
recorded using --out_dir's literal value, openforensics_groups.csv's full_path values will
NEVER match master_metadata.csv's full_path values for the exact same files - group-aware
splitting then silently does nothing (apply_group_map falls back to "no match" for every row)
even though the sidecar loads fine and looks correct. Pass --record_prefix (the CONTAINER-side
equivalent of --out_dir) so full_path is written in the CONTAINER's own path-namespace instead -
files are still physically written under --out_dir; only the RECORDED strings change.

OpenForensics convention: category_id 0 = real face, 1 = manipulated (fake) face.

Usage (host):
  python3 scripts/extract_openforensics.py \
      --root /vol1/share/DeepFake/OpenForensics \
      --out_dir /vol2/pitsec_sose26_topic8/sharedDockerDir/dataset/openforensics \
      --record_prefix /pitsec_sose26_topic8/dataset/openforensics \
      --splits Val --per_class_limit 300
"""
import argparse
import csv
import json
import os
import random
from pathlib import Path

LABEL_MAP = {0: "real", 1: "fake"}
# Generator/category names mirror configs/config.yaml so the provenance CSV is consistent with
# how build_master_index.py will label these rows (it derives them from the config, not here).
GENERATOR_MAP = {"real": "OpenForensics", "fake": "OpenForensics-fake"}
CATEGORY_MAP = {"real": "real", "fake": "out_of_set"}


def _collect(root, split, json_file):
    """Cheap pass (no image decode): return [(img_info, ann, label)] for one split's JSON."""
    with open(root / json_file) as fh:
        data = json.load(fh)
    images_by_id = {img["id"]: img for img in data["images"]}
    out = []
    for ann in data["annotations"]:
        label = LABEL_MAP.get(ann.get("category_id"))
        img_info = images_by_id.get(ann.get("image_id"))
        if label is None or img_info is None:
            continue
        out.append((img_info, ann, label))
    return out


def main(args):
    from PIL import Image

    root = Path(args.root)
    out_dir = Path(args.out_dir)
    # record_root is used ONLY for the full_path strings written into the CSVs (metadata +
    # group sidecar); out_dir remains the REAL filesystem location for mkdir/save. Defaults to
    # out_dir (unchanged behavior) when --record_prefix is not given.
    record_root = Path(args.record_prefix) if args.record_prefix else out_dir
    rng = random.Random(args.seed)

    # 1) Gather candidate annotations across the requested splits (metadata only, so capping
    #    does not require decoding 150k images).
    buckets = {"real": [], "fake": []}
    for split in args.splits:
        json_file = "%s_poly.json" % split
        if not (root / json_file).exists():
            print("Skipping %s: %s not found" % (split, json_file))
            continue
        for img_info, ann, label in _collect(root, split, json_file):
            buckets[label].append((split, img_info, ann))
        print("Collected so far: real=%d fake=%d" % (len(buckets["real"]), len(buckets["fake"])))

    # 2) Seeded selection per class (shuffle then cap), then crop ONLY the selected faces.
    fieldnames = ["filename", "full_path", "label", "generator",
                  "category", "source_dataset", "width", "height",
                  "source_image_id", "source_split", "annotation_id"]
    counts = {"real": 0, "fake": 0}
    skipped = 0
    groups_rows = []  # full_path,source_image_id sidecar for group-aware splitting
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "openforensics_metadata.csv", "w", newline="") as cf:
        writer = csv.DictWriter(cf, fieldnames=fieldnames)
        writer.writeheader()
        for label in ("real", "fake"):
            (out_dir / label).mkdir(parents=True, exist_ok=True)
            items = buckets[label]
            rng.shuffle(items)
            if args.per_class_limit and args.per_class_limit > 0:
                items = items[: args.per_class_limit]
            for split, img_info, ann in items:
                rel_path = img_info["file_name"].replace("Images/", "", 1)
                src_path = root / rel_path
                if not src_path.exists():
                    skipped += 1
                    continue
                x, y, w, h = (int(v) for v in ann["bbox"])
                try:
                    with Image.open(src_path) as im:
                        im = im.convert("RGB")
                        crop = im.crop((max(0, x), max(0, y),
                                        min(im.width, x + w), min(im.height, y + h)))
                except Exception as exc:  # noqa: BLE001
                    print("  Failed on %s: %s" % (src_path, exc))
                    skipped += 1
                    continue
                out_filename = "openforensics_%s_%s.jpg" % (split, ann["id"])
                dst = out_dir / label / out_filename           # REAL filesystem path (host)
                record_dst = record_root / label / out_filename  # path recorded in the CSVs
                crop.save(dst, "JPEG", quality=95)
                source_image_id = "%s:%s" % (split, img_info["id"])  # split-qualified: image
                # ids are only unique WITHIN one split's JSON, not across Train/Val/Test-*.
                writer.writerow({
                    "filename": out_filename,
                    "full_path": str(record_dst),
                    "label": label,
                    "generator": GENERATOR_MAP[label],
                    "category": CATEGORY_MAP[label],
                    "source_dataset": "openforensics",
                    "width": crop.width,
                    "height": crop.height,
                    "source_image_id": source_image_id,
                    "source_split": split,
                    "annotation_id": ann["id"],
                })
                groups_rows.append((str(record_dst), source_image_id))
                counts[label] += 1
            print("  %s: wrote %d (requested cap %s)" % (label, counts[label], args.per_class_limit))

    # Group-aware split sidecar: full_path -> source_image_id. Downstream scripts (see the
    # GROUP-AWARE SPLITTING note above) keep every crop sharing a source_image_id on the SAME
    # side of train/val/test, closing the same-source-photo real/fake leak. Every OTHER dataset
    # in the pipeline has no such sidecar, so its rows fall back to singleton groups (=their own
    # full_path) and split exactly as before - this is additive, not a behavior change elsewhere.
    with open(out_dir / "openforensics_groups.csv", "w", newline="") as gf:
        gw = csv.writer(gf)
        gw.writerow(["full_path", "source_image_id"])
        gw.writerows(groups_rows)
    print("Wrote group-aware split sidecar: %s (%d rows)"
          % (out_dir / "openforensics_groups.csv", len(groups_rows)))

    print("\nDone. real=%d fake=%d skipped=%d -> %s/{real,fake}/"
          % (counts["real"], counts["fake"], skipped, out_dir))
    if counts["real"] < (args.per_class_limit or 0) or counts["fake"] < (args.per_class_limit or 0):
        print("NOTE: a class came up short of the cap (missing source images or too few "
              "annotations in the chosen split). Add more --splits to reach the cap.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Crop OpenForensics faces into real/ + fake/.")
    parser.add_argument("--root", default="/vol1/share/DeepFake/OpenForensics",
                        help="OpenForensics source (contains <Split>/ + <Split>_poly.json).")
    parser.add_argument("--out_dir", required=True,
                        help="Host path that the container sees as "
                             "${WTP_ROOT}/dataset/openforensics (real/ + fake/ created here).")
    parser.add_argument("--record_prefix", default=None,
                        help="CONTAINER-side equivalent of --out_dir (e.g. "
                             "/pitsec_sose26_topic8/dataset/openforensics), used ONLY for the "
                             "full_path strings written into openforensics_metadata.csv and "
                             "openforensics_groups.csv - files are still physically written "
                             "under --out_dir. REQUIRED whenever --out_dir is a host path "
                             "different from what build_master_index.py (run inside the "
                             "container) will use for these same files, or group-aware "
                             "splitting will silently match nothing (see the module docstring).")
    parser.add_argument("--splits", nargs="+", default=["Val"],
                        help="Splits to draw from (default: Val, the smallest). Add more to "
                             "reach the cap or increase diversity.")
    parser.add_argument("--per_class_limit", type=int, default=300,
                        help="Max crops per class (0 = all). Balanced by construction.")
    parser.add_argument("--seed", type=int, default=42, help="Seed for the per-class selection.")
    main(parser.parse_args())
