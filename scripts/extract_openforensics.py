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

RUN ON THE HOST: the OpenForensics source under /vol1 is not mounted inside the container.
Point --out_dir at the host path that the CONTAINER sees as ${WTP_ROOT}/dataset/openforensics
(so the crops land where build_master_index.py, run inside the container, will look).

OpenForensics convention: category_id 0 = real face, 1 = manipulated (fake) face.

Usage (host):
  python3 scripts/extract_openforensics.py \
      --root /vol1/share/DeepFake/OpenForensics \
      --out_dir <host path mapping to /pitsec_sose26_topic8/dataset/openforensics> \
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
                  "category", "source_dataset", "width", "height"]
    counts = {"real": 0, "fake": 0}
    skipped = 0
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
                dst = out_dir / label / out_filename
                crop.save(dst, "JPEG", quality=95)
                writer.writerow({
                    "filename": out_filename,
                    "full_path": str(dst),
                    "label": label,
                    "generator": GENERATOR_MAP[label],
                    "category": CATEGORY_MAP[label],
                    "source_dataset": "openforensics",
                    "width": crop.width,
                    "height": crop.height,
                })
                counts[label] += 1
            print("  %s: wrote %d (requested cap %s)" % (label, counts[label], args.per_class_limit))

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
    parser.add_argument("--splits", nargs="+", default=["Val"],
                        help="Splits to draw from (default: Val, the smallest). Add more to "
                             "reach the cap or increase diversity.")
    parser.add_argument("--per_class_limit", type=int, default=300,
                        help="Max crops per class (0 = all). Balanced by construction.")
    parser.add_argument("--seed", type=int, default=42, help="Seed for the per-class selection.")
    main(parser.parse_args())
