#!/usr/bin/env python3
"""
Generate the SD1.5 img2img attribution class from London-DB neutral-front faces.

Every output records its source image and identity. Run make_img2img_group_map.py afterwards
so each London real and all of its derivatives are assigned to one train/val/test group.

Use the SD1.5/DE-FAKE environment on the GPU server:
  $WTP_PY_DEFAKE scripts/generate_sd15_img2img.py --revision <pinned-hf-commit>
"""
import argparse
import csv
import hashlib
import json
import os
import re
from datetime import datetime
from pathlib import Path

import torch

# Compatibility patch for the server's older torch used by the existing SD1.5 script.
if not hasattr(torch, "xpu"):
    import types
    torch.xpu = types.SimpleNamespace(empty_cache=lambda: None, is_available=lambda: False)

from diffusers import StableDiffusionImg2ImgPipeline  # noqa: E402
from PIL import Image, ImageOps  # noqa: E402


MODEL_ID = "runwayml/stable-diffusion-v1-5"
PROMPT = (
    "RAW photo, photorealistic studio portrait of a person with a calm neutral expression, "
    "facing camera, natural skin texture, skin pores visible, soft even studio lighting, "
    "seamless grey studio backdrop, 85mm portrait lens, sharp focus."
)
NEGATIVE_PROMPT = (
    "cartoon, illustration, anime, painting, blurry, low quality, deformed, ugly, watermark, "
    "text, signature, nude, naked, nudity, nsfw, distorted face, deformed face, bad anatomy, "
    "malformed, extra fingers, mutated hands, artistic, stylized, render, 3d, digital art, "
    "over-processed, crossed eyes, misaligned eyes, asymmetric face, airbrushed, plastic skin"
)


def _safe_stem(path):
    return re.sub(r"[^A-Za-z0-9_-]+", "_", Path(path).stem).strip("_") or "identity"


def _source_images(init_dir):
    exts = {".jpg", ".jpeg", ".png"}
    return sorted(p for p in Path(init_dir).iterdir()
                  if p.is_file() and p.suffix.lower() in exts)


def _identity_partition(source, seed, test_size, val_size):
    identity = "londondb:%s" % _safe_stem(source)
    digest = hashlib.sha256(
        ("%d:GROUP:%s" % (seed, identity)).encode("utf-8")).hexdigest()
    score = int(digest[:16], 16) / float(1 << 64)
    if score < test_size:
        return "test"
    if score < test_size + val_size:
        return "val"
    return "train"


def _filter_sources(sources, args):
    if args.purpose == "pilot" and args.identity_partition != "train":
        raise SystemExit("Pilot generation is restricted to train-hashed identities.")
    if args.purpose == "authoritative" and args.identity_partition != "all":
        raise SystemExit("Authoritative generation must use --identity_partition all.")
    if args.purpose == "pilot" and Path(args.output_root).name == "sd15_img2img":
        raise SystemExit("Pilot outputs must use a separate non-indexed --output_root.")

    selected = list(sources)
    if args.identity_partition != "all":
        selected = [
            p for p in selected
            if _identity_partition(
                p, args.split_seed, args.test_size, args.val_size)
            == args.identity_partition
        ]
    if args.max_sources is not None:
        selected = selected[:args.max_sources]
    if not selected:
        raise SystemExit("No source identities remain after partition filtering.")
    return selected


def _manifest(args, source_count, source_pool_count):
    return {
        "purpose": args.purpose,
        "model_id": args.model_id,
        "revision": args.revision or "<default-repository-revision>",
        "prompt": PROMPT,
        "negative_prompt": NEGATIVE_PROMPT,
        "strength": args.strength,
        "steps": args.steps,
        "cfg": args.cfg,
        "size": args.size,
        "seed_start": args.seed_start,
        "num_images": args.num_images,
        "source_count": source_count,
        "source_pool_count": source_pool_count,
        "identity_partition": args.identity_partition,
        "split_seed": args.split_seed,
        "test_size": args.test_size,
        "val_size": args.val_size,
        "source_preprocess": "EXIF transpose, RGB, aspect-preserving center crop",
    }


def _write_or_validate_manifest(path, expected):
    if path.exists():
        with open(path, encoding="utf-8") as fh:
            actual = json.load(fh)
        comparable = {k: actual.get(k) for k in expected}
        if comparable != expected:
            raise SystemExit(
                "Existing generation manifest does not match requested settings. "
                "Use a new output directory rather than mixing generation regimes.\n"
                "existing=%s\nrequested=%s" % (comparable, expected))
        return
    payload = dict(expected)
    payload["created_at"] = datetime.now().isoformat()
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2)


def _load_existing(metadata_path):
    if not metadata_path.exists():
        return set()
    with open(metadata_path, newline="", encoding="utf-8") as fh:
        return {row["filename"] for row in csv.DictReader(fh)}


def _load_pipeline(args):
    kwargs = {
        "torch_dtype": torch.float16,
        "cache_dir": args.model_cache,
        "safety_checker": None,
        "requires_safety_checker": False,
    }
    if args.revision:
        kwargs["revision"] = args.revision
    pipe = StableDiffusionImg2ImgPipeline.from_pretrained(args.model_id, **kwargs)
    pipe = pipe.to("cuda")
    pipe.enable_attention_slicing()
    return pipe


def main(args):
    if not args.revision:
        raise SystemExit("--revision is required so the generated dataset pins an exact model.")
    if not (0.0 < args.strength < 1.0):
        raise SystemExit("--strength must be strictly between 0 and 1.")
    source_pool = _source_images(args.init_dir)
    if not source_pool:
        raise SystemExit("No London-DB source images found in %s" % args.init_dir)
    sources = _filter_sources(source_pool, args)

    output_root = Path(args.output_root)
    images_dir = output_root / "images"
    metadata_path = output_root / "metadata.csv"
    images_dir.mkdir(parents=True, exist_ok=True)
    Path(args.model_cache).mkdir(parents=True, exist_ok=True)
    _write_or_validate_manifest(
        output_root / "generation_manifest.json",
        _manifest(args, len(sources), len(source_pool)))
    existing = _load_existing(metadata_path)

    pipe = _load_pipeline(args)
    resampling = getattr(Image, "Resampling", Image).LANCZOS
    fields = [
        "filename", "output_path", "generator", "mode", "purpose",
        "identity_partition", "model_id", "model_revision",
        "source_image", "source_identity", "source_index", "source_repeat", "prompt",
        "negative_prompt", "strength", "seed", "steps", "cfg", "width", "height",
        "source_preprocess", "scheduler", "torch_version", "timestamp",
    ]
    write_header = not metadata_path.exists()
    generated = skipped = 0
    with open(metadata_path, "a", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields)
        if write_header:
            writer.writeheader()
        for output_index in range(args.num_images):
            source_index = output_index % len(sources)
            source_repeat = output_index // len(sources)
            source = sources[source_index]
            seed = args.seed_start + output_index
            filename = "sd15_img2img_i%03d_r%02d_%s_s%d.png" % (
                source_index, source_repeat, _safe_stem(source), seed)
            output_path = images_dir / filename
            if filename in existing and output_path.exists():
                skipped += 1
                continue

            with Image.open(source) as raw:
                init = ImageOps.exif_transpose(raw).convert("RGB")
                init = ImageOps.fit(init, (args.size, args.size),
                                    method=resampling, centering=(0.5, 0.5))
            generator = torch.Generator("cuda").manual_seed(seed)
            image = pipe(
                prompt=PROMPT,
                negative_prompt=NEGATIVE_PROMPT,
                image=init,
                strength=args.strength,
                num_inference_steps=args.steps,
                guidance_scale=args.cfg,
                generator=generator,
            ).images[0]
            image.save(output_path)
            writer.writerow({
                "filename": filename,
                "output_path": str(output_path),
                "generator": "SD1.5",
                "mode": "img2img",
                "purpose": args.purpose,
                "identity_partition": args.identity_partition,
                "model_id": args.model_id,
                "model_revision": args.revision or "<default-repository-revision>",
                "source_image": str(source),
                "source_identity": _safe_stem(source),
                "source_index": source_index,
                "source_repeat": source_repeat,
                "prompt": PROMPT,
                "negative_prompt": NEGATIVE_PROMPT,
                "strength": args.strength,
                "seed": seed,
                "steps": args.steps,
                "cfg": args.cfg,
                "width": args.size,
                "height": args.size,
                "source_preprocess": "exif_transpose+rgb+aspect_center_crop",
                "scheduler": pipe.scheduler.__class__.__name__,
                "torch_version": torch.__version__,
                "timestamp": datetime.now().isoformat(),
            })
            fh.flush()
            generated += 1
            print("[%d/%d] %s <- %s" % (
                output_index + 1, args.num_images, filename, source.name))

    print("Generated=%d skipped=%d total=%d metadata=%s" % (
        generated, skipped, generated + skipped, metadata_path))


if __name__ == "__main__":
    root = os.environ.get("WTP_ROOT", "/pitsec_sose26_topic8")
    parser = argparse.ArgumentParser(description="Generate SD1.5 img2img London-DB faces.")
    parser.add_argument("--init_dir",
                        default=os.path.join(root, "dataset", "londondb",
                                             "neutral_front", "neutral_front"))
    parser.add_argument("--output_root",
                        default=os.path.join(root, "dataset", "sd15_img2img"))
    parser.add_argument("--model_cache", default=os.path.join(root, "models"))
    parser.add_argument("--model_id", default=MODEL_ID)
    parser.add_argument("--revision", default=os.environ.get("WTP_SD15_REVISION"),
                        help="Pinned Hugging Face commit/revision. Strongly recommended.")
    parser.add_argument("--purpose", choices=["authoritative", "pilot"],
                        default="authoritative")
    parser.add_argument("--identity_partition", choices=["all", "train", "val", "test"],
                        default="all",
                        help="Pilot must use train; authoritative generation must use all.")
    parser.add_argument("--split_seed", type=int, default=42)
    parser.add_argument("--test_size", type=float, default=0.2)
    parser.add_argument("--val_size", type=float, default=0.1)
    parser.add_argument("--max_sources", type=int, default=None)
    parser.add_argument("--num_images", type=int, default=108)
    parser.add_argument("--strength", type=float, default=0.6)
    parser.add_argument("--steps", type=int, default=40)
    parser.add_argument("--cfg", type=float, default=8.5)
    parser.add_argument("--size", type=int, default=512)
    parser.add_argument("--seed_start", type=int, default=200000)
    main(parser.parse_args())
