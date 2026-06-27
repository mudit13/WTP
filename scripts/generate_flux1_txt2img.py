#!/usr/bin/env python3
"""
FLUX.1-schnell txt2img Generation Script
PITSEC SoSe26 - Topic 8: AI Image Detection & Attribution
Author: Vishnu

Generates >=100 images from FLUX.1-schnell for DE-FAKE evaluation.
Output: /pitsec_sose26_topic8/dataset/flux1_txt2img/

Usage (inside Docker) - FLUX uses its OWN venv (venv_flux1), not venv_sd15:
    source /pitsec_sose26_topic8/venv_flux1/bin/activate
    python3.9 scripts/generate_flux1_txt2img.py

NOTE: output/model paths below are hardcoded to the container layout
(/pitsec_sose26_topic8/...). This is intentional for the generation scripts (the data lives
there regardless of repo location); the analysis scripts use configs/paths.env instead.
"""

import torch
# Compatibility patch: same as Sushmita's SD1.5 script
class _DeviceMock:
    def __getattr__(self, name):
        return lambda *args, **kwargs: None
    def is_available(self): return False
    def device_count(self): return 0
torch.backends.cuda.enable_flash_sdp(False)
torch.backends.cuda.enable_mem_efficient_sdp(False)
torch.backends.cuda.enable_math_sdp(True)

for _dev in ["xpu", "mps", "npu", "mlu", "musa"]:
    if not hasattr(torch, _dev):
        setattr(torch, _dev, _DeviceMock())

from diffusers import FluxPipeline
import csv
from pathlib import Path
from datetime import datetime

# ── Configuration ──────────────────────────────────────────────────────────────

OUTPUT_DIR    = Path("/pitsec_sose26_topic8/dataset/flux1_txt2img")
IMAGES_DIR    = OUTPUT_DIR / "images"
METADATA_PATH = OUTPUT_DIR / "metadata.csv"
MODEL_CACHE   = Path("/pitsec_sose26_topic8/models")
MODEL_ID      = "black-forest-labs/FLUX.1-schnell"

NUM_STEPS        = 4      # FLUX.1-schnell is optimized for 4 steps
CFG_SCALE        = 0.0    # schnell variant does not use classifier-free guidance
IMAGE_SIZE       = 512
SEEDS_PER_PROMPT = 12     # 9 prompts x 12 seeds = 108 images

NEGATIVE_PROMPT = (
    "cartoon, illustration, anime, painting, blurry, low quality, "
    "deformed, ugly, watermark, text, signature, "
    "nude, naked, nudity, nsfw, bare skin, exposed body, "
    "distorted face, deformed face, bad anatomy, malformed, "
    "crooked teeth, extra fingers, mutated hands, "
    "artistic, stylized, render, 3d, digital art, over-processed, "
    "camera, dslr, photography equipment, back of head, rear view, "
    "crossed eyes, misaligned eyes, asymmetric face, wall-eye, lazy eye, "
    "uneven eyes, different sized eyes, squinting, airbrushed, plastic skin"
)
# 9 prompts - identical to Sushmita's SD1.5 for cross-generator consistency
PROMPTS = [
    # Expression variants (subtle, closed-mouth)
    "RAW photo, photorealistic studio portrait of a person with a warm genuine smile, "
    "symmetrical face, aligned eyes, skin pores visible, natural skin texture, "
    "subsurface scattering, soft even studio lighting, seamless grey studio backdrop, "
    "facing camera, 85mm portrait lens, shallow depth of field, sharp focus.",
    "RAW photo, photorealistic studio portrait of a person with a calm serious expression, "
    "symmetrical face, aligned eyes, skin pores visible, natural skin texture, "
    "subsurface scattering, soft even studio lighting, seamless grey studio backdrop, "
    "facing camera, 85mm portrait lens, shallow depth of field, sharp focus.",
    "RAW photo, photorealistic studio portrait of a person with slightly raised eyebrows "
    "and a gentle curious look, symmetrical face, aligned eyes, skin pores visible, "
    "natural skin texture, subsurface scattering, soft even studio lighting, "
    "seamless grey studio backdrop, facing camera, 85mm portrait lens, sharp focus.",
    # Lighting variants
    "RAW photo, photorealistic studio portrait of a person with harsh side-lighting "
    "casting visible shadows on the face, symmetrical face, aligned eyes, "
    "skin pores visible, natural skin texture, seamless white studio backdrop, "
    "facing camera, 85mm portrait lens, sharp focus.",
    "RAW photo, photorealistic studio portrait of a person with warm soft rim lighting, "
    "symmetrical face, aligned eyes, skin pores visible, natural skin texture, "
    "subsurface scattering, seamless dark studio backdrop, facing camera, "
    "85mm portrait lens, shallow depth of field, sharp focus.",
    "RAW photo, photorealistic studio portrait of a person under low ambient indoor lighting, "
    "symmetrical face, aligned eyes, skin pores visible, natural skin texture, "
    "seamless black studio backdrop, facing camera, 85mm portrait lens, sharp focus.",
    # Outdoor portrait variants (facing camera)
    "RAW photo, photorealistic outdoor portrait of a person facing the camera, "
    "symmetrical face, aligned eyes, skin pores visible, wearing casual clothes, "
    "soft natural daylight, bokeh blurred green background, "
    "85mm portrait lens, shallow depth of field, sharp focus.",
    "RAW photo, photorealistic outdoor portrait of a person facing the camera "
    "with wind-swept hair, symmetrical face, aligned eyes, skin pores visible, "
    "wearing casual clothes, soft natural daylight, bokeh blurred outdoor background, "
    "85mm portrait lens, shallow depth of field, sharp focus.",
    "RAW photo, photorealistic outdoor portrait of a person facing the camera, "
    "symmetrical face, aligned eyes, skin pores visible, wearing casual clothes, "
    "dappled natural sunlight, bokeh blurred outdoor background, "
    "relaxed expression, 85mm portrait lens, sharp focus.",
]

# ── Helpers ────────────────────────────────────────────────────────────────────

def setup_dirs():
    IMAGES_DIR.mkdir(parents=True, exist_ok=True)
    MODEL_CACHE.mkdir(parents=True, exist_ok=True)
    print(f"Output directory: {IMAGES_DIR}")


def load_existing_metadata():
    """Return set of filenames already generated — allows resuming a crashed run."""
    existing = set()
    if METADATA_PATH.exists():
        with open(METADATA_PATH, "r") as f:
            for row in csv.DictReader(f):
                existing.add(row["filename"])
    return existing


def load_pipeline():
    torch.cuda.empty_cache()  # add this
    torch.cuda.reset_peak_memory_stats()  # add this

    print(f"Loading {MODEL_ID} (cache: {MODEL_CACHE}) ...")
    pipe = FluxPipeline.from_pretrained(
        MODEL_ID,
        torch_dtype=torch.bfloat16,    # FLUX.1 requires bfloat16
        cache_dir=str(MODEL_CACHE),
    )
    pipe.enable_model_cpu_offload()

    print("Pipeline ready.\n")
    return pipe


# ── Generation ─────────────────────────────────────────────────────────────────

def generate(pipe, existing_files):
    total   = len(PROMPTS) * SEEDS_PER_PROMPT
    skipped = 0
    done    = 0

    write_header = not METADATA_PATH.exists()
    csv_file = open(METADATA_PATH, "a", newline="")
    writer   = csv.DictWriter(csv_file, fieldnames=[
        "filename", "generator", "mode",
        "prompt_idx", "prompt", "negative_prompt",
        "seed", "steps", "cfg", "width", "height", "timestamp",
    ])
    if write_header:
        writer.writeheader()

    for p_idx, prompt in enumerate(PROMPTS):
        for s_idx in range(SEEDS_PER_PROMPT):

            # deterministic seed: same scheme as Sushmita for consistency
            seed     = p_idx * 1000 + s_idx
            filename = f"flux1_txt2img_p{p_idx:02d}_s{s_idx:03d}.png"
            count    = done + skipped + 1

            if filename in existing_files:
                print(f"[{count}/{total}] SKIP {filename} (already exists)")
                skipped += 1
                continue

            print(f"[{count}/{total}] Generating {filename}  "
                  f"(prompt {p_idx+1}/{len(PROMPTS)}, seed {seed})")

            generator = torch.Generator("cuda").manual_seed(seed)

            image = pipe(
                prompt=prompt,
                num_inference_steps=NUM_STEPS,
                guidance_scale=CFG_SCALE,
                width=IMAGE_SIZE,
                height=IMAGE_SIZE,
                generator=generator,
            ).images[0]

            image.save(IMAGES_DIR / filename)

            writer.writerow({
                "filename":        filename,
                "generator":       "flux1-schnell",
                "mode":            "txt2img",
                "prompt_idx":      p_idx,
                "prompt":          prompt,
                "negative_prompt": NEGATIVE_PROMPT,
                "seed":            seed,
                "steps":           NUM_STEPS,
                "cfg":             CFG_SCALE,
                "width":           IMAGE_SIZE,
                "height":          IMAGE_SIZE,
                "timestamp":       datetime.now().isoformat(),
            })
            csv_file.flush()    # write row immediately — safe against crashes
            done += 1

    csv_file.close()
    print(f"\n{'─'*50}")
    print(f"Finished.  Generated: {done}  |  Skipped: {skipped}  |  Total: {done+skipped}")
    print(f"Images   → {IMAGES_DIR}")
    print(f"Metadata → {METADATA_PATH}")


# ── Entry point ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 50)
    print("FLUX.1-schnell txt2img Dataset Generation")
    print("=" * 50)

    setup_dirs()

    existing = load_existing_metadata()
    if existing:
        print(f"Resuming: {len(existing)} images already exist.\n")

    pipe = load_pipeline()
    generate(pipe, existing)
