#!/usr/bin/env python3
"""
SD 1.5 txt2img Generation Script
PITSEC SoSe26 - Topic 8: AI Image Detection & Attribution
Author: Sushmita
 
Generates ≥100 images from SD 1.5 for DE-FAKE evaluation.
Output: /pitsec_sose26_topic8/dataset/sd15_txt2img/
"""
 
import torch
# Compatibility patch: torch 1.12.1 has no xpu module
if not hasattr(torch, 'xpu'):
    import types
    torch.xpu = types.SimpleNamespace(empty_cache=lambda: None, is_available=lambda: False)
 
from diffusers import StableDiffusionPipeline
import csv
from pathlib import Path
from datetime import datetime
 
# ── Configuration ──────────────────────────────────────────────────────────────
 
OUTPUT_DIR    = Path("/pitsec_sose26_topic8/dataset/sd15_txt2img")
IMAGES_DIR    = OUTPUT_DIR / "images"
METADATA_PATH = OUTPUT_DIR / "metadata.csv"
MODEL_CACHE   = Path("/pitsec_sose26_topic8/models")   # persistent cache
MODEL_ID      = "runwayml/stable-diffusion-v1-5"
 
NUM_STEPS        = 40     # increased from 30 for better detail
CFG_SCALE        = 8.5    # increased from 7.5 for stronger prompt adherence
IMAGE_SIZE       = 512
SEEDS_PER_PROMPT = 12    # 9 prompts × 12 seeds = 108 images
 
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
 
# 9 prompts — 3 groups × 3 variants each
# Maximum realism: RAW photo prefix, skin detail, 85mm portrait lens, bokeh
PROMPTS = [
    # ── Expression variants (subtle, closed-mouth) ────────────────────────────
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
 
    # ── Lighting variants ─────────────────────────────────────────────────────
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
 
    # ── Outdoor portrait variants (facing camera) ─────────────────────────────
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
    print(f"Loading {MODEL_ID} (cache: {MODEL_CACHE}) ...")
    pipe = StableDiffusionPipeline.from_pretrained(
        MODEL_ID,
        torch_dtype=torch.float16,
        cache_dir=str(MODEL_CACHE),
        safety_checker=None,          # disabled for research use
        requires_safety_checker=False,
    )
    pipe = pipe.to("cuda")
    pipe.enable_attention_slicing()   # reduces VRAM pressure
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
 
            # deterministic seed: unique per (prompt, seed_index) pair
            seed     = p_idx * 1000 + s_idx
            filename = f"sd15_txt2img_p{p_idx:02d}_s{s_idx:03d}.png"
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
                negative_prompt=NEGATIVE_PROMPT,
                num_inference_steps=NUM_STEPS,
                guidance_scale=CFG_SCALE,
                width=IMAGE_SIZE,
                height=IMAGE_SIZE,
                generator=generator,
            ).images[0]
 
            image.save(IMAGES_DIR / filename)
 
            writer.writerow({
                "filename":        filename,
                "generator":       "sd15",
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
            csv_file.flush()   # write row immediately — safe against crashes
            done += 1
 
    csv_file.close()
    print(f"\n{'─'*50}")
    print(f"Finished.  Generated: {done}  |  Skipped: {skipped}  |  Total: {done+skipped}")
    print(f"Images   → {IMAGES_DIR}")
    print(f"Metadata → {METADATA_PATH}")
 
 
# ── Entry point ────────────────────────────────────────────────────────────────
 
if __name__ == "__main__":
    print("=" * 50)
    print("SD 1.5 txt2img Dataset Generation")
    print("=" * 50)
 
    setup_dirs()
 
    existing = load_existing_metadata()
    if existing:
        print(f"Resuming: {len(existing)} images already exist.\n")
 
    pipe = load_pipeline()
    generate(pipe, existing)