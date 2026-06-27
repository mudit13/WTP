#!/usr/bin/env python3
"""
StyleGAN3-FFHQ Seed-based Generation Script
PITSEC SoSe26 - Topic 8: AI Image Detection & Attribution
Author: Mudit

Generates >=100 face images from StyleGAN3-FFHQ for DE-FAKE evaluation.
No text prompts — purely seed-based GAN generation.
Output: /pitsec_sose26_topic8/dataset/stylegan3/

Usage (inside Docker):
    source /pitsec_sose26_topic8/venv_stylegan3/bin/activate
    python3.9 /pitsec_sose26_topic8/generate_stylegan3.py

One-time setup before running:
    python3.9 -m virtualenv /pitsec_sose26_topic8/venv_stylegan3
    source /pitsec_sose26_topic8/venv_stylegan3/bin/activate
    pip install torch==1.12.1+cu113 torchvision==0.13.1+cu113 --extra-index-url https://download.pytorch.org/whl/cu113
    pip install numpy==1.23.1 pillow requests ninja scipy
    git clone https://github.com/NVlabs/stylegan3.git /pitsec_sose26_topic8/stylegan3
"""

import sys
import os

# StyleGAN3 repo MUST be on sys.path before ANY other imports
# because pickle.load needs to find StyleGAN3's custom classes during deserialization
STYLEGAN3_REPO = "/pitsec_sose26_topic8/stylegan3"
if not os.path.exists(STYLEGAN3_REPO):
    print(f"ERROR: StyleGAN3 repo not found at {STYLEGAN3_REPO}")
    print("Run this first:")
    print(f"  git clone https://github.com/NVlabs/stylegan3.git {STYLEGAN3_REPO}")
    sys.exit(1)
if STYLEGAN3_REPO not in sys.path:
    sys.path.insert(0, STYLEGAN3_REPO)

import torch

# Compatibility patch: same pattern as Sushmita's SD1.5 script
class _DeviceMock:
    def __getattr__(self, name):
        return lambda *args, **kwargs: None
    def is_available(self): return False
    def device_count(self): return 0
for _dev in ["xpu", "mps", "npu", "mlu", "musa"]:
    if not hasattr(torch, _dev):
        setattr(torch, _dev, _DeviceMock())

import numpy as np
import pickle
import csv
import requests
from pathlib import Path
from datetime import datetime
from PIL import Image

# ── Configuration ──────────────────────────────────────────────────────────────

OUTPUT_DIR    = Path("/pitsec_sose26_topic8/dataset/stylegan3")
IMAGES_DIR    = OUTPUT_DIR / "images"
METADATA_PATH = OUTPUT_DIR / "metadata.csv"
MODEL_CACHE   = Path("/pitsec_sose26_topic8/models")

# Official NVIDIA pretrained StyleGAN3-r FFHQ 1024x1024 weights
MODEL_URL  = "https://api.ngc.nvidia.com/v2/models/nvidia/research/stylegan3/versions/1/files/stylegan3-r-ffhq-1024x1024.pkl"
MODEL_PATH = MODEL_CACHE / "stylegan3-r-ffhq-1024x1024.pkl"

IMAGE_SIZE     = 512    # resize from native 1024 to 512 to match SD1.5 and FLUX.1
NUM_IMAGES     = 108    # matches Sushmita's 108 and Vishnu's 108
SEED_START     = 0      # seeds 0 to 107
TRUNCATION_PSI = 0.7    # 0.7 = good balance of diversity vs quality (standard for FFHQ)

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


def download_model():
    """Download StyleGAN3 weights if not already cached."""
    if MODEL_PATH.exists():
        size_mb = MODEL_PATH.stat().st_size / 1_000_000
        print(f"Model already cached at {MODEL_PATH} ({size_mb:.0f}MB)")
        return

    print(f"Downloading StyleGAN3-FFHQ weights (~350MB) ...")
    response = requests.get(MODEL_URL, stream=True)
    if response.status_code != 200:
        print(f"\nERROR: Download failed (HTTP {response.status_code})")
        print("Download manually from:")
        print("  https://catalog.ngc.nvidia.com/orgs/nvidia/teams/research/models/stylegan3")
        print(f"Place the .pkl file at: {MODEL_PATH}")
        sys.exit(1)

    total      = int(response.headers.get("content-length", 0))
    downloaded = 0
    with open(MODEL_PATH, "wb") as f:
        for chunk in response.iter_content(chunk_size=8192):
            f.write(chunk)
            downloaded += len(chunk)
            if total:
                pct = downloaded / total * 100
                print(f"\r  {pct:.1f}%  ({downloaded//1_000_000}MB / {total//1_000_000}MB)", end="")
    print(f"\nDownload complete → {MODEL_PATH}")


def load_model():
    """Load StyleGAN3 generator from pickle."""
    print(f"Loading StyleGAN3 from {MODEL_PATH} ...")
    with open(MODEL_PATH, "rb") as f:
        data = pickle.load(f)

    # pkl is a dict with keys: G, G_ema, D, training_set_kwargs
    # G_ema = exponential moving average of weights = best image quality
    G = data["G_ema"].cuda()
    G.eval()
    print(f"Generator loaded.")
    print(f"  z_dim : {G.z_dim}")
    print(f"  c_dim : {G.c_dim}  (0 = unconditional, no class label needed)")
    print(f"  Native resolution: {G.img_resolution}x{G.img_resolution}")
    print(f"  Output will be resized to: {IMAGE_SIZE}x{IMAGE_SIZE}")
    return G


# ── Generation ─────────────────────────────────────────────────────────────────

def generate(G, existing_files):
    total   = NUM_IMAGES
    skipped = 0
    done    = 0

    write_header = not METADATA_PATH.exists()
    csv_file = open(METADATA_PATH, "a", newline="")
    writer   = csv.DictWriter(csv_file, fieldnames=[
        "filename", "generator", "mode",
        "seed", "truncation_psi", "width", "height", "timestamp",
    ])
    if write_header:
        writer.writeheader()

    for i in range(NUM_IMAGES):
        seed     = SEED_START + i
        filename = f"stylegan3_ffhq_s{seed:04d}.png"
        count    = i + 1

        if filename in existing_files:
            print(f"[{count}/{total}] SKIP {filename} (already exists)")
            skipped += 1
            continue

        print(f"[{count}/{total}] Generating {filename}  (seed {seed})")

        # Deterministic latent vector from seed
        z = torch.from_numpy(
            np.random.RandomState(seed).randn(1, G.z_dim)
        ).cuda().float()

        # Unconditional generation — FFHQ has c_dim=0 so label is empty
        with torch.no_grad():
            # c_dim=0 means no class conditioning — pass zeros
            label = torch.zeros([1, G.c_dim], device="cuda")
            img   = G(z, label, truncation_psi=TRUNCATION_PSI, noise_mode="const")

        # img shape: [1, 3, H, W], range [-1, 1]
        # Convert to [H, W, 3] uint8
        img     = (img.permute(0, 2, 3, 1) * 127.5 + 128).clamp(0, 255).to(torch.uint8)
        pil_img = Image.fromarray(img[0].cpu().numpy(), "RGB")

        # Resize from 1024x1024 to 512x512 to match SD1.5 and FLUX.1
        pil_img = pil_img.resize((IMAGE_SIZE, IMAGE_SIZE), Image.LANCZOS)
        pil_img.save(IMAGES_DIR / filename)

        writer.writerow({
            "filename":       filename,
            "generator":      "stylegan3-r-ffhq",
            "mode":           "seed-based",
            "seed":           seed,
            "truncation_psi": TRUNCATION_PSI,
            "width":          IMAGE_SIZE,
            "height":         IMAGE_SIZE,
            "timestamp":      datetime.now().isoformat(),
        })
        csv_file.flush()    # write immediately — safe against crashes
        done += 1

    csv_file.close()
    print(f"\n{'─'*50}")
    print(f"Finished.  Generated: {done}  |  Skipped: {skipped}  |  Total: {done+skipped}")
    print(f"Images   → {IMAGES_DIR}")
    print(f"Metadata → {METADATA_PATH}")


# ── Entry point ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 50)
    print("StyleGAN3-FFHQ Dataset Generation")
    print("=" * 50)

    setup_dirs()

    existing = load_existing_metadata()
    if existing:
        print(f"Resuming: {len(existing)} images already exist.\n")

    download_model()
    G = load_model()
    generate(G, existing)