#!/usr/bin/env python3
"""
GAN Fingerprints (Yu2019) verification + scope-note helper.

NOTE: the actual PyTorch reproduction now lives in scripts/lib/ganfp.py +
scripts/train_ganfp.py + scripts/run_ganfp_infer.py (residual/spectrum fingerprints + a
small learned classifier), per GOLD_ALIGNMENT.md / PROJECT_LOG section 5. This script is
retained as a lightweight weight-discovery + reduced-scope-note helper; the reproduction
does not depend on legacy pretrained weights.

WTP.md claimed pretrained GAN-Fingerprints weights live in models/, but the supervisor
email only confirms two DE-FAKE checkpoints there (clip_linear.pt, finetune_clip.pt). This
script first VERIFIES what is actually present, then either:
  - orchestrates inference if compatible weights exist, or
  - emits a documented reduced-scope note (the honest, GOLD-aligned outcome) listing what
    would be required to train GAN-Fingerprints per GAN set on the Titan RTX.

GAN Fingerprints attributes GAN images via learned model-specific residuals. It is expected
to work for GAN sources (StyleGAN3, DFFD) and to be category-mismatched on diffusion images
(SD1.5, FLUX) - that mismatch is documented as designed behavior, not a failure.

Modes:
  verify  : scan the models dir + GANFingerprints repo, report weight availability.
  note    : write results/ganfp_scope_note.md describing the reduced-scope decision.

Usage:
  $WTP_PY_DEFAKE scripts/run_ganfp.py --mode verify
  $WTP_PY_DEFAKE scripts/run_ganfp.py --mode note --out results/ganfp_scope_note.md
"""
import argparse
import glob
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from lib import io_utils  # noqa: E402


# Filenames known to be DE-FAKE (NOT GAN-Fingerprints).
DEFAKE_WEIGHTS = {"clip_linear.pt", "finetune_clip.pt"}
# Heuristic tokens that would indicate a GAN-Fingerprints checkpoint.
GANFP_TOKENS = ("fingerprint", "ganfp", "yu", "stylegan", "progan", "attribution_net")


def verify(logger):
    env = io_utils.load_env()
    root = env.get("WTP_ROOT", "/pitsec_sose26_topic8")
    models_dir = os.path.join(root, "models")
    ganfp_dir = env.get("WTP_GANFP_DIR", "/workspace/GANFingerprints")

    found = sorted(os.path.basename(p) for p in glob.glob(os.path.join(models_dir, "*"))
                   if os.path.isfile(p))
    logger.info("models/ contents: %s", found or "(empty or missing)")

    candidates = [f for f in found
                  if f not in DEFAKE_WEIGHTS
                  and any(tok in f.lower() for tok in GANFP_TOKENS)]
    # Also scan the read-only repo for bundled checkpoints.
    repo_ckpts = []
    if os.path.isdir(ganfp_dir):
        for ext in ("*.pt", "*.pth", "*.pkl", "*.ckpt", "*.npz"):
            repo_ckpts += glob.glob(os.path.join(ganfp_dir, "**", ext), recursive=True)
    logger.info("GANFingerprints repo checkpoints: %s",
                [os.path.relpath(p, ganfp_dir) for p in repo_ckpts] or "(none)")

    if candidates or repo_ckpts:
        logger.info("LIKELY GAN-Fingerprints weights found -> proceed to inference.")
        return True
    logger.warning("No GAN-Fingerprints weights found. models/ holds only DE-FAKE "
                   "checkpoints (%s). Use --mode note and raise item #3 with the supervisor.",
                   sorted(DEFAKE_WEIGHTS))
    return False


def write_note(out_path, logger):
    text = """# GAN Fingerprints (Yu2019) - scope note

## Finding
The supervisor email placed only DE-FAKE checkpoints in models/
(clip_linear.pt, finetune_clip.pt). No pretrained GAN-Fingerprints classifier was provided,
and the read-only /workspace/GANFingerprints repo ships training code, not ready weights.

## Implication
GAN Fingerprints attributes images by a learned, model-specific residual. Without
pretrained weights it must be TRAINED per GAN set. Training requires:
- A labeled set per source GAN (e.g. StyleGAN3, plus the DFFD GAN families) and a real set.
- Yu2019's preprocessing (fixed crop/resize) and their attribution-network training loop.
- GPU time on the single Titan RTX (24 GB).

## Decision (pending the supervisor; see docs/GOLD_ALIGNMENT.md, GAN-Fingerprints note)
Option A (preferred if time allows): train GAN-Fingerprints on StyleGAN3 + DFFD GAN classes
+ a real class, then evaluate attribution; expect strong results on GANs and category
mismatch on diffusion (SD1.5/FLUX), which we report as designed behavior.

Option B (reduced scope): if training is infeasible, document GAN-Fingerprints as a
literature baseline only and rely on DE-FAKE (semantic) attribution for the empirical
results. Clearly state the limitation in the report.

## Expected behavior either way
- GAN sources (StyleGAN3, DFFD): meaningful residual attribution.
- Diffusion sources (SD1.5, FLUX): forced/uncertain - a feature-space mismatch, not a bug.
"""
    io_utils.ensure_dir(os.path.dirname(os.path.abspath(out_path)))
    with open(out_path, "w", encoding="utf-8") as fh:
        fh.write(text)
    logger.info("Wrote scope note to %s", out_path)


def main(args):
    logger = io_utils.setup_logging("run_ganfp")
    if args.mode == "verify":
        verify(logger)
    elif args.mode == "note":
        write_note(args.out, logger)
    else:
        raise SystemExit("Unknown mode")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="GAN Fingerprints verify/orchestrate.")
    parser.add_argument("--mode", choices=["verify", "note"], default="verify")
    parser.add_argument("--out", default="results/ganfp_scope_note.md")
    main(parser.parse_args())
