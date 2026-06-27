"""
CLIP embedding extraction for DE-FAKE-style attribution.

DE-FAKE builds its classifier on a 1024-dim embedding = CLIP image features (512) concat
CLIP text features (512) of a BLIP caption (see De-Fake-patched/train.py and
run_defake_batch.py). For the head fine-tuning (Phase E) we freeze the CLIP backbone and
learn only a small MLP head. This module supports:
  - image-only 512-dim features (lightweight default), and
  - faithful 1024-dim image+text features when BLIP captions are supplied (the team's
    run_defake_batch.py already emits a blip_caption per image, so we can reuse them).

Run with the DE-FAKE interpreter (venv_sd15 on the server: it has clip + torch).
ASCII-only; Python 3.9.
"""
from typing import List, Optional, Tuple

import numpy as np


def get_clip(model_name: str = "ViT-B/32", device: str = "cuda"):
    """Load the OpenAI CLIP model + preprocess transform.

    DE-FAKE uses OpenAI CLIP (ViT-B/32). We import lazily so non-CLIP scripts do not pay
    the torch import cost and so this file can be linted off-server.
    """
    try:
        import clip  # OpenAI CLIP, installed on the container system python
        import torch
    except ImportError as exc:  # pragma: no cover
        raise SystemExit(
            "CLIP/torch not importable. Use /usr/bin/python3.9 inside the container; "
            "do not run this from venv_sd15."
        ) from exc

    if device == "cuda" and not torch.cuda.is_available():
        device = "cpu"
    model, preprocess = clip.load(model_name, device=device)
    model.eval()
    return model, preprocess, device


def extract_features(image_paths: List[str],
                     model,
                     preprocess,
                     device: str,
                     batch_size: int = 64,
                     captions: Optional[List[str]] = None,
                     augment=None) -> Tuple[np.ndarray, List[str]]:
    """Return L2-normalized CLIP embeddings for the given paths.

    If `captions` is provided (aligned 1:1 with image_paths), the output is the faithful
    DE-FAKE 1024-dim image+text embedding; otherwise it is the 512-dim image embedding.

    If `augment` is given, it is a callable(img, path) -> img applied to each loaded RGB image
    before CLIP preprocessing (used for training-time JPEG augmentation).

    Returns (features [N, D] float32, kept_paths). Unreadable images are skipped and
    excluded from kept_paths so the caller can keep labels aligned.
    """
    import clip
    import torch
    from PIL import Image

    use_text = captions is not None
    feats = []
    kept: List[str] = []
    batch_tensors = []
    batch_caps: List[str] = []
    batch_paths: List[str] = []

    def _flush():
        if not batch_tensors:
            return
        with torch.no_grad():
            stack = torch.stack(batch_tensors).to(device)
            img_emb = model.encode_image(stack)
            img_emb = img_emb / img_emb.norm(dim=-1, keepdim=True)
            if use_text:
                tokens = clip.tokenize([c[:300] for c in batch_caps], truncate=True).to(device)
                txt_emb = model.encode_text(tokens)
                txt_emb = txt_emb / txt_emb.norm(dim=-1, keepdim=True)
                emb = torch.cat((img_emb, txt_emb), dim=1)
            else:
                emb = img_emb
            feats.append(emb.cpu().float().numpy())
        kept.extend(batch_paths)
        batch_tensors.clear()
        batch_caps.clear()
        batch_paths.clear()

    for i, path in enumerate(image_paths):
        try:
            img = Image.open(path).convert("RGB")
        except Exception:  # noqa: BLE001 - skip unreadable files, keep the run going
            continue
        if augment is not None:
            img = augment(img, path)
        batch_tensors.append(preprocess(img))
        batch_caps.append(captions[i] if use_text else "")
        batch_paths.append(path)
        if len(batch_tensors) >= batch_size:
            _flush()
    _flush()

    dim = 1024 if use_text else 512
    if not feats:
        return np.zeros((0, dim), dtype=np.float32), kept
    return np.concatenate(feats, axis=0).astype(np.float32), kept
