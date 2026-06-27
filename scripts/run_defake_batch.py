"""
run_defake_batch.py
PITSEC SoSe26 Topic 8 - Batch DE-FAKE inference (BINARY real/fake detector).

Loads CLIP, BLIP, and the DE-FAKE classifier ONCE, then loops over every row in
master_metadata.csv running detection on each image. Mirrors De-Fake-patched/test.py
logic exactly, restructured for batch use.

Paths are read from the environment (configs/paths.env) with the original hardcoded
values as defaults, so behaviour is unchanged if the env is not sourced.

Run inside the container (interpreter that has clip+torch+blipmodels = venv_sd15):
    set -a && source configs/paths.env && set +a        # optional; defaults match
    source /pitsec_sose26_topic8/venv_sd15/bin/activate
    python scripts/run_defake_batch.py            # full run
    python scripts/run_defake_batch.py --test     # first 10 images only
"""

import argparse
import csv
import os
import sys
import time

import clip
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms

# ---- PATHS (env-overridable; defaults preserve the original behaviour) ------
WTP_ROOT = os.environ.get("WTP_ROOT", "/pitsec_sose26_topic8")
DEFAKE_DIR = os.environ.get("WTP_DEFAKE_DIR", f"{WTP_ROOT}/De-Fake-patched")
MASTER_CSV = os.environ.get("WTP_MASTER_CSV", f"{WTP_ROOT}/dataset/master_metadata.csv")
OUTPUT_CSV = os.environ.get("WTP_PRED_CSV", f"{WTP_ROOT}/dataset/defake_predictions.csv")
FINETUNE_CLIP_PATH = os.environ.get("WTP_DEFAKE_FINETUNE_CLIP", f"{WTP_ROOT}/models/finetune_clip.pt")
CLIP_LINEAR_PATH = os.environ.get("WTP_DEFAKE_CLIP_LINEAR", f"{WTP_ROOT}/models/clip_linear.pt")
BLIP_URL = os.environ.get(
    "WTP_BLIP_URL",
    "https://storage.googleapis.com/sfr-vision-language-research"
    "/BLIP/models/model_base_capfilt_large.pth",
)
IMAGE_SIZE = 224

# blipmodels is a package inside De-Fake-patched - must be on path
sys.path.insert(0, DEFAKE_DIR)
from blipmodels import blip_decoder  # noqa: E402


# ---- Classifier head -------------------------------------------------------
# Must be defined BEFORE torch.load(CLIP_LINEAR_PATH) since pickle needs this
# class definition to reconstruct the saved object.
class NeuralNet(nn.Module):
    def __init__(self, input_size, hidden_size_list, num_classes):
        super(NeuralNet, self).__init__()
        self.dropout2 = nn.Dropout(0.5)
        self.fc1 = nn.Linear(input_size, hidden_size_list[0])
        self.fc2 = nn.Linear(hidden_size_list[0], hidden_size_list[1])
        self.fc3 = nn.Linear(hidden_size_list[1], num_classes)

    def forward(self, x):
        out = self.fc1(x)
        out = F.relu(out)
        out = self.dropout2(out)
        out = self.fc2(out)
        out = F.relu(out)
        out = self.fc3(out)
        return out


# ---- Model loading ---------------------------------------------------------
def load_models(device):
    print("Loading CLIP ViT-B/32 (for preprocess transform)...")
    _, preprocess = clip.load("ViT-B/32")

    print("Loading BLIP decoder (may use cached weights)...")
    blip = blip_decoder(pretrained=BLIP_URL, image_size=IMAGE_SIZE, vit="base")
    blip.eval()
    blip = blip.to(device)

    # SECURITY: weights_only=False executes pickle and can run arbitrary code on load.
    # Only ever point these at the project's OWN trusted checkpoints under $WTP_ROOT/models
    # (the supervisor-provided clip_linear.pt / finetune_clip.pt). Never a downloaded/untrusted .pt.
    print(f"Loading fine-tuned CLIP from {FINETUNE_CLIP_PATH}...")
    finetuned_clip = torch.load(FINETUNE_CLIP_PATH, map_location=device, weights_only=False).to(device)

    print(f"Loading linear classifier from {CLIP_LINEAR_PATH}...")
    # torch.load reconstructs NeuralNet via pickle - class must be defined above
    linear = torch.load(CLIP_LINEAR_PATH, map_location=device, weights_only=False)
    linear = linear.to(device)
    linear.eval()

    return preprocess, blip, finetuned_clip, linear


# ---- Per-image inference ---------------------------------------------------
def run_inference(img_path, preprocess, blip, finetuned_clip, linear, device):
    # Open once, apply both transforms from the same PIL object
    img_pil = Image.open(img_path).convert("RGB")

    # BLIP side: Resize -> CenterCrop -> ToTensor (mirrors test.py exactly)
    blip_transform = transforms.Compose([
        transforms.Resize(IMAGE_SIZE),
        transforms.CenterCrop(IMAGE_SIZE),
        transforms.ToTensor(),
    ])
    blip_img = blip_transform(img_pil).unsqueeze(0).to(device)
    caption = blip.generate(blip_img, sample=False, num_beams=3, max_length=60, min_length=5)
    text = clip.tokenize(list(caption)).to(device)

    # CLIP side: squash-resize to 224x224, then CLIP preprocess (mirrors test.py)
    clip_img = preprocess(img_pil.resize((IMAGE_SIZE, IMAGE_SIZE)))
    clip_img = clip_img.unsqueeze(0).to(device)

    with torch.no_grad():
        image_features = finetuned_clip.encode_image(clip_img)
        text_features = finetuned_clip.encode_text(text)
        emb = torch.cat((image_features, text_features), 1)
        output = linear(emb.float())
        probs = F.softmax(output, dim=1).cpu().numpy()[0]
        predict = int(output.argmax(1).cpu().numpy()[0])

    return predict, float(probs[0]), float(probs[1]), caption[0]


# ---- Main ------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--test", action="store_true",
                        help="Run on first 10 images only (sanity check)")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    preprocess, blip, finetuned_clip, linear = load_models(device)

    with open(MASTER_CSV, newline="") as f:
        rows = list(csv.DictReader(f))

    if args.test:
        rows = rows[:10]
        print(f"TEST MODE: running on first {len(rows)} images only")
    else:
        print(f"FULL RUN: {len(rows)} images")

    fieldnames = list(rows[0].keys()) + [
        "defake_predict", "prob_real", "prob_fake", "blip_caption"
    ]
    out_rows = []
    errors = 0
    t0 = time.time()

    for i, row in enumerate(rows):
        img_path = row["full_path"]
        try:
            predict, prob_real, prob_fake, caption = run_inference(
                img_path, preprocess, blip, finetuned_clip, linear, device
            )
        except Exception as e:
            print(f"[error] {img_path}: {e}")
            predict, prob_real, prob_fake, caption = -1, None, None, ""
            errors += 1

        row["defake_predict"] = predict
        row["prob_real"] = prob_real
        row["prob_fake"] = prob_fake
        row["blip_caption"] = caption
        out_rows.append(row)

        if (i + 1) % 10 == 0 or (i + 1) == len(rows):
            elapsed = time.time() - t0
            print(f"  {i+1}/{len(rows)} done | {elapsed:.1f}s elapsed | {errors} errors")

    out_path = OUTPUT_CSV.replace(".csv", "_test.csv") if args.test else OUTPUT_CSV
    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(out_rows)

    print(f"\nDone -> {out_path}")
    print(f"Errors: {errors}/{len(rows)}")

    # Quick preview of predictions
    label_map = {0: "real", 1: "fake"}
    print("\nSample predictions:")
    print(f"{'filename':<40} {'true_label':<12} {'predict':<10} {'prob_fake':<10} {'caption'}")
    for r in out_rows[:10]:
        p = int(r["defake_predict"])
        label = label_map.get(p, "error")
        print(f"{r['filename']:<40} {r['label']:<12} {label:<10} "
              f"{r['prob_fake'] or 'N/A':<10} {r['blip_caption'][:50]}")


if __name__ == "__main__":
    main()
