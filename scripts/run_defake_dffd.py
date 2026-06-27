"""
run_defake_dffd.py
PITSEC SoSe26 Topic 8 - DE-FAKE inference restricted to the DFFD rows.

Identical model/inference logic to run_defake_batch.py, but only scores rows whose
source_dataset starts with "dffd_". Kept separate so the (large) DFFD pass can be run
independently and merged later with merge_predictions.py.

Paths are env-overridable (configs/paths.env) with the original defaults preserved.

Run inside the container:
    source /pitsec_sose26_topic8/venv_sd15/bin/activate
    python scripts/run_defake_dffd.py
"""

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

WTP_ROOT = os.environ.get("WTP_ROOT", "/pitsec_sose26_topic8")
DEFAKE_DIR = os.environ.get("WTP_DEFAKE_DIR", f"{WTP_ROOT}/De-Fake-patched")
MASTER_CSV = os.environ.get("WTP_MASTER_CSV", f"{WTP_ROOT}/dataset/master_metadata.csv")
OUTPUT_CSV = os.environ.get("WTP_PRED_DFFD_CSV", f"{WTP_ROOT}/dataset/defake_predictions_dffd.csv")
FINETUNE_CLIP_PATH = os.environ.get("WTP_DEFAKE_FINETUNE_CLIP", f"{WTP_ROOT}/models/finetune_clip.pt")
CLIP_LINEAR_PATH = os.environ.get("WTP_DEFAKE_CLIP_LINEAR", f"{WTP_ROOT}/models/clip_linear.pt")
BLIP_URL = os.environ.get(
    "WTP_BLIP_URL",
    "https://storage.googleapis.com/sfr-vision-language-research"
    "/BLIP/models/model_base_capfilt_large.pth",
)
IMAGE_SIZE = 224

sys.path.insert(0, DEFAKE_DIR)
from blipmodels import blip_decoder  # noqa: E402


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


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    _, preprocess = clip.load("ViT-B/32")
    blip = blip_decoder(pretrained=BLIP_URL, image_size=IMAGE_SIZE, vit="base")
    blip.eval()
    blip = blip.to(device)
    finetuned_clip = torch.load(FINETUNE_CLIP_PATH, map_location=device, weights_only=False).to(device)
    linear = torch.load(CLIP_LINEAR_PATH, map_location=device, weights_only=False)
    linear = linear.to(device)
    linear.eval()

    with open(MASTER_CSV, newline="") as f:
        all_rows = list(csv.DictReader(f))

    rows = [r for r in all_rows if r["source_dataset"].startswith("dffd_")]
    print(f"Running inference on {len(rows)} DFFD images...")

    fieldnames = list(rows[0].keys()) + ["defake_predict", "prob_real", "prob_fake", "blip_caption"]
    out_rows = []
    errors = 0
    t0 = time.time()

    for i, row in enumerate(rows):
        try:
            img_pil = Image.open(row["full_path"]).convert("RGB")
            blip_transform = transforms.Compose([
                transforms.Resize(IMAGE_SIZE),
                transforms.CenterCrop(IMAGE_SIZE),
                transforms.ToTensor(),
            ])
            blip_img = blip_transform(img_pil).unsqueeze(0).to(device)
            caption = blip.generate(blip_img, sample=False, num_beams=3, max_length=60, min_length=5)
            text = clip.tokenize(list(caption)).to(device)
            clip_img = preprocess(img_pil.resize((IMAGE_SIZE, IMAGE_SIZE))).unsqueeze(0).to(device)
            with torch.no_grad():
                image_features = finetuned_clip.encode_image(clip_img)
                text_features = finetuned_clip.encode_text(text)
                emb = torch.cat((image_features, text_features), 1)
                output = linear(emb.float())
                probs = F.softmax(output, dim=1).cpu().numpy()[0]
                predict = int(output.argmax(1).cpu().numpy()[0])
            row["defake_predict"] = predict
            row["prob_real"] = float(probs[0])
            row["prob_fake"] = float(probs[1])
            row["blip_caption"] = caption[0]
        except Exception as e:
            print(f"[error] {row['full_path']}: {e}")
            row["defake_predict"] = -1
            row["prob_real"] = row["prob_fake"] = None
            row["blip_caption"] = ""
            errors += 1
        out_rows.append(row)
        if (i + 1) % 50 == 0 or (i + 1) == len(rows):
            print(f"  {i+1}/{len(rows)} done | {time.time()-t0:.1f}s | {errors} errors")

    with open(OUTPUT_CSV, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(out_rows)

    print(f"Done -> {OUTPUT_CSV}")
    print(f"Errors: {errors}/{len(rows)}")


if __name__ == "__main__":
    main()
