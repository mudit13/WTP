# Dev-here / run-on-server workflow

Code is authored locally (this repo) and executed inside the GPU container. Nothing here
hardcodes absolute paths or assumes a GUI.

## 1. Connect

```bash
# Connect EduVPN first, then:
ssh pitsec_sose26_topic8@gensynth.cs.uni-magdeburg.de
sudo pitsec_sose26_topic8.docker PITSEC26   # enter container
```

Host <-> container path mapping:

- Host  `/vol2/pitsec_sose26_topic8/sharedDockerDir/`
- Container `/pitsec_sose26_topic8/`

## 2. Real server layout (as of latest sync)

The repo root **is** `/pitsec_sose26_topic8` (= `sharedDockerDir`). Tracked code lives in
`scripts/`, `scripts/lib/`, `configs/`, `docs/`, `De-Fake-patched/`. Large data, weights, and
venvs live in the same dir but are git-ignored.

```
/pitsec_sose26_topic8/                 (= sharedDockerDir = repo root)
├── scripts/                           generate_*, run_defake_*, build_master_index, analysis
│   └── lib/                           schema, metrics, io, image ops, clip/features, head
├── De-Fake-patched/                   (blipmodels package + test.py/train.py)
├── configs/  docs/  report/  results/  logs/
│   ── git-ignored (present on server, not committed) ──
├── dataset/
│   ├── sd15_txt2img/images/           *.png  (fake, SD1.5, near_in_set)
│   ├── flux1_txt2img/images/          *.png  (fake, FLUX.1-schnell, out_of_set)
│   ├── stylegan3/images/              *.png  (fake, StyleGAN3-FFHQ, out_of_set)
│   ├── londondb/neutral_front/neutral_front/  *.jpg (real, London-DB)  <- narrow
│   ├── master_metadata.csv
│   ├── defake_predictions.csv / _dffd.csv / _all.csv
│   └── (variants/, robust/  <- created by this repo)
├── models/                            clip_linear.pt, finetune_clip.pt  (DE-FAKE, BINARY)
├── venv_sd15/  venv_flux1/  venv_stylegan3/
└── stylegan3/                         (StyleGAN3 code)

/share/DeepFake/DFFD_Images/<model>/test/   ffhq(real), pggan_v1, pggan_v2, stargan, faceapp
```

## 3. Get the repo onto the server + set env

Clone the repo **as** the project root so absolute paths and venvs keep working:

```bash
cd /pitsec_sose26_topic8
# first time only, into the existing dir (or git pull if already a clone):
git clone https://github.com/mudit13/WTP.git .    # data/models/venvs are git-ignored, untouched
cp configs/paths.example.env configs/paths.env     # already points at container paths
```

Re-importing later is just `git pull` from `/pitsec_sose26_topic8` - it moves only tracked
code; `dataset/`, `models/`, and `venv_*` are git-ignored and stay in place. See
`docs/ENVIRONMENTS.md` before recreating any venv.

## 4. Interpreters (do not mix)

Three generation venvs, one per generator. DE-FAKE inference and this repo's analysis
scripts run inside `venv_sd15` (it has clip + torch + blipmodels), matching the team's
working run_defake_batch.py.

```bash
export $(grep -v '^#' configs/paths.env | xargs)   # load WTP_* vars
$WTP_PY_DEFAKE   -> venv_sd15/bin/python3   (DE-FAKE + this repo)
$WTP_PY_FLUX1    -> venv_flux1/bin/python3  (FLUX generation only)
$WTP_PY_STYLEGAN3-> venv_stylegan3/bin/python3 (StyleGAN3 generation only)
```

Never use bare `python`.

## 5. Standard run pattern

```bash
$WTP_PY_DEFAKE scripts/<entry>.py --config configs/config.yaml ... \
    2>&1 | tee logs/run_$(date +%Y%m%d_%H%M%S).log
```

## 6. Data hygiene before every batch

```bash
find /pitsec_sose26_topic8 -name "._*" -o -name ".DS_Store" | wc -l    # must be 0
$WTP_PY_DEFAKE scripts/build_master_index.py --config configs/config.yaml \
    --out /pitsec_sose26_topic8/dataset/master_metadata.csv
```

## 7. Coding standards (enforced)

- Python 3.9; ASCII-only source (no box-drawing chars - they corrupt over SSH paste).
- LF line endings; argparse CLIs; no hardcoded absolute paths (use config + paths.env).
- Test on a 10-image subset before full-dataset runs (run_defake_batch.py has --test).
