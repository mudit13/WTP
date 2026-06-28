# Python environments (venvs)

There are three virtual environments on the server, one per generator. DE-FAKE inference and
all analysis scripts in this repo run inside **`venv_sd15`** (it already has `clip`, `torch`,
`torchvision`, and the `blipmodels` package used by DE-FAKE).

| venv | Interpreter (env var) | Used for |
|------|-----------------------|----------|
| `venv_sd15` | `$WTP_PY_SD15` / `$WTP_PY_DEFAKE` | SD1.5 generation, DE-FAKE inference, **this repo's scripts** |
| `venv_flux1` | `$WTP_PY_FLUX1` | FLUX.1 generation only |
| `venv_stylegan3` | `$WTP_PY_STYLEGAN3` | StyleGAN3 generation only |

## Will the venvs still work after the repo reorg? Yes.

The reorg only moves *tracked code* (scripts). It does not touch the venvs because:

1. **venvs are git-ignored** (`venv_sd15/`, `venv_flux1/`, `venv_stylegan3/`), so `git pull`
   never moves or deletes them. They stay at `/pitsec_sose26_topic8/venv_*`.
2. **A venv is activated by its absolute path** (`source /pitsec_sose26_topic8/venv_sd15/bin/activate`)
   and is completely independent of where the script file lives. Moving scripts into `scripts/`
   has no effect on the interpreter.
3. **The repo root stays mapped to `/pitsec_sose26_topic8`**, so every absolute path the
   scripts and venvs rely on remains valid.

## The one rule: a venv is NOT relocatable

A Python venv hardcodes its own absolute path inside `pyvenv.cfg`, `bin/activate`, and the
shebangs of console scripts (e.g. `pip`). Therefore:

- Do **not** move a venv directory, and never `git`-commit one (a committed venv carries
  another machine's absolute shebangs and breaks on checkout). `.gitignore` enforces this.
- If the container is ever reset, or the project root path changes, **recreate** the venvs
  rather than copying them.

## Recreating the venvs (only if the container is reset)

```bash
cd /pitsec_sose26_topic8

# StyleGAN3 (Python 3.9, torch 1.12.1 + cu113) - see header of scripts/generate_stylegan3.py
python3.9 -m virtualenv venv_stylegan3
source venv_stylegan3/bin/activate
pip install torch==1.12.1+cu113 torchvision==0.13.1+cu113 \
    --extra-index-url https://download.pytorch.org/whl/cu113
pip install numpy==1.23.1 pillow requests ninja scipy
git clone https://github.com/NVlabs/stylegan3.git /pitsec_sose26_topic8/stylegan3
deactivate

# SD1.5 / DE-FAKE / analysis (the env this repo runs in)
python3.9 -m virtualenv venv_sd15
source venv_sd15/bin/activate
pip install torch torchvision diffusers transformers ftfy regex
pip install git+https://github.com/openai/CLIP.git
pip install -r requirements.txt            # this repo's analysis deps (numpy/pandas/sklearn/...)
deactivate

# FLUX.1 - see header of scripts/generate_flux1_txt2img.py for its exact pins
python3.9 -m virtualenv venv_flux1
# ... install per that script's documented requirements ...
```

After recreating, `cp configs/paths.example.env configs/paths.env` (the defaults already point
at `/pitsec_sose26_topic8/venv_*`).

## Capture exact venv pins (do this once, on the server)

The install commands above are approximate. To make the environments truly reproducible,
freeze the *actual* installed versions from each venv and commit the lock files:

```bash
cd /pitsec_sose26_topic8
for v in sd15 flux1 stylegan3; do
  ./venv_$v/bin/python -m pip freeze > requirements-$v.lock
done
git add requirements-*.lock && git commit -m "chore: capture exact venv pins"
```

These `requirements-*.lock` files ARE committed (unlike the venvs themselves) so anyone can
rebuild the exact environment. Note they include the CUDA-tagged torch builds, so install with
the matching `--extra-index-url https://download.pytorch.org/whl/cuXXX`.

## Installing more packages later (safely)

You can install more packages any time - it will NOT harm the container. Understand where it
lands:

- The venvs live on the **mounted volume** (`/pitsec_sose26_topic8/venv_*` = `sharedDockerDir`
  on the host), NOT inside the Docker image. So `venv` installs **persist** across container
  restarts, are **isolated** from the image, and are **shared** with the whole team.
- Installing into **system** Python (`sudo pip`, or plain `pip` outside a venv) writes to the
  container's throwaway layer: lost on restart and can affect others. **Never do this.**

Rules when adding a package:

1. **Install into the venv**, never with sudo:
   ```bash
   $WTP_PY_DEFAKE -m pip install <pkg>     # = venv_sd15/bin/python3 -m pip
   ```
2. **Never let it upgrade torch or numpy** (breaks the CUDA build). Pin numpy and verify:
   ```bash
   PY=$WTP_PY_DEFAKE
   NUMPY_VER=$($PY -c "import numpy; print(numpy.__version__)")
   $PY -m pip install "numpy==$NUMPY_VER" <pkg>
   $PY -c "import torch; print(torch.__version__)"   # expect 2.1.0+cu118, unchanged
   ```
   For anything that lists torch as a dependency (e.g. OpenAI CLIP), use `--no-deps` and add
   its light deps explicitly.
3. **Document it**: add the package to `requirements.txt` in the same change so the team and
   future recovery know it is needed.
4. **Re-freeze after the env stabilizes** (not after every install): `pip freeze >
   requirements-sd15.lock`, then commit. The lock is recovery insurance, not a lock-out.

### CLIP stack (added manually; the image did NOT ship it)

DE-FAKE needs OpenAI CLIP, which was not pre-installed. It was added to `venv_sd15` as:
```bash
PY=$WTP_PY_DEFAKE
$PY -m pip install --no-deps ftfy regex wcwidth      # CLIP's light deps (no torch pull)
$PY -m pip install --no-deps git+https://github.com/openai/CLIP.git
$PY -c "import clip, torch; print('clip OK | torch', torch.__version__)"
```
`--no-deps` is essential: a plain `pip install` of CLIP would try to (re)install torch and can
break the pinned `2.1.0+cu118` build.
