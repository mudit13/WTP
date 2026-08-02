# WTP Topic 8 - AI Image Detection & Attribution

Unified repository for the project (github.com/mudit13/WTP): the generation/inference pipeline
that runs on the GPU container, and the analysis/experiments layer (DCT linear-SVM detector,
fine-tuned DE-FAKE attribution head, out-of-set generalization, robustness) built on top of it.

Code is authored locally and executed inside the GPU container. The repo root maps to the
container project root (`/pitsec_sose26_topic8`, i.e. it *is* `sharedDockerDir`); large data,
model weights, and venvs live alongside the code but are git-ignored (see `.gitignore`).

## Start here

- **New to the project or receiving this handover?** Read **`HANDOVER.md`** first.
- **Full documentation map:** `docs/README.md` - lists every doc, its role, and the authority
  order to use when documents disagree.
- **Project design and research questions:** `docs/IMPLEMENTATION_GUIDE.md`.
- **The only executable server runbook:** `docs/PIPELINE.md`.

## Layout

```
configs/            config.yaml + paths.example.env (copy to paths.env; never committed)
scripts/            All Python entry points (argparse CLIs, Python 3.9)
scripts/legacy/     Archived one-off utilities, not used by any active runbook
scripts/lib/        Shared package: schema, metrics, io, image ops, clip/features, head
De-Fake-patched/    Vendored DE-FAKE code (blipmodels package + test.py/train.py)
docs/               Documentation map, runbook, safeguards, provenance, project log
report/             Report outline and authoritative results draft
results/  logs/     Generated outputs (git-ignored except .gitkeep)
--- not in git (live alongside on the server, git-ignored) ---
dataset/  models/  venv_sd15/  venv_flux1/  venv_stylegan3/  stylegan3/
```

The authoritative GPU runtime is Python 3.9. GitHub CI intentionally uses Python 3.11 for
CPU-only compatibility smoke tests; it does not reproduce the CUDA/DE-FAKE environment.
