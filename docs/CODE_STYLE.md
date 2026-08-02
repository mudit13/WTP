# Code style note

Short, so it stays followed. This project does not run a formatter/linter in CI; consistency
comes from convention, not tooling.

## Python version

Target **Python 3.9** for everything under `scripts/` (the Magdeburg GPU container's
interpreter). ASCII-only source. GitHub CI runs Python 3.11 for CPU-only smoke tests only - it
does not replace a 3.9 compatibility check, so avoid 3.10+-only syntax (e.g. `match` statements,
`X | Y` union types in annotations).

## Naming

- Modules/functions/variables: `snake_case`. Classes: `PascalCase`.
- Script filenames describe the stage they run (`finetune_defake_head.py`,
  `leave_one_generator_out.py`), not the underlying method name alone.
- Generator names in code/config/CSVs are the exact strings in
  `configs/config.yaml: attribution.fake_generators` / `real_generators` (e.g. `"SD1.5-img2img"`,
  `"StyleGAN3-FFHQ"`). Never introduce a second spelling for an existing generator.

## Docstrings and comments

- Every entry-point script's module docstring states: what it does, what it reads/writes, and
  one runnable example command.
- Prefer **evidence-backed** comments: cite the specific bug, PROJECT_LOG.md section, or
  invariant a piece of code exists to protect, instead of restating what the next line already
  says. A comment earns its place if removing it would let a future edit silently reintroduce a
  past bug (e.g. a leakage, split, or cache-boundary mistake).
- Do not add comments that only narrate control flow ("loop over rows", "return the result").
- Keep comments that explain **why**, not just **what** - especially around: group-aware
  splitting, train/test/val boundaries, which cache is "clean" vs "training-only augmented", and
  anything that was previously a silent leakage bug (see `docs/PROJECT_LOG.md` and
  `docs/REVIEW_SAFEGUARDS.md`).

## Formatting

No mass reformatting passes. Match the surrounding file's existing style (quote style, wrapping,
blank-line conventions) rather than introducing a new one in an otherwise-unrelated change.
