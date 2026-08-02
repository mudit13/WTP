# Legacy scripts

Utilities kept for history only. None of them are called by `run_experiment.py`, are referenced
by an active runbook (`docs/PIPELINE.md`, `docs/SERVER_WORKFLOW.md`, `README.md`,
`docs/README.md`), or produce evidence used by the authoritative run
`2026-08-01_eightway_v1`. They are not deleted outright so the history of how the dataset was
assembled stays inspectable.

| Script | Why it is archived | Superseded by |
|---|---|---|
| `ingest_openforensics.py` | One-off bridge from a flat, un-split OpenForensics crop dump into `real/`/`fake/` subdirectories. | `scripts/extract_openforensics.py`, which writes `real/`/`fake/` directly and records the `source_image_id` group sidecar. |
| `sample_dataset.py` | Manual byte-copy sampler used before per-dataset sampling was config-driven. | `sample_size` in `configs/config.yaml`, applied by `build_master_index.py` (seeded random subset per dataset). |
| `merge_predictions.py` | Concatenated a separate DFFD prediction CSV into `defake_predictions_all.csv`. | `master_metadata.csv` already contains DFFD rows, so `run_defake_batch.py` scores every row in one pass (see docs/PIPELINE.md "Pipeline note"). |
| `ganfp_sweep.py` | One-off CNN channel-width hyperparameter sweep. | Its winning config (`[16, 32, 64]`) is committed directly in `configs/config.yaml` (`ganfp.cnn.channels`). |
| `run_ganfp_infer.py` | Standalone inference helper for an already-trained `ganfp_head.pt`. | The `ganfp` stage in `run_experiment.py` scores GAN-fp end-to-end via `train_ganfp.py` + `benchmark_attribution.py`. |

If you need to run one of these against the current dataset layout, run it from the repository
root exactly as documented in its own docstring (paths and imports still resolve correctly from
`scripts/legacy/`). Do not add new runbook steps that depend on this directory.
