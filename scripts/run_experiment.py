#!/usr/bin/env python3
"""
One-command experiment orchestrator for the WTP Topic 8 pipeline.

This is a THIN wrapper over the existing per-stage scripts documented in docs/PIPELINE.md -
it introduces no new science, it just runs the stages in the right order with consistent
variant / jpeg-aug / path naming so a single command reproduces a full run. Every stage still
shells out to the same script you would call by hand, so behaviour matches the runbook exactly.

Interpreter: the sub-scripts need the DE-FAKE venv (CLIP + torch). By default we invoke them
with $WTP_PY_DEFAKE (falling back to this interpreter), matching PIPELINE.md.

Examples:
  # headline confound-controlled run (aspect variant, JPEG-aug on), just print the plan:
  python scripts/run_experiment.py --dry_run
  # actually run only the attribution + out-of-set stages on the aspect variant:
  python scripts/run_experiment.py --stages attribution,oos
  # raw geometry baseline (scaled variant, no JPEG aug) for the confound comparison:
  python scripts/run_experiment.py --variant scaled --jpeg_aug off --stages attribution,oos
  # include the heavy stages too:
  python scripts/run_experiment.py --stages index,variants,confound,detect,dct,attribution,oos,ganfp,robustness,aggregate
"""
import argparse
import os
import subprocess
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from lib import io_utils  # noqa: E402

SCRIPTS = os.path.dirname(os.path.abspath(__file__))

# Perturbations must match configs/config.yaml robustness block (and PIPELINE.md step 7).
PERTURBATIONS = ["jpeg30", "jpeg50", "jpeg70", "blur1", "blur2",
                 "resize0.5", "resize0.75", "sharpen1"]

ALL_STAGES = ["index", "variants", "confound", "detect", "dct",
              "attribution", "oos", "ganfp", "robustness", "aggregate"]
# Default = the confound-controlled headline path. ganfp + robustness are heavy -> opt in.
DEFAULT_STAGES = ["index", "variants", "confound", "detect", "dct",
                  "attribution", "oos", "aggregate"]


class Ctx:
    """Resolved paths / naming shared by every stage builder."""

    def __init__(self, args):
        self.py = args.python or os.environ.get("WTP_PY_DEFAKE") or sys.executable
        self.cfg = args.config
        self.variant = args.variant
        self.jpeg_aug = args.jpeg_aug            # "on" | "off"
        self.device = args.device
        self.results = args.results_dir.rstrip("/")
        root = os.environ.get("WTP_ROOT", "/pitsec_sose26_topic8")
        self.ds = (args.dataset_dir or os.path.join(root, "dataset")).rstrip("/")

        self.augtag = "jpegaug" if self.jpeg_aug == "on" else "raw"
        self.index = f"{self.results}/index_{self.variant}.csv"
        self.master = f"{self.ds}/master_metadata.csv"
        self.pred = f"{self.ds}/defake_predictions_{self.variant}.csv"
        # Captions source for the faithful DE-FAKE 1024-dim image+text features. Prefer an
        # explicit override, then a project-wide merged file if it exists, else the same-variant
        # detect output (created by the `detect` stage; its full_paths match this index, so
        # captions join cleanly). This avoids hard-coding a `defake_predictions_all.csv` that
        # may not exist on a fresh setup (which crashed the fine-tune while reading it).
        all_captions = f"{self.ds}/defake_predictions_all.csv"
        self.captions = args.captions_csv or (
            all_captions if os.path.exists(all_captions) else self.pred)
        self.feats = f"{self.results}/clip_feats_{self.variant}_{self.augtag}.npz"
        self.finetune_out = f"{self.results}/finetune_{self.variant}_{self.augtag}/"
        self.attr_eval_out = f"{self.results}/attr_eval_{self.variant}/"
        dctaug = "_jpegaug" if self.jpeg_aug == "on" else ""
        self.dct_feats = f"{self.results}/dct_features_{self.variant}{dctaug}.npz"
        self.dct_svm_out = f"{self.results}/dct_svm_{self.variant}/"
        self.robust_dir = f"{self.results}/robust"

    def s(self, name):
        return os.path.join(SCRIPTS, name)


def _step(desc, cmd, env=None):
    return {"desc": desc, "cmd": cmd, "env": env}


def stage_index(c):
    return [
        _step("Build master index", [c.py, c.s("build_master_index.py"), "--config", c.cfg,
              "--out", c.master, "--reconcile", f"{c.ds}/defake_predictions_all.csv"]),
        _step("Datasheets", [c.py, c.s("make_datasheets.py"), "--metadata", c.master,
              "--out", f"{c.results}/datasheets.md"]),
    ]


def stage_variants(c):
    return [_step("Preprocessing variants", [c.py, c.s("prepare_variants.py"), "--config", c.cfg,
            "--master", c.master, "--out_root", f"{c.ds}/variants", "--index_dir", c.results])]


def stage_confound(c):
    return [
        _step("Confound probe (raw master)", [c.py, c.s("metadata_confound_probe.py"),
              "--config", c.cfg, "--metadata", c.master,
              "--out_dir", f"{c.results}/confound_probe_raw/"]),
        _step("Confound probe (variant)", [c.py, c.s("metadata_confound_probe.py"),
              "--config", c.cfg, "--metadata", c.index,
              "--out_dir", f"{c.results}/confound_probe_{c.variant}/"]),
        _step("Confound probe (OpenForensics only)", [c.py, c.s("metadata_confound_probe.py"),
              "--config", c.cfg, "--metadata", c.master, "--source_filter", "openforensics",
              "--out_dir", f"{c.results}/confound_probe_of/"]),
    ]


def stage_detect(c):
    # run_defake_batch.py mirrors DE-FAKE test.py (squash to 224); feed it the aspect variant
    # index via env to geometry-control detection (see PIPELINE.md 3a).
    env = dict(os.environ, WTP_MASTER_CSV=c.index, WTP_PRED_CSV=c.pred)
    return [
        _step("DE-FAKE detection inference", [c.py, c.s("run_defake_batch.py")], env=env),
        _step("Score DE-FAKE detection", [c.py, c.s("score_defake_detection.py"),
              "--predictions", c.pred, "--out_dir", f"{c.results}/defake_detection_{c.variant}/"]),
    ]


def stage_dct(c):
    extract = [c.py, c.s("dct_extract_features.py"), "--index", c.index, "--out", c.dct_feats]
    if c.jpeg_aug == "on":
        extract.append("--jpeg_aug")
    return [
        _step("DCT feature extraction", extract),
        _step("DCT-SVM (random split)", [c.py, c.s("dct_svm.py"), "--features", c.dct_feats,
              "--out_dir", c.dct_svm_out, "--mode", "random"]),
        _step("DCT-SVM (out-of-set)", [c.py, c.s("dct_svm.py"), "--features", c.dct_feats,
              "--out_dir", f"{c.results}/dct_svm_{c.variant}_oos/", "--mode", "out_of_set",
              "--holdout_generators", "FLUX.1-schnell", "StyleGAN3-FFHQ"]),
    ]


def stage_attribution(c):
    return [
        _step("Fine-tune DE-FAKE head", [c.py, c.s("finetune_defake_head.py"), "--config", c.cfg,
              "--index", c.index, "--jpeg_aug", c.jpeg_aug, "--out_dir", c.finetune_out,
              "--features_cache", c.feats, "--captions_csv", c.captions, "--device", c.device]),
        _step("Evaluate attribution", [c.py, c.s("eval_defake_attribution.py"), "--config", c.cfg,
              "--predictions", f"{c.finetune_out}finetune_per_image.csv",
              "--out_dir", c.attr_eval_out, "--pred_col", "pred_generator"]),
        _step("Leave-one-generator-out", [c.py, c.s("leave_one_generator_out.py"), "--config",
              c.cfg, "--index", c.index, "--jpeg_aug", c.jpeg_aug,
              "--out_dir", f"{c.results}/logo_{c.variant}_{c.augtag}/",
              "--features_cache", c.feats, "--captions_csv", c.captions,
              "--targets", "FLUX.1-schnell", "StyleGAN3-FFHQ"]),
    ]


def stage_oos(c):
    return [_step("Out-of-set analysis", [c.py, c.s("out_of_set_analysis.py"), "--config", c.cfg,
            "--out_dir", f"{c.results}/oos_{c.variant}/", "--inputs",
            f"finetuned={c.finetune_out}finetune_per_image.csv",
            f"attr_eval={c.attr_eval_out}attribution_per_image.csv"])]


def stage_ganfp(c):
    return [
        _step("GAN-fp features + MLP (path A)", [c.py, c.s("train_ganfp.py"), "--config", c.cfg,
              "--index", c.index, "--jpeg_aug", c.jpeg_aug,
              "--features_cache", f"{c.results}/ganfp_feats_{c.variant}.npz",
              "--out_dir", f"{c.results}/ganfp_feature_{c.variant}/"]),
        _step("GAN-fp CNN (path B)", [c.py, c.s("train_ganfp_cnn.py"), "--config", c.cfg,
              "--index", c.index, "--jpeg_aug", c.jpeg_aug, "--device", c.device,
              "--out_dir", f"{c.results}/ganfp_cnn_{c.variant}/"]),
        _step("GAN-fp benchmark vs DE-FAKE", [c.py, c.s("benchmark_attribution.py"), "--config",
              c.cfg, "--index", c.index, "--out_dir", f"{c.results}/ganfp_benchmark_{c.variant}/",
              "--defake_csv", f"{c.finetune_out}finetune_per_image.csv"]),
    ]


def stage_robustness(c):
    """Generate perturbations from the CURRENT test split, then score DE-FAKE + DCT-SVM +
    attribution on the SAME perturbed set (apples-to-apples method comparison)."""
    train_idx = f"{c.results}/train_index.csv"
    test_idx = f"{c.results}/test_index.csv"
    rd = c.robust_dir
    steps = [
        _step("Make split", [c.py, c.s("make_split.py"), "--config", c.cfg, "--index", c.index,
              "--train_out", train_idx, "--test_out", test_idx]),
        _step("Generate perturbations", [c.py, c.s("robustness_perturb.py"), "--mode", "generate",
              "--config", c.cfg, "--index", test_idx, "--out_root", f"{c.ds}/robust",
              "--index_dir", f"{rd}/"]),
    ]
    # DE-FAKE clean baseline + per-perturbation inference/scoring.
    clean_pred = f"{c.ds}/robust_clean_pred.csv"
    steps.append(_step("DE-FAKE clean baseline", [c.py, c.s("run_defake_batch.py")],
                 env=dict(os.environ, WTP_MASTER_CSV=test_idx, WTP_PRED_CSV=clean_pred)))
    # DCT clean baseline (predict with the trained SVM on the clean test features).
    dct_clean_feats = f"{rd}/dct_clean.npz"
    steps.append(_step("DCT clean features", [c.py, c.s("dct_extract_features.py"),
                 "--index", test_idx, "--out", dct_clean_feats]
                 + (["--jpeg_aug"] if c.jpeg_aug == "on" else [])))
    steps.append(_step("DCT clean predict", [c.py, c.s("dct_svm.py"), "--mode", "predict",
                 "--model", f"{c.dct_svm_out}dct_svm.joblib", "--features", dct_clean_feats,
                 "--out_dir", f"{rd}/dct_clean/"]))
    # Attribution clean baseline (predict with the fine-tuned head on the clean test set).
    steps.append(_step("Attribution clean predict", [c.py, c.s("predict_defake_head.py"),
                 "--config", c.cfg, "--head", f"{c.finetune_out}defake_head.pt",
                 "--index", test_idx, "--captions_csv", c.captions, "--device", c.device,
                 "--out", f"{rd}/attr_clean.csv"]))
    for name in PERTURBATIONS:
        pidx = f"{rd}/index_{name}.csv"
        # DE-FAKE
        steps.append(_step(f"[{name}] DE-FAKE inference", [c.py, c.s("run_defake_batch.py")],
                     env=dict(os.environ, WTP_MASTER_CSV=pidx,
                              WTP_PRED_CSV=f"{c.ds}/robust_{name}_pred.csv")))
        steps.append(_step(f"[{name}] DE-FAKE score", [c.py, c.s("robustness_perturb.py"),
                     "--mode", "score", "--clean", clean_pred,
                     "--perturbed", f"{c.ds}/robust_{name}_pred.csv",
                     "--out", f"{rd}/{name}_drop.json"]))
        # DCT-SVM
        dfeat = f"{rd}/dct_{name}.npz"
        steps.append(_step(f"[{name}] DCT features", [c.py, c.s("dct_extract_features.py"),
                     "--index", pidx, "--out", dfeat]
                     + (["--jpeg_aug"] if c.jpeg_aug == "on" else [])))
        steps.append(_step(f"[{name}] DCT predict", [c.py, c.s("dct_svm.py"), "--mode", "predict",
                     "--model", f"{c.dct_svm_out}dct_svm.joblib", "--features", dfeat,
                     "--out_dir", f"{rd}/dct_{name}/"]))
        steps.append(_step(f"[{name}] DCT score", [c.py, c.s("robustness_perturb.py"), "--mode",
                     "score", "--clean", f"{rd}/dct_clean/dct_per_image.csv",
                     "--perturbed", f"{rd}/dct_{name}/dct_per_image.csv",
                     "--source_index", pidx,
                     "--pred_col", "pred", "--conf_col", "score",
                     "--out", f"{rd}/dct_{name}_drop.json"]))
        # Attribution (which-generator flips)
        steps.append(_step(f"[{name}] Attribution predict", [c.py, c.s("predict_defake_head.py"),
                     "--config", c.cfg, "--head", f"{c.finetune_out}defake_head.pt",
                     "--index", pidx, "--captions_csv", c.captions, "--device", c.device,
                     "--out", f"{rd}/attr_{name}.csv"]))
        steps.append(_step(f"[{name}] Attribution score", [c.py, c.s("robustness_perturb.py"),
                     "--mode", "score", "--clean", f"{rd}/attr_clean.csv",
                     "--perturbed", f"{rd}/attr_{name}.csv", "--source_index", pidx,
                     "--pred_col", "pred_generator",
                     "--conf_col", "confidence", "--out", f"{rd}/attr_{name}_drop.json"]))
    return steps


def stage_aggregate(c):
    return [_step("Aggregate report", [c.py, c.s("aggregate_results.py"),
            "--results_dir", f"{c.results}/", "--out", f"{c.results}/REPORT_SUMMARY.md"])]


BUILDERS = {
    "index": stage_index, "variants": stage_variants, "confound": stage_confound,
    "detect": stage_detect, "dct": stage_dct, "attribution": stage_attribution,
    "oos": stage_oos, "ganfp": stage_ganfp, "robustness": stage_robustness,
    "aggregate": stage_aggregate,
}


def _check_prereqs(c, stages):
    """Fail fast (before running anything) when a downstream stage needs an artifact that a
    skipped upstream stage would have produced. Prevents the deep FileNotFoundError you hit
    when running ganfp/robustness without first running attribution/dct/detect."""
    missing = []

    def need(stage_here, producer, artifact, why):
        if stage_here in stages and producer not in stages and not os.path.exists(artifact):
            missing.append("  - %s needs '%s' (run the '%s' stage first)" % (why, artifact, producer))

    head = f"{c.finetune_out}defake_head.pt"
    dct_model = f"{c.dct_svm_out}dct_svm.joblib"
    finetune_csv = f"{c.finetune_out}finetune_per_image.csv"

    need("robustness", "attribution", head, "attribution-robustness")
    need("robustness", "dct", dct_model, "DCT-SVM-robustness")
    need("robustness", "detect", c.pred, "captions/detect-robustness")
    need("ganfp", "attribution", finetune_csv, "GAN-fp vs DE-FAKE comparison")

    if missing:
        raise SystemExit(
            "Missing prerequisites for the requested stages:\n" + "\n".join(missing) +
            "\n\nEither add the producer stage(s) to --stages, or run the headline stages first, "
            "e.g.:\n  --stages detect,dct,attribution,oos,aggregate")


def main(args):
    logger = io_utils.setup_logging("run_experiment")
    stages = [s.strip() for s in args.stages.split(",") if s.strip()]
    unknown = [s for s in stages if s not in BUILDERS]
    if unknown:
        raise SystemExit("Unknown stage(s): %s. Valid: %s" % (unknown, ", ".join(ALL_STAGES)))

    c = Ctx(args)
    if not args.dry_run:
        _check_prereqs(c, stages)
    logger.info("Variant=%s jpeg_aug=%s device=%s", c.variant, c.jpeg_aug, c.device)
    logger.info("Interpreter=%s  dataset=%s  results=%s", c.py, c.ds, c.results)
    logger.info("Stages: %s%s", ", ".join(stages), "  [DRY RUN]" if args.dry_run else "")

    steps = []
    for st in stages:
        steps.extend(BUILDERS[st](c))

    if not args.dry_run:
        io_utils.ensure_dir(c.results)

    failures = []
    for i, step in enumerate(steps, 1):
        printable = " ".join(step["cmd"])
        envnote = ""
        if step["env"] is not None:
            extra = {k: step["env"][k] for k in ("WTP_MASTER_CSV", "WTP_PRED_CSV")
                     if k in step["env"]}
            envnote = "  [env: %s]" % extra if extra else ""
        logger.info("[%d/%d] %s%s", i, len(steps), step["desc"], envnote)
        logger.info("      $ %s", printable)
        if args.dry_run:
            continue
        try:
            subprocess.run(step["cmd"], env=step["env"], check=True)
        except subprocess.CalledProcessError as exc:
            logger.error("Step failed (exit %s): %s", exc.returncode, step["desc"])
            if args.keep_going:
                failures.append(step["desc"])
                continue
            raise SystemExit("Aborting at step %d/%d (%s). Use --keep_going to continue."
                             % (i, len(steps), step["desc"]))

    if args.dry_run:
        logger.info("Dry run complete: %d step(s) planned.", len(steps))
    elif failures:
        logger.warning("Finished with %d failed step(s): %s", len(failures), failures)
    else:
        logger.info("All %d step(s) completed. Summary -> %s/REPORT_SUMMARY.md", len(steps),
                    c.results)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Orchestrate the WTP pipeline stages (see PIPELINE.md).")
    p.add_argument("--config", default="configs/config.yaml")
    p.add_argument("--variant", default="aspect", choices=["aspect", "scaled", "cropped"],
                   help="Preprocessing variant index to run on (aspect = confound-controlled).")
    p.add_argument("--jpeg_aug", default="on", choices=["on", "off"],
                   help="JPEG augmentation (on = controlled; off = raw baseline).")
    p.add_argument("--device", default="cuda")
    p.add_argument("--stages", default=",".join(DEFAULT_STAGES),
                   help="Comma list. Valid: " + ", ".join(ALL_STAGES))
    p.add_argument("--results_dir", default="results")
    p.add_argument("--dataset_dir", default=None,
                   help="Dataset root (default: $WTP_ROOT/dataset).")
    p.add_argument("--python", default=None,
                   help="Interpreter for sub-scripts (default: $WTP_PY_DEFAKE or this python).")
    p.add_argument("--captions_csv", default=None,
                   help="Captions CSV (full_path, blip_caption) for 1024-dim DE-FAKE features. "
                        "Default: defake_predictions_all.csv if present, else the same-variant "
                        "detect output (defake_predictions_<variant>.csv).")
    p.add_argument("--dry_run", action="store_true", help="Print the command plan, run nothing.")
    p.add_argument("--keep_going", action="store_true",
                   help="Continue after a failing step instead of aborting.")
    main(p.parse_args())
