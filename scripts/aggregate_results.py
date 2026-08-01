#!/usr/bin/env python3
"""
Aggregate all metric JSONs under results/ into a single markdown summary for the report.

Walks results/ for known metric files (detection_metrics.json, attribution_metrics.json,
metrics.json from dct_svm, finetune_metrics.json, logo_summary.json, out_of_set_summary.json)
and renders compact tables. This turns scattered run outputs into one paste-ready report
appendix.

Usage:
  /usr/bin/python3.9 scripts/aggregate_results.py --results_dir results/ \
      --out results/REPORT_SUMMARY.md
"""
import argparse
import glob
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from lib import io_utils  # noqa: E402


def _load(path):
    try:
        with open(path, "r", encoding="utf-8") as fh:
            return json.load(fh)
    except Exception:  # noqa: BLE001
        return None


def main(args):
    logger = io_utils.setup_logging("aggregate_results")
    lines = ["# Auto-aggregated results summary", ""]

    detection = sorted(glob.glob(os.path.join(args.results_dir, "**", "detection_metrics.json"),
                                 recursive=True))
    if detection:
        lines += ["## Detection (DE-FAKE)", ""]
        for path in detection:
            data = _load(path)
            if not data:
                continue
            tag = os.path.relpath(os.path.dirname(path), args.results_dir)
            ov = data.get("overall", {})
            lines.append("- %s: acc=%.3f balAcc=%.3f F1=%.3f kappa=%.3f%s" % (
                tag, ov.get("accuracy", float("nan")), ov.get("balanced_accuracy", float("nan")),
                ov.get("macro_f1", float("nan")), ov.get("cohen_kappa", float("nan")),
                (" AUROC=%.3f" % ov["auroc"]) if "auroc" in ov else ""))
        lines.append("")

    svm = sorted(glob.glob(os.path.join(args.results_dir, "**", "metrics.json"), recursive=True))
    svm = [p for p in svm if "dct" in p.lower()]
    if svm:
        lines += ["## Detection (DCT linear-SVM)", ""]
        for path in svm:
            data = _load(path) or {}
            tag = os.path.relpath(os.path.dirname(path), args.results_dir)
            t = data.get("test", {})
            lines.append("- %s [%s]: balAcc=%.3f F1=%.3f kappa=%.3f%s" % (
                tag, data.get("mode", "?"), t.get("balanced_accuracy", float("nan")),
                t.get("macro_f1", float("nan")), t.get("cohen_kappa", float("nan")),
                (" AUROC=%.3f" % t["auroc"]) if "auroc" in t else ""))
        lines.append("")

    attr = sorted(glob.glob(os.path.join(args.results_dir, "**", "attribution_metrics.json"),
                            recursive=True))
    attr += sorted(glob.glob(os.path.join(args.results_dir, "**", "finetune_metrics.json"),
                             recursive=True))
    if attr:
        lines += ["## Attribution", ""]
        for path in attr:
            data = _load(path) or {}
            tag = os.path.relpath(os.path.dirname(path), args.results_dir)
            # OOS top-1/kappa are definitional when the true class is absent, and `all_fakes`
            # mixes known and absent-label populations. Keep those in raw JSON only.
            for split_name in ("in_set", "test"):
                res = data.get(split_name)
                if isinstance(res, dict) and "top1_accuracy" in res:
                    lines.append("- %s [%s]: top1=%.3f macroF1=%.3f balAcc=%.3f "
                                 "kappa=%.3f" % (
                        tag, split_name, res["top1_accuracy"], res["macro_f1"],
                        res["balanced_accuracy"], res.get("cohen_kappa", float("nan"))))
        lines.append("")

    logo = sorted(glob.glob(os.path.join(args.results_dir, "**", "logo_summary.json"),
                            recursive=True))
    if logo:
        lines += ["## Leave-one-generator-out (forced labels on unseen generators)", ""]
        for path in logo:
            data = _load(path) or {}
            for gen, info in data.items():
                lines.append("- held-out %s (n=%d): forced=%s meanConf=%.3f meanEnt=%.3f" % (
                    gen, info.get("n_held_out", 0), info.get("forced_label_distribution", {}),
                    info.get("mean_confidence", float("nan")),
                    info.get("mean_entropy", float("nan"))))
        lines.append("")

    oos = sorted(glob.glob(os.path.join(args.results_dir, "**", "out_of_set_summary.json"),
                           recursive=True))
    if oos:
        lines += ["## Out-of-set confidence summary", ""]
        for path in oos:
            data = _load(path) or {}
            for name, res in data.items():
                io = res.get("in_set", {})
                oo = res.get("out_of_set", {})
                lines.append("- %s: in-set conf=%.3f (n=%d) vs out-of-set conf=%.3f (n=%d)" % (
                    name, io.get("mean_confidence", float("nan")), io.get("n", 0),
                    oo.get("mean_confidence", float("nan")), oo.get("n", 0)))
        lines.append("")

    cascade = sorted(glob.glob(os.path.join(args.results_dir, "**", "cascade_metrics.json"),
                               recursive=True))
    if cascade:
        lines += ["## End-to-end cascade (DCT-SVM -> DE-FAKE attribution)", ""]
        for path in cascade:
            data = _load(path) or {}
            tag = os.path.relpath(os.path.dirname(path), args.results_dir)
            known = data.get("known_fake", {})
            conditional = known.get("conditional_attribution") or {}
            end_to_end = known.get("end_to_end_attribution") or {}
            lines.append(
                "- %s: known-fake n=%d detected=%d conditionalTop1=%.3f "
                "conditionalKappa=%.3f endToEndTop1=%.3f endToEndKappa=%.3f" % (
                    tag, known.get("n", 0), known.get("n_detected", 0),
                    conditional.get("top1_accuracy", float("nan")),
                    conditional.get("cohen_kappa", float("nan")),
                    end_to_end.get("top1_accuracy", float("nan")),
                    end_to_end.get("cohen_kappa", float("nan"))))
        lines.append("")

    ffhq = sorted(glob.glob(os.path.join(args.results_dir, "**", "ffhq_ablation.json"),
                            recursive=True))
    if ffhq:
        lines += ["## FFHQ removal diagnostic", ""]
        for path in ffhq:
            data = _load(path) or {}
            delta = data.get("delta_without_minus_with", {})
            lines.append(
                "- without-minus-with: top1=%+.3f balAcc=%+.3f kappa=%+.3f "
                "StyleGAN3Recall=%+.3f StyleGAN3RealRate=%+.3f" % (
                    delta.get("top1_accuracy", float("nan")),
                    delta.get("balanced_accuracy", float("nan")),
                    delta.get("cohen_kappa", float("nan")),
                    delta.get("stylegan3_recall", float("nan")),
                    delta.get("stylegan3_real_prediction_rate", float("nan"))))
        lines.append("")

    ci_files = sorted(glob.glob(os.path.join(args.results_dir, "ci_*.json")))
    if ci_files:
        lines += ["## Bootstrap 95% confidence intervals", ""]
        for path in ci_files:
            data = _load(path) or {}
            overall = data.get("overall", {})
            tag = os.path.splitext(os.path.basename(path))[0]
            parts = []
            for metric in ("top1_accuracy", "balanced_accuracy", "cohen_kappa", "auroc"):
                value = overall.get(metric)
                if isinstance(value, dict) and value.get("point") is not None:
                    parts.append("%s=%.3f [%.3f, %.3f]" % (
                        metric, value["point"], value.get("lo", float("nan")),
                        value.get("hi", float("nan"))))
            if parts:
                lines.append("- %s: %s (n=%s)" % (
                    tag, "; ".join(parts), overall.get("n", "?")))
        lines.append("")

    seed_files = sorted(glob.glob(os.path.join(args.results_dir, "seed_sweep*.json")))
    if seed_files:
        lines += ["## Attribution seed sensitivity", ""]
        for path in seed_files:
            data = _load(path) or {}
            tag = os.path.splitext(os.path.basename(path))[0]
            parts = []
            for metric in ("top1_accuracy", "balanced_accuracy", "cohen_kappa"):
                value = data.get(metric)
                if isinstance(value, dict) and value.get("mean") is not None:
                    parts.append("%s=%.3f+/-%.3f" % (
                        metric, value["mean"], value.get("std", float("nan"))))
            if parts:
                lines.append("- %s: %s (%d seeds)" % (
                    tag, "; ".join(parts), len(data.get("seeds", []))))
        lines.append("")

    significance_path = os.path.join(args.results_dir, "defake_vs_dct_significance.json")
    significance = _load(significance_path)
    if significance:
        lines += ["## Paired DCT vs pretrained DE-FAKE comparison", ""]
        auroc = significance.get("auroc", {}).get("diff_defake_minus_dct", {})
        bal = significance.get("balanced_accuracy", {}).get("diff_defake_minus_dct", {})
        mcnemar = significance.get("mcnemar", {})
        lines.append(
            "- shared n=%d; McNemar p=%.4f; DE-FAKE-minus-DCT AUROC=%+.3f "
            "[%.3f, %.3f], p=%.3f; balancedAcc=%+.3f [%.3f, %.3f], p=%.3f" % (
                significance.get("n_shared", 0), mcnemar.get("p_value", float("nan")),
                auroc.get("point", float("nan")), auroc.get("lo", float("nan")),
                auroc.get("hi", float("nan")), auroc.get("p_bootstrap", float("nan")),
                bal.get("point", float("nan")), bal.get("lo", float("nan")),
                bal.get("hi", float("nan")), bal.get("p_bootstrap", float("nan"))))
        lines.append("")

    # One authoritative audit only. Historical `_full`/`_corrected` diagnostics may coexist in
    # an old run directory and must not republish superseded warnings.
    audit_files = [
        path for path in [
            os.path.join(args.results_dir, "leakage_audit_8way.json"),
            os.path.join(args.results_dir, "leakage_audit_8way_corrected.json"),
        ] if os.path.exists(path)
    ][:1]
    if audit_files:
        lines += ["## Split-integrity audits", ""]
        for path in audit_files:
            data = _load(path) or {}
            group = data.get("group_straddle", {})
            exact = data.get("exact_cross_split_duplicates", {})
            near = data.get("near_cross_split_duplicates", {})
            lines.append("- %s: groups=%d/%d straddling; exact=%d; near-dHash=%d "
                         "(diagnostic only)" % (
                os.path.basename(path), group.get("n_groups_straddling", 0),
                group.get("n_groups_checked", 0), exact.get("count", 0),
                near.get("count", 0)))
        lines.append("")

    io_utils.ensure_dir(os.path.dirname(os.path.abspath(args.out)))
    with open(args.out, "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines))
    logger.info("Wrote %s", args.out)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Aggregate metric JSONs into one summary.")
    parser.add_argument("--results_dir", default="results/")
    parser.add_argument("--out", default="results/REPORT_SUMMARY.md")
    main(parser.parse_args())
