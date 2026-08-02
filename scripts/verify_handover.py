#!/usr/bin/env python3
"""
Read-only handover verifier for one immutable run directory.

Confirms that results/<run_id>/ is complete, internally consistent, and matches the known
authoritative headline numbers - WITHOUT re-running anything or touching the dataset. This is
the last check before handing the repository + evidence bundle to a supervisor.

Checks performed:
  1. run_manifest.json exists with a run_id/config_sha256/core_commit, and any post-creation
     analysis_history entries are surfaced (not hidden).
  2. Required metrics/CI/seed/audit/image-only artifacts are present.
  3. The canonical leakage audit shows 0 group-straddles and 0 exact cross-split duplicates.
  4. Known headline numbers match within tolerance (only for run_ids in HEADLINE_SPOT_CHECKS).
  5. No stale flat (non-run-scoped) REPORT_SUMMARY.md sits next to this run directory, and the
     canonical leakage audit file (not just a historical _full/_corrected one) is present.
  6. If data_manifest.csv / checkpoint_manifest.csv (from create_data_manifest.py) are present,
     every non-missing row is re-hashed against the mounted file at its recorded path. This only
     verifies anything meaningful when run FROM the same server/mount the manifest was built
     from; pass --verify_mounts to make an unreachable mount a hard failure instead of a warning
     (use that on the server itself, not when reviewing evidence off-server).

Usage:
  python scripts/verify_handover.py --results_dir results/2026-08-01_eightway_v1/
"""
import argparse
import hashlib
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from lib import io_utils  # noqa: E402

REQUIRED_ARTIFACTS = [
    "run_manifest.json",
    "REPORT_SUMMARY.md",
    "leakage_audit_8way.json",
    "seed_sweep_8way.json",
    "seed_sweep_8way_image_only.json",
    "ci_attr_8way.json",
    "ci_attr_8way_image_only.json",
    "ci_attr_9way.json",
    "ci_dct_detection.json",
    "ci_defake_detection.json",
    "ci_cascade_conditional.json",
    "ci_cascade_end_to_end.json",
    "defake_vs_dct_significance.json",
]

# Known headline numbers for the named authoritative run (docs/EXPERIMENT_CATALOG.md,
# report/AUTHORITATIVE_RESULTS_DRAFT.md). A mismatch beyond --tolerance means the evidence in
# --results_dir is NOT that run and must not be handed over labeled as such.
HEADLINE_SPOT_CHECKS = {
    "2026-08-01_eightway_v1": [
        ("dct_svm_{variant}/metrics.json", ["test", "balanced_accuracy"], 0.608),
        ("attr_eval_8way_{variant}/attribution_metrics.json", ["in_set", "top1_accuracy"], 0.873),
        ("seed_sweep_8way.json", ["top1_accuracy", "mean"], 0.814),
    ],
}


def _load(path):
    with open(path, "r", encoding="utf-8") as fh:
        return json.load(fh)


def _get(data, keys):
    for key in keys:
        if not isinstance(data, dict) or key not in data:
            return None
        data = data[key]
    return data


def _sha256_file(path, chunk_size=1 << 20):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(chunk_size), b""):
            h.update(chunk)
    return h.hexdigest()


def _truthy(value):
    return str(value).strip().lower() in ("true", "1", "yes")


class Verifier:
    """Accumulates errors/warnings; construct once, call .run(), then inspect .errors."""

    def __init__(self, results_dir, run_id=None, variant="aspect", jpeg_aug="on",
                tolerance=0.01, require_data_manifest=False, require_checkpoint_manifest=False,
                verify_mounts=False):
        self.results_dir = results_dir.rstrip("/").rstrip("\\")
        self.run_id = run_id
        self.variant = variant
        self.augtag = "jpegaug" if jpeg_aug == "on" else "raw"
        self.tolerance = tolerance
        self.require_data_manifest = require_data_manifest
        self.require_checkpoint_manifest = require_checkpoint_manifest
        self.verify_mounts = verify_mounts
        self.errors = []
        self.warnings = []

    def _path(self, rel):
        return os.path.join(self.results_dir, rel)

    def check_manifest(self):
        manifest_path = self._path("run_manifest.json")
        if not os.path.exists(manifest_path):
            self.errors.append("Missing run_manifest.json in %s" % self.results_dir)
            return None
        manifest = _load(manifest_path)
        if self.run_id and manifest.get("run_id") != self.run_id:
            self.errors.append("run_manifest.json run_id=%r does not match expected %r" % (
                manifest.get("run_id"), self.run_id))
        if not manifest.get("config_sha256"):
            self.errors.append("run_manifest.json missing config_sha256")
        core_commit = manifest.get("core_commit") or manifest.get("git_commit")
        if not core_commit:
            self.errors.append("run_manifest.json missing core_commit/git_commit")
        history = manifest.get("analysis_history") or []
        if history:
            self.warnings.append(
                "run_manifest.json records %d post-creation analysis_history entr%s (code "
                "drifted from the core commit at least once after creation): reasons=%s" % (
                    len(history), "y" if len(history) == 1 else "ies",
                    [h.get("reason") for h in history]))
        return manifest

    def check_required_artifacts(self):
        for rel in REQUIRED_ARTIFACTS:
            if not os.path.exists(self._path(rel)):
                self.errors.append("Missing required artifact: %s" % rel)

    def check_audit_gates(self):
        path = self._path("leakage_audit_8way.json")
        if not os.path.exists(path):
            return  # already reported by check_required_artifacts
        data = _load(path)
        n_straddling = _get(data, ["group_straddle", "n_groups_straddling"])
        n_exact = _get(data, ["exact_cross_split_duplicates", "count"])
        if n_straddling != 0:
            self.errors.append(
                "leakage_audit_8way.json group_straddle.n_groups_straddling=%s (must be 0)"
                % n_straddling)
        if n_exact != 0:
            self.errors.append(
                "leakage_audit_8way.json exact_cross_split_duplicates.count=%s (must be 0)"
                % n_exact)

    def check_headline_spot_checks(self):
        for rel_tpl, keys, expected in HEADLINE_SPOT_CHECKS.get(self.run_id, []):
            rel = rel_tpl.format(variant=self.variant, augtag=self.augtag)
            path = self._path(rel)
            if not os.path.exists(path):
                self.errors.append("Cannot spot-check %s: missing %s" % (".".join(keys), rel))
                continue
            actual = _get(_load(path), keys)
            if actual is None:
                self.errors.append(
                    "Cannot spot-check %s in %s: key path not found" % (".".join(keys), rel))
                continue
            if abs(actual - expected) > self.tolerance:
                self.errors.append(
                    "Headline spot check failed: %s %s = %.4f, expected ~%.3f (+/-%.3f). This "
                    "evidence set does not match the known authoritative %s numbers - do not "
                    "hand it over labeled as that run." % (
                        rel, ".".join(keys), actual, expected, self.tolerance, self.run_id))

    def check_no_stale_flat_summaries(self):
        # A flat, non-run-scoped REPORT_SUMMARY.md sitting next to run directories is exactly
        # the cross-run-mixing failure mode aggregate_results.py now refuses to produce; its
        # presence means someone ran an old/unscoped aggregator against the wrong directory.
        parent = os.path.dirname(self.results_dir)
        flat_summary = os.path.join(parent, "REPORT_SUMMARY.md")
        if os.path.exists(flat_summary):
            self.errors.append(
                "Found a flat, non-run-scoped %s next to %s. This can silently mix runs and "
                "must not ship in the handover evidence archive." % (
                    flat_summary, self.results_dir))
        legacy_only = (
            not os.path.exists(self._path("leakage_audit_8way.json"))
            and (os.path.exists(self._path("leakage_audit_8way_full.json"))
                 or os.path.exists(self._path("leakage_audit_8way_corrected.json"))))
        if legacy_only:
            self.errors.append(
                "Only a historical leakage_audit_8way_full.json/_corrected.json is present; "
                "the canonical leakage_audit_8way.json is missing.")

    def _verify_manifest_file(self, manifest_path, kind, require_present):
        """Re-hash every non-missing row's `path` and compare to its recorded sha256. A file
        this machine cannot reach (no mount) is reported as a warning, unless --verify_mounts
        was requested (the server-side, sign-off mode), in which case it is a hard failure."""
        if not os.path.exists(manifest_path):
            note = ("Missing %s; run scripts/create_data_manifest.py to generate it." %
                    os.path.basename(manifest_path))
            (self.errors if require_present else self.warnings).append(note)
            return

        import csv as _csv
        mismatches = []
        unreachable = 0
        checked = 0
        with open(manifest_path, newline="", encoding="utf-8") as fh:
            for row in _csv.DictReader(fh):
                if _truthy(row.get("missing")):
                    continue  # already known-missing when the manifest was generated
                path = row.get("path")
                if not path or not os.path.isfile(path):
                    unreachable += 1
                    continue
                checked += 1
                if _sha256_file(path) != row.get("sha256"):
                    mismatches.append(path)

        name = os.path.basename(manifest_path)
        if mismatches:
            self.errors.append(
                "%s: %d file(s) do not match their recorded sha256 - the mounted %s has "
                "drifted from what this run's evidence was built from: %s%s" % (
                    name, len(mismatches), kind, mismatches[:3],
                    " ..." if len(mismatches) > 3 else ""))
        if unreachable:
            note = ("%s: %d file(s) could not be reached from this machine (expected when "
                    "reviewing evidence off the server, without the dataset/model mount)." % (
                        name, unreachable))
            if self.verify_mounts:
                self.errors.append(note + " --verify_mounts was set, so this is a failure.")
            else:
                self.warnings.append(note)
        elif checked:
            self.warnings.append(
                "%s: re-hashed %d file(s) against the mount, 0 mismatches." % (name, checked))

    def check_data_manifest(self):
        self._verify_manifest_file(self._path("data_manifest.csv"), "dataset",
                                   self.require_data_manifest)

    def check_checkpoint_manifest(self):
        self._verify_manifest_file(self._path("checkpoint_manifest.csv"), "checkpoints",
                                   self.require_checkpoint_manifest)

    def run(self):
        manifest = self.check_manifest()
        self.check_required_artifacts()
        self.check_audit_gates()
        self.check_headline_spot_checks()
        self.check_no_stale_flat_summaries()
        self.check_data_manifest()
        self.check_checkpoint_manifest()
        return manifest


def main(args):
    logger = io_utils.setup_logging("verify_handover")
    verifier = Verifier(args.results_dir, run_id=args.run_id, variant=args.variant,
                        jpeg_aug=args.jpeg_aug, tolerance=args.tolerance,
                        require_data_manifest=args.require_data_manifest,
                        require_checkpoint_manifest=args.require_checkpoint_manifest,
                        verify_mounts=args.verify_mounts)
    manifest = verifier.run()

    for warning in verifier.warnings:
        logger.warning(warning)
    for error in verifier.errors:
        logger.error(error)

    if verifier.errors:
        logger.error("HANDOVER VERIFICATION FAILED: %d error(s).", len(verifier.errors))
        raise SystemExit(1)

    logger.info(
        "HANDOVER VERIFICATION PASSED for %s (run_id=%s, core_commit=%s).",
        args.results_dir, manifest.get("run_id") if manifest else "?",
        ((manifest.get("core_commit") or manifest.get("git_commit")) if manifest else "?"))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Read-only verification that a run directory is complete, internally "
                    "consistent, and matches known authoritative headline numbers.")
    parser.add_argument("--results_dir", required=True,
                        help="Run-scoped evidence directory, e.g. results/2026-08-01_eightway_v1/")
    parser.add_argument("--run_id", default="2026-08-01_eightway_v1",
                        help="Expected run_id; also selects which headline spot checks to run.")
    parser.add_argument("--variant", default="aspect")
    parser.add_argument("--jpeg_aug", default="on", choices=["on", "off"])
    parser.add_argument("--tolerance", type=float, default=0.01)
    parser.add_argument("--require_data_manifest", action="store_true",
                        help="Fail if data_manifest.csv is absent (default: warn only).")
    parser.add_argument("--require_checkpoint_manifest", action="store_true",
                        help="Fail if checkpoint_manifest.csv is absent (default: warn only).")
    parser.add_argument("--verify_mounts", action="store_true",
                        help="Treat an unreachable dataset/checkpoint mount as a failure "
                             "instead of a warning. Use on the server itself, where the mount "
                             "is expected to be present.")
    main(parser.parse_args())
