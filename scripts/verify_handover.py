#!/usr/bin/env python3
"""Read-only verifier for one immutable WTP release.

The recommended interface is a completed release manifest:

    python scripts/verify_handover.py \
        --release releases/<release-id>/release.yaml \
        --verify_mounts

For compatibility, a run directory can still be checked directly with --results_dir and
--run_id. Direct mode verifies structure and manifests but has no embedded run-specific metric
values. Headline checks belong in the release manifest, not in reusable Python code.
"""

import argparse
import csv
import hashlib
import json
import logging
import os
import sys
from pathlib import Path

import yaml


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_REQUIRED_ARTIFACTS = [
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

# Backward-compatible import names. Run-specific values are intentionally absent.
REQUIRED_ARTIFACTS = DEFAULT_REQUIRED_ARTIFACTS
HEADLINE_SPOT_CHECKS = {}
PLACEHOLDER_MARKERS = ("REPLACE_WITH", "<", ">")


def _load_json(path):
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def _load_yaml(path):
    with open(path, "r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle)
    if not isinstance(data, dict):
        raise ValueError("Release manifest must contain a YAML mapping at the top level")
    return data


def _get(data, keys):
    value = data
    for key in keys:
        if not isinstance(value, dict) or key not in value:
            return None
        value = value[key]
    return value


def _sha256_file(path, chunk_size=1 << 20):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _truthy(value):
    return str(value).strip().lower() in ("true", "1", "yes")


def _is_placeholder(value):
    if not isinstance(value, str):
        return False
    return any(marker in value for marker in PLACEHOLDER_MARKERS)


def _hash_matches(actual, expected):
    """Allow full hashes or unambiguous prefixes while rejecting empty values."""
    if not actual or not expected:
        return False
    actual = str(actual).strip().lower()
    expected = str(expected).strip().lower()
    return actual.startswith(expected) or expected.startswith(actual)


def _as_bool(value, default=False):
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    return _truthy(value)


def validate_release_config(release):
    """Raise ValueError when a release file is incomplete or malformed."""
    required = ("release_id", "run_id", "results_dir", "git_commit")
    for key in required:
        value = release.get(key)
        if value in (None, "") or _is_placeholder(value):
            raise ValueError("Release manifest field %r is missing or still a placeholder" % key)

    if release.get("schema_version") not in (1, "1"):
        raise ValueError("Unsupported or missing release schema_version; expected 1")

    commit = str(release["git_commit"]).strip()
    if len(commit) < 7:
        raise ValueError("git_commit must contain at least a seven-character Git hash")

    config_hash = release.get("config_sha256")
    if config_hash is not None and (_is_placeholder(config_hash) or len(str(config_hash)) < 7):
        raise ValueError("config_sha256 is incomplete or still a placeholder")

    artifacts = release.get("required_artifacts", DEFAULT_REQUIRED_ARTIFACTS)
    if not isinstance(artifacts, list) or not all(isinstance(item, str) and item for item in artifacts):
        raise ValueError("required_artifacts must be a list of non-empty relative paths")

    checks = release.get("headline_checks", [])
    if not isinstance(checks, list):
        raise ValueError("headline_checks must be a list")
    for index, check in enumerate(checks):
        if not isinstance(check, dict):
            raise ValueError("headline_checks[%d] must be a mapping" % index)
        if not isinstance(check.get("file"), str) or not check["file"]:
            raise ValueError("headline_checks[%d].file is required" % index)
        keys = check.get("keys")
        if not isinstance(keys, list) or not keys or not all(isinstance(key, str) for key in keys):
            raise ValueError("headline_checks[%d].keys must be a non-empty string list" % index)
        expected = check.get("expected")
        if isinstance(expected, bool) or not isinstance(expected, (int, float)):
            raise ValueError(
                "headline_checks[%d].expected must be a numeric value copied from the run artifact"
                % index
            )
        tolerance = check.get("tolerance", release.get("tolerance", 1e-6))
        if isinstance(tolerance, bool) or not isinstance(tolerance, (int, float)) or tolerance < 0:
            raise ValueError("headline_checks[%d] has an invalid tolerance" % index)


def resolve_results_dir(value):
    path = Path(value)
    if not path.is_absolute():
        path = REPO_ROOT / path
    return str(path.resolve())


class Verifier:
    """Accumulate release verification errors and warnings without modifying evidence."""

    def __init__(
        self,
        results_dir,
        run_id=None,
        variant="aspect",
        jpeg_aug="on",
        tolerance=0.01,
        require_data_manifest=False,
        require_checkpoint_manifest=False,
        verify_mounts=False,
        required_artifacts=None,
        headline_checks=None,
        expected_commit=None,
        expected_config_sha256=None,
        release_id=None,
    ):
        self.results_dir = os.path.abspath(str(results_dir).rstrip("/").rstrip("\\"))
        self.run_id = run_id
        self.variant = variant
        self.augtag = "jpegaug" if jpeg_aug == "on" else "raw"
        self.tolerance = float(tolerance)
        self.require_data_manifest = bool(require_data_manifest)
        self.require_checkpoint_manifest = bool(require_checkpoint_manifest)
        self.verify_mounts = bool(verify_mounts)
        self.required_artifacts = list(required_artifacts or DEFAULT_REQUIRED_ARTIFACTS)
        self.headline_checks = list(headline_checks or [])
        self.expected_commit = expected_commit
        self.expected_config_sha256 = expected_config_sha256
        self.release_id = release_id
        self.errors = []
        self.warnings = []

    def _path(self, relative):
        return os.path.join(self.results_dir, relative)

    def check_results_directory(self):
        if not os.path.isdir(self.results_dir):
            self.errors.append("Results directory does not exist: %s" % self.results_dir)

    def check_manifest(self):
        manifest_path = self._path("run_manifest.json")
        if not os.path.exists(manifest_path):
            self.errors.append("Missing run_manifest.json in %s" % self.results_dir)
            return None

        try:
            manifest = _load_json(manifest_path)
        except (OSError, ValueError, json.JSONDecodeError) as error:
            self.errors.append("Cannot read run_manifest.json: %s" % error)
            return None

        if self.run_id and manifest.get("run_id") != self.run_id:
            self.errors.append(
                "run_manifest.json run_id=%r does not match expected %r"
                % (manifest.get("run_id"), self.run_id)
            )

        actual_config = manifest.get("config_sha256")
        if not actual_config:
            self.errors.append("run_manifest.json missing config_sha256")
        elif self.expected_config_sha256 and not _hash_matches(
            actual_config, self.expected_config_sha256
        ):
            self.errors.append(
                "run_manifest.json config_sha256=%s does not match release value %s"
                % (actual_config, self.expected_config_sha256)
            )

        actual_commit = manifest.get("core_commit") or manifest.get("git_commit")
        if not actual_commit:
            self.errors.append("run_manifest.json missing core_commit/git_commit")
        elif self.expected_commit and not _hash_matches(actual_commit, self.expected_commit):
            self.errors.append(
                "run_manifest.json commit=%s does not match release git_commit=%s"
                % (actual_commit, self.expected_commit)
            )

        history = manifest.get("analysis_history") or []
        if history:
            self.warnings.append(
                "run_manifest.json contains %d post-creation analysis_history entry or entries; "
                "review them before sign-off" % len(history)
            )
        return manifest

    def check_required_artifacts(self):
        for relative in self.required_artifacts:
            if os.path.isabs(relative) or ".." in Path(relative).parts:
                self.errors.append("Unsafe required artifact path in release: %s" % relative)
                continue
            if not os.path.exists(self._path(relative)):
                self.errors.append("Missing required artifact: %s" % relative)

    def check_audit_gates(self):
        path = self._path("leakage_audit_8way.json")
        if not os.path.exists(path):
            return
        try:
            data = _load_json(path)
        except (OSError, ValueError, json.JSONDecodeError) as error:
            self.errors.append("Cannot read leakage_audit_8way.json: %s" % error)
            return

        n_straddling = _get(data, ["group_straddle", "n_groups_straddling"])
        n_exact = _get(data, ["exact_cross_split_duplicates", "count"])
        if n_straddling != 0:
            self.errors.append(
                "leakage_audit_8way.json group_straddle.n_groups_straddling=%s, must be 0"
                % n_straddling
            )
        if n_exact != 0:
            self.errors.append(
                "leakage_audit_8way.json exact_cross_split_duplicates.count=%s, must be 0"
                % n_exact
            )

    def check_headline_spot_checks(self):
        for index, check in enumerate(self.headline_checks):
            relative = check["file"].format(variant=self.variant, augtag=self.augtag)
            keys = check["keys"]
            expected = float(check["expected"])
            tolerance = float(check.get("tolerance", self.tolerance))
            path = self._path(relative)

            if not os.path.exists(path):
                self.errors.append(
                    "Cannot spot-check %s: missing %s" % (".".join(keys), relative)
                )
                continue
            try:
                data = _load_json(path)
            except (OSError, ValueError, json.JSONDecodeError) as error:
                self.errors.append("Cannot read headline artifact %s: %s" % (relative, error))
                continue

            actual = _get(data, keys)
            if actual is None:
                self.errors.append(
                    "Cannot spot-check %s in %s: key path not found"
                    % (".".join(keys), relative)
                )
                continue
            if isinstance(actual, bool) or not isinstance(actual, (int, float)):
                self.errors.append(
                    "Cannot spot-check %s in %s: value is not numeric"
                    % (".".join(keys), relative)
                )
                continue
            if abs(float(actual) - expected) > tolerance:
                self.errors.append(
                    "Headline check %d failed: %s %s=%.10g, expected %.10g (+/- %.10g)"
                    % (index + 1, relative, ".".join(keys), actual, expected, tolerance)
                )

    def check_no_stale_flat_summaries(self):
        parent = os.path.dirname(self.results_dir)
        flat_summary = os.path.join(parent, "REPORT_SUMMARY.md")
        if os.path.exists(flat_summary):
            self.errors.append(
                "Found a flat, non-run-scoped REPORT_SUMMARY.md next to run directories: %s"
                % flat_summary
            )

        canonical = self._path("leakage_audit_8way.json")
        legacy_only = not os.path.exists(canonical) and (
            os.path.exists(self._path("leakage_audit_8way_full.json"))
            or os.path.exists(self._path("leakage_audit_8way_corrected.json"))
        )
        if legacy_only:
            self.errors.append(
                "Only a historical leakage audit is present; leakage_audit_8way.json is required"
            )

    def _verify_manifest_file(self, manifest_path, kind, require_present):
        if not os.path.exists(manifest_path):
            if require_present:
                self.errors.append("Missing required %s manifest: %s" % (kind, manifest_path))
            return

        mismatches = []
        unreachable = []
        declared_missing = []
        malformed = []
        checked = 0

        try:
            with open(manifest_path, newline="", encoding="utf-8") as handle:
                reader = csv.DictReader(handle)
                if not reader.fieldnames or "path" not in reader.fieldnames or "sha256" not in reader.fieldnames:
                    self.errors.append(
                        "%s manifest must contain path and sha256 columns" % kind.capitalize()
                    )
                    return
                for line_number, row in enumerate(reader, start=2):
                    path_value = (row.get("path") or "").strip()
                    expected_hash = (row.get("sha256") or "").strip()
                    if _truthy(row.get("missing")):
                        declared_missing.append(path_value or "line %d" % line_number)
                        continue
                    if not path_value or not expected_hash:
                        malformed.append("line %d" % line_number)
                        continue
                    path = Path(path_value)
                    if not path.is_absolute():
                        path = REPO_ROOT / path
                    if not path.is_file():
                        unreachable.append(str(path))
                        continue
                    checked += 1
                    if _sha256_file(path) != expected_hash:
                        mismatches.append(str(path))
        except (OSError, csv.Error) as error:
            self.errors.append("Cannot read %s manifest %s: %s" % (kind, manifest_path, error))
            return

        name = os.path.basename(manifest_path)
        if malformed:
            self.errors.append(
                "%s contains %d malformed row(s): %s"
                % (name, len(malformed), malformed[:3])
            )
        if declared_missing:
            message = "%s records %d input(s) as missing: %s" % (
                name,
                len(declared_missing),
                declared_missing[:3],
            )
            if require_present:
                self.errors.append(message)
            else:
                self.warnings.append(message)
        if mismatches:
            self.errors.append(
                "%s: %d file(s) do not match their recorded sha256: %s%s"
                % (
                    name,
                    len(mismatches),
                    mismatches[:3],
                    " ..." if len(mismatches) > 3 else "",
                )
            )
        if unreachable:
            message = (
                "%s: %d file(s) cannot be reached from this machine: %s%s"
                % (
                    name,
                    len(unreachable),
                    unreachable[:3],
                    " ..." if len(unreachable) > 3 else "",
                )
            )
            if self.verify_mounts:
                self.errors.append(message + "; --verify_mounts makes this a failure")
            else:
                self.warnings.append(message)
        elif checked:
            self.warnings.append(
                "%s: re-hashed %d mounted file(s), 0 mismatches" % (name, checked)
            )

    def check_data_manifest(self):
        self._verify_manifest_file(
            self._path("data_manifest.csv"), "data", self.require_data_manifest
        )

    def check_checkpoint_manifest(self):
        self._verify_manifest_file(
            self._path("checkpoint_manifest.csv"),
            "checkpoint",
            self.require_checkpoint_manifest,
        )

    def run(self):
        self.check_results_directory()
        manifest = self.check_manifest()
        self.check_required_artifacts()
        self.check_audit_gates()
        self.check_headline_spot_checks()
        self.check_no_stale_flat_summaries()
        self.check_data_manifest()
        self.check_checkpoint_manifest()
        return manifest


def build_settings(args):
    release = None
    release_path = getattr(args, "release", None)
    if release_path:
        release_path = Path(release_path)
        if not release_path.is_absolute():
            release_path = REPO_ROOT / release_path
        release = _load_yaml(release_path)
        validate_release_config(release)

    results_value = getattr(args, "results_dir", None) or (
        release.get("results_dir") if release else None
    )
    if not results_value:
        raise ValueError("Provide --release or --results_dir")

    def choose(name, default=None):
        cli_value = getattr(args, name, None)
        if cli_value is not None:
            return cli_value
        if release and name in release:
            return release[name]
        return default

    return {
        "release": release,
        "results_dir": resolve_results_dir(results_value),
        "run_id": choose("run_id"),
        "variant": choose("variant", "aspect"),
        "jpeg_aug": choose("jpeg_aug", "on"),
        "tolerance": float(choose("tolerance", 0.01)),
        "require_data_manifest": _as_bool(choose("require_data_manifest", False)),
        "require_checkpoint_manifest": _as_bool(
            choose("require_checkpoint_manifest", False)
        ),
        "verify_mounts": bool(getattr(args, "verify_mounts", False)),
        "required_artifacts": (
            release.get("required_artifacts", DEFAULT_REQUIRED_ARTIFACTS)
            if release
            else DEFAULT_REQUIRED_ARTIFACTS
        ),
        "headline_checks": release.get("headline_checks", []) if release else [],
        "expected_commit": release.get("git_commit") if release else None,
        "expected_config_sha256": release.get("config_sha256") if release else None,
        "release_id": release.get("release_id") if release else None,
    }


def main(args):
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    logger = logging.getLogger("verify_handover")

    try:
        settings = build_settings(args)
    except (OSError, ValueError, yaml.YAMLError) as error:
        logger.error("Invalid handover verification input: %s", error)
        raise SystemExit(2)

    verifier = Verifier(
        results_dir=settings["results_dir"],
        run_id=settings["run_id"],
        variant=settings["variant"],
        jpeg_aug=settings["jpeg_aug"],
        tolerance=settings["tolerance"],
        require_data_manifest=settings["require_data_manifest"],
        require_checkpoint_manifest=settings["require_checkpoint_manifest"],
        verify_mounts=settings["verify_mounts"],
        required_artifacts=settings["required_artifacts"],
        headline_checks=settings["headline_checks"],
        expected_commit=settings["expected_commit"],
        expected_config_sha256=settings["expected_config_sha256"],
        release_id=settings["release_id"],
    )
    manifest = verifier.run()

    for warning in verifier.warnings:
        logger.warning(warning)
    for error in verifier.errors:
        logger.error(error)

    if verifier.errors:
        logger.error("HANDOVER VERIFICATION FAILED: %d error(s)", len(verifier.errors))
        raise SystemExit(1)

    if settings["release"] is None:
        logger.warning(
            "Direct --results_dir mode has no run-specific headline checks. "
            "Use a completed --release manifest for supervisor sign-off."
        )

    logger.info(
        "HANDOVER VERIFICATION PASSED for %s (release_id=%s, run_id=%s, commit=%s)",
        settings["results_dir"],
        settings["release_id"] or "direct-mode",
        manifest.get("run_id") if manifest else "?",
        (
            manifest.get("core_commit") or manifest.get("git_commit")
            if manifest
            else "?"
        ),
    )
    return manifest


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--release",
        help="Completed release YAML. Recommended for final supervisor sign-off.",
    )
    parser.add_argument(
        "--results_dir",
        help="Run directory. Overrides results_dir from --release when supplied.",
    )
    parser.add_argument("--run_id", default=None, help="Expected run ID override")
    parser.add_argument("--variant", default=None, help="Artifact path variant override")
    parser.add_argument(
        "--jpeg_aug",
        choices=("on", "off"),
        default=None,
        help="Artifact path augmentation tag override",
    )
    parser.add_argument(
        "--tolerance",
        type=float,
        default=None,
        help="Default numeric tolerance override for headline checks",
    )
    parser.add_argument(
        "--require_data_manifest",
        action="store_true",
        default=None,
        help="Require data_manifest.csv even when no release value is supplied",
    )
    parser.add_argument(
        "--require_checkpoint_manifest",
        action="store_true",
        default=None,
        help="Require checkpoint_manifest.csv even when no release value is supplied",
    )
    parser.add_argument(
        "--verify_mounts",
        action="store_true",
        help="Treat unreachable data/checkpoint manifest paths as failures",
    )
    return parser.parse_args(argv)


if __name__ == "__main__":
    main(parse_args())
