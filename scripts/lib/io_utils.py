"""
IO and config helpers shared across the pipeline.

- Loads configs/config.yaml and resolves ${VAR} placeholders from configs/paths.env
  (falling back to the real process environment).
- Provides timestamped logging into logs/.
- ASCII-only; Python 3.9 compatible.
"""
import logging
import os
import re
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional

_VAR_PATTERN = re.compile(r"\$\{([A-Za-z_][A-Za-z0-9_]*)\}")


def repo_root() -> Path:
    """Return the repository root (parent of the scripts/ directory)."""
    return Path(__file__).resolve().parents[2]


def load_env(env_file: Optional[str] = None) -> Dict[str, str]:
    """Load KEY=VALUE pairs from configs/paths.env, then overlay os.environ.

    The real environment wins so a script can be pointed at alternate paths without
    editing files.
    """
    env: Dict[str, str] = {}
    if env_file is None:
        candidate = repo_root() / "configs" / "paths.env"
        env_file = str(candidate) if candidate.exists() else None
    if env_file and os.path.exists(env_file):
        with open(env_file, "r", encoding="utf-8") as handle:
            for raw in handle:
                line = raw.strip()
                if not line or line.startswith("#") or "=" not in line:
                    continue
                key, value = line.split("=", 1)
                env[key.strip()] = value.strip()
    # Real environment overrides file values.
    for key, value in os.environ.items():
        env[key] = value
    return env


def resolve_placeholders(value, env: Dict[str, str]):
    """Recursively replace ${VAR} placeholders inside strings/lists/dicts."""
    if isinstance(value, str):
        def _sub(match):
            name = match.group(1)
            if name not in env:
                raise KeyError(
                    "Path variable '%s' is not set. Add it to configs/paths.env "
                    "(see configs/paths.example.env) or export it." % name
                )
            return env[name]
        return _VAR_PATTERN.sub(_sub, value)
    if isinstance(value, list):
        return [resolve_placeholders(item, env) for item in value]
    if isinstance(value, dict):
        return {key: resolve_placeholders(item, env) for key, item in value.items()}
    return value


def load_config(config_path: str, env_file: Optional[str] = None) -> dict:
    """Load the YAML config and resolve all ${VAR} placeholders."""
    try:
        import yaml
    except ImportError as exc:  # pragma: no cover
        raise SystemExit(
            "Missing dependency pyyaml. Run: pip install -r requirements.txt") from exc
    with open(config_path, "r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    env = load_env(env_file)
    return resolve_placeholders(config, env)


def ensure_dir(path: str) -> str:
    """Create a directory (and parents) if needed; return the path."""
    Path(path).mkdir(parents=True, exist_ok=True)
    return path


def load_group_map(paths, logger=None) -> Dict[str, str]:
    """Load one or more `full_path,source_image_id` sidecar CSVs (e.g. the
    `openforensics_groups.csv` written by extract_openforensics.py) into a single
    {full_path: group_id} dict, for group-aware splitting (defake_head.stratified_split's
    `groups=` argument). Rows/files not present are silently skipped (a missing sidecar just
    means that dataset has no known coupling and falls back to singleton groups, i.e. the split
    behaves exactly as before for it). `paths` may be a single path or a list of paths.
    """
    import csv as _csv

    if isinstance(paths, str):
        paths = [paths]
    group_map: Dict[str, str] = {}
    for p in paths or []:
        if not p or not os.path.exists(p):
            continue
        with open(p, "r", newline="", encoding="utf-8") as fh:
            for row in _csv.DictReader(fh):
                full_path = row.get("full_path")
                group_id = row.get("source_image_id")
                if full_path and group_id:
                    group_map[full_path] = group_id
        if logger:
            logger.info("Loaded group map %s (%d cumulative entries)", p, len(group_map))
    return group_map


def default_group_map_paths(config: dict):
    """Conventional sidecar location(s) to auto-load for group-aware splitting when a script's
    --group_map flag is not passed. Currently just OpenForensics's
    <dataset_root>/openforensics/openforensics_groups.csv; extend this list if another dataset
    grows the same same-source-multi-crop coupling risk."""
    root = config.get("dataset_root")
    if not root:
        return []
    return [os.path.join(str(root), "openforensics", "openforensics_groups.csv")]


def apply_group_map(paths, group_map: Dict[str, str], logger=None):
    """Map an array-like of full_path values to group ids via `group_map`, falling back to the
    path itself (a singleton group) for any path not present -- so calling this with an empty
    map is a no-op that reproduces the pre-group-aware split exactly.

    When `logger` is given, also checks whether any path that missed an EXACT match would have
    matched by FILENAME ALONE against some group_map key. This does NOT change the matching
    behavior (no basename-based fallback is applied - that would risk false matches for
    datasets with non-unique basenames); it only surfaces, loudly, the specific failure mode
    that made group-aware splitting a silent no-op in practice: the sidecar and the index being
    built with DIFFERENT absolute-path prefixes for the same physical files (e.g.
    extract_openforensics.py run on the HOST with a host --out_dir, while build_master_index.py
    runs INSIDE THE CONTAINER and records the container-side prefix instead). If this ever logs
    a warning, group-aware splitting is not actually doing anything for the affected paths -
    fix the sidecar's recorded prefix (see extract_openforensics.py --record_prefix) rather than
    silently living with the warning.
    """
    import numpy as np
    paths = [str(p) for p in paths]
    result = [group_map.get(p, p) for p in paths]
    if logger is not None and group_map:
        missed = [p for p, r in zip(paths, result) if r == p and p not in group_map]
        if missed:
            basename_index: Dict[str, str] = {}
            for k in group_map:
                basename_index.setdefault(os.path.basename(k), k)
            near_misses = [(p, basename_index[os.path.basename(p)]) for p in missed
                          if os.path.basename(p) in basename_index]
            if near_misses:
                example_p, example_k = near_misses[0]
                logger.warning(
                    "GROUP MAP PREFIX MISMATCH: %d/%d path(s) did not match the group map by "
                    "full path but WOULD match by filename alone (e.g. index path %r vs sidecar "
                    "key %r - same filename, different prefix). Group-aware splitting is doing "
                    "NOTHING for these %d row(s) even though the sidecar loaded successfully. "
                    "This is almost always a host-vs-container absolute-path mismatch - see "
                    "extract_openforensics.py --record_prefix.",
                    len(near_misses), len(paths), example_p, example_k, len(near_misses))
    return np.array(result, dtype=object)


def setup_logging(name: str, log_dir: Optional[str] = None) -> logging.Logger:
    """Configure a logger that writes to stdout and a timestamped file in logs/."""
    if log_dir is None:
        log_dir = str(repo_root() / "logs")
    ensure_dir(log_dir)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = os.path.join(log_dir, "%s_%s.log" % (name, stamp))

    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    logger.handlers = []  # avoid duplicate handlers on re-import
    fmt = logging.Formatter("%(asctime)s %(levelname)s %(message)s")

    stream = logging.StreamHandler(sys.stdout)
    stream.setFormatter(fmt)
    logger.addHandler(stream)

    file_handler = logging.FileHandler(log_path, encoding="utf-8")
    file_handler.setFormatter(fmt)
    logger.addHandler(file_handler)

    logger.info("Logging to %s", log_path)
    return logger
