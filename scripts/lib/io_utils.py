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
