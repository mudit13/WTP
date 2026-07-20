"""
Shared class-space rules for DE-FAKE generator attribution.

The professor-facing primary task is fake-only attribution over eight named generators.
An optional joint task adds one merged Real class. OpenForensics-fake is never trainable:
it remains an out-of-set challenge set in both modes.

Pure numpy helpers live here so fine-tuning, evaluation, LOGO, tests, and orchestration use
one taxonomy implementation instead of re-encoding subtly different class lists.
"""
import hashlib
from typing import Dict, Iterable, List, Optional

import numpy as np


FAKE_ONLY = "fake_only"
JOINT = "joint"
VALID_MODES = (FAKE_ONLY, JOINT)


def attribution_config(config: dict) -> dict:
    return config.get("attribution", {}) or {}


def fake_generators(config: dict) -> List[str]:
    """Configured fake-generator training space, preserving declared order."""
    attr = attribution_config(config)
    declared = attr.get("fake_generators")
    if declared is None:
        declared = (list(attr.get("in_set_generators", []))
                    + list(attr.get("finetune_new_classes", [])))
    return list(dict.fromkeys(str(x) for x in declared))


def real_generators(config: dict) -> List[str]:
    return list(dict.fromkeys(
        str(x) for x in attribution_config(config).get("real_generators", [])))


def out_of_set_generators(config: dict) -> List[str]:
    return list(dict.fromkeys(
        str(x) for x in attribution_config(config).get("out_of_set_generators", [])))


def class_mode(config: dict, requested: Optional[str] = None) -> str:
    mode = requested or attribution_config(config).get("primary_mode", FAKE_ONLY)
    if mode not in VALID_MODES:
        raise ValueError("Unknown attribution class mode %r; expected one of %s"
                         % (mode, ", ".join(VALID_MODES)))
    return mode


def real_class_name(config: dict) -> str:
    return str(attribution_config(config).get("real_class_name", "real"))


def display_name(config: dict, canonical_name: str) -> str:
    names = attribution_config(config).get("display_names", {}) or {}
    return str(names.get(str(canonical_name), canonical_name))


def display_names(config: dict, canonical_names: Iterable[str]) -> List[str]:
    return [display_name(config, name) for name in canonical_names]


def classes_for_mode(config: dict, mode: Optional[str] = None) -> List[str]:
    mode = class_mode(config, mode)
    classes = fake_generators(config)
    if mode == JOINT:
        classes = classes + [real_class_name(config)]
    return classes


def remap_reals(generators: Iterable[str], config: dict,
                mode: Optional[str] = None) -> np.ndarray:
    """Map all configured real-source names to one Real label in joint mode."""
    mode = class_mode(config, mode)
    values = np.asarray([str(x) for x in generators], dtype=object)
    if mode == JOINT:
        reals = set(real_generators(config))
        merged = real_class_name(config)
        values = np.asarray([merged if x in reals else x for x in values], dtype=object)
    return values


def _stable_score(path: str, seed: int) -> str:
    return hashlib.sha256(("%d:%s" % (seed, path)).encode("utf-8")).hexdigest()


def balanced_real_mask(generators: Iterable[str], paths: Iterable[str], config: dict,
                       cap: Optional[int] = None, seed: int = 42) -> np.ndarray:
    """Select at most `cap` real rows, balanced across real source datasets.

    Rows within each source are ranked by a stable SHA-256 key. Sources are then sampled in
    round-robin order, so a cap of 300 over four sufficiently large sources yields 75 per source.
    This is deterministic, independent of input row order, and never samples fake rows.
    """
    generators = np.asarray([str(x) for x in generators], dtype=object)
    paths = np.asarray([str(x) for x in paths], dtype=object)
    configured = [g for g in real_generators(config) if np.any(generators == g)]
    all_real = np.isin(generators, configured)
    if cap is None:
        raw_cap = attribution_config(config).get("real_sample_cap")
        cap = int(raw_cap) if raw_cap is not None else int(all_real.sum())
    cap = max(0, int(cap))
    if int(all_real.sum()) <= cap:
        return all_real

    ranked: Dict[str, List[int]] = {}
    for source in configured:
        idx = np.where(generators == source)[0].tolist()
        ranked[source] = sorted(idx, key=lambda i: _stable_score(paths[i], seed))

    selected = []
    offset = 0
    while len(selected) < cap:
        added = False
        for source in configured:
            rows = ranked[source]
            if offset < len(rows) and len(selected) < cap:
                selected.append(rows[offset])
                added = True
        if not added:
            break
        offset += 1

    mask = np.zeros(len(generators), dtype=bool)
    mask[selected] = True
    return mask


def prepare_population(generators: Iterable[str], paths: Iterable[str], config: dict,
                       mode: Optional[str] = None, seed: int = 42,
                       require_all_fakes: bool = True) -> dict:
    """Return consistent train/OOS masks, remapped labels, and class metadata."""
    mode = class_mode(config, mode)
    raw = np.asarray([str(x) for x in generators], dtype=object)
    paths = np.asarray([str(x) for x in paths], dtype=object)
    fakes = fake_generators(config)
    overlap = set(fakes) & set(out_of_set_generators(config))
    if overlap:
        raise ValueError("Generator(s) cannot be both trained and out-of-set: %s"
                         % ", ".join(sorted(overlap)))
    present = set(raw.tolist())
    missing = [g for g in fakes if g not in present]
    if require_all_fakes and missing:
        raise ValueError("Missing configured fake attribution class(es): %s"
                         % ", ".join(missing))

    fake_mask = np.isin(raw, fakes)
    real_mask = np.zeros(len(raw), dtype=bool)
    if mode == JOINT:
        real_mask = balanced_real_mask(raw, paths, config, seed=seed)
    train_mask = fake_mask | real_mask
    mapped = remap_reals(raw, config, mode)
    classes = [g for g in fakes if g in present]
    if mode == JOINT and real_mask.any():
        classes.append(real_class_name(config))

    out_set = set(out_of_set_generators(config))
    oos_mask = np.asarray([g in out_set for g in raw], dtype=bool)
    return {
        "mode": mode,
        "raw_generators": raw,
        "mapped_generators": mapped,
        "train_mask": train_mask,
        "oos_mask": oos_mask,
        "classes": classes,
        "fake_classes": [g for g in fakes if g in present],
        "missing_fake_classes": missing,
        "real_mask": real_mask,
    }
