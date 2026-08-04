#!/usr/bin/env python3
"""Validate active WTP documentation links and handover hygiene.

This checker intentionally skips historical and archived material. It verifies that active
Markdown links resolve inside the repository, required entry-point documents exist, and active
handover documentation does not contain a concrete SSH login command.

Usage:
    python3 scripts/check_docs.py
"""

import os
import re
import sys
from pathlib import Path
from urllib.parse import unquote


ROOT = Path(__file__).resolve().parents[1]
REQUIRED = [
    "README.md",
    "docs/RUNBOOK.md",
    "docs/METHODOLOGY.md",
    "docs/DATA_PROVENANCE.md",
    "docs/EXPERIMENTS.md",
    "docs/CLI_REFERENCE.md",
    "docs/REFERENCES.md",
    "releases/release.template.yaml",
]
REDIRECTS = []
LINK_RE = re.compile(r"!?\[[^\]]*\]\(([^)]+)\)")
SSH_RE = re.compile(r"\bssh\s+([^\s`]+)@([^\s`]+)", re.IGNORECASE)


def active_markdown_files():
    """Return committed-style active Markdown paths, excluding history and archive trees."""
    files = []
    for path in ROOT.rglob("*.md"):
        rel = path.relative_to(ROOT)
        parts = set(rel.parts)
        if ".git" in parts or "history" in parts or "archive" in parts:
            continue
        if rel.parts and rel.parts[0] in {"results", "logs", "dataset", "models"}:
            continue
        files.append(path)
    return sorted(files)


def parse_link_target(raw):
    """Return the path part of a Markdown target, without an optional quoted title."""
    value = raw.strip()
    if value.startswith("<") and value.endswith(">"):
        value = value[1:-1]
    match = re.match(r"^(\S+)(?:\s+[\"'].*[\"'])?$", value)
    return match.group(1) if match else value


def main():
    errors = []
    warnings = []

    for rel in REQUIRED:
        if not (ROOT / rel).exists():
            errors.append("Missing required handover file: %s" % rel)

    for path in active_markdown_files():
        rel = path.relative_to(ROOT)
        text = path.read_text(encoding="utf-8")

        for match in LINK_RE.finditer(text):
            target = parse_link_target(match.group(1))
            if not target or target.startswith(("http://", "https://", "mailto:", "#")):
                continue
            target = target.split("#", 1)[0].split("?", 1)[0]
            target = unquote(target)
            if not target or "<" in target or ">" in target:
                continue
            resolved = (path.parent / target).resolve()
            try:
                resolved.relative_to(ROOT.resolve())
            except ValueError:
                errors.append("Link escapes repository in %s: %s" % (rel, target))
                continue
            if not resolved.exists():
                errors.append("Broken link in %s: %s" % (rel, target))

        for match in SSH_RE.finditer(text):
            token = match.group(0)
            if "<" not in token and ">" not in token:
                errors.append("Concrete SSH login found in active documentation %s: %s" % (
                    rel, token))

    gitignore = ROOT / ".gitignore"
    if gitignore.exists():
        lines = [line.strip() for line in gitignore.read_text(encoding="utf-8").splitlines()]
        for pattern in ("*.png", "*.jpg", "*.jpeg"):
            if pattern in lines:
                errors.append(
                    "Global image ignore %r prevents documentation/report figures from being tracked" % pattern
                )

    for warning in warnings:
        print("WARNING: %s" % warning, file=sys.stderr)
    for error in errors:
        print("ERROR: %s" % error, file=sys.stderr)

    if errors:
        print("Documentation check failed with %d error(s)." % len(errors), file=sys.stderr)
        return 1

    print("Documentation check passed: %d active Markdown file(s)." % len(active_markdown_files()))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
