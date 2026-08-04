# Release records

Each supervisor-ready evidence set has one immutable release directory:

```text
releases/<release-id>/release.yaml
```

Create it from `release.template.yaml` after the final server run. The release manifest connects a Git commit to one result directory, its required artifacts, headline spot checks, and mounted-file verification policy.

Do not commit a completed release manifest until:

- the run has finished
- metrics have been read from generated artifacts
- data and checkpoint manifests exist
- environment locks have been captured
- `scripts/verify_handover.py --release ... --verify_mounts` passes

A release manifest is a verification contract, not a manually written results summary.
