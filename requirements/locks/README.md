# Server environment locks

These files are generated on the server after the final successful run:

```text
sd15-defake.txt
flux1.txt
stylegan3.txt
```

Generate them with:

```bash
bash scripts/capture_env_locks.sh
```

The files are exact `pip freeze` snapshots of the environments that produced the verified evidence. Review local paths, editable installs, and unpinned VCS entries before committing.

Do not create placeholder lock files. Their absence means the final server environment has not yet been captured for this branch.
