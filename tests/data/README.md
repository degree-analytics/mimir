# Test Data Snapshots

This directory stores compressed fixtures consumed by the integration suite. To refresh the
Spacewalker documentation snapshot:

```bash
python scripts/freeze_spacewalker_docs.py
```

The script pulls from `../spacewalker/docs` relative to the repository root, normalises metadata
for repeatable diffs, and rewrites `spacewalker_docs.tar.gz`. Always inspect the reported SHA256
and size in your PR so reviewers can tell when the corpus changes.
