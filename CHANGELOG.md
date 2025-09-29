---
purpose: "Track released versions and their key changes"
audience: "Maintainers, release engineers"
owner: "Release Eng"
review: "2025-10-01 (Quarterly)"
status: "Active"
---

# Changelog

## When to Use This

- Summarize changes included in a tagged release
- Confirm whether a fix already shipped

## Prerequisites

- Tagged release notes (generated via automation)

## v1.0.0 - 2025-09-27

- Replaced the experimental CLI with the streamlined command set (`mimir index`,
  `mimir search`, `mimir ask`, `mimir status`, `mimir version`).
- Added pytest coverage for the indexer, search engine, and CLI workflows.
- Disabled vector search by default while keeping optional extras available when
  embeddings are installed.
- Documented installation, configuration, and usage in the refreshed README and
  CLI reference docs.
- Introduced release automation scaffolding (`__version__`, packaging
  metadata) in preparation for GitHub Actions driven tagging.

## References

- `docs/index.md`
- `docs/SECURITY_FIXES.md`
