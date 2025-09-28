---
purpose: "Overview and quick start for the Mímir documentation search CLI"
audience: "Developers integrating docs search; repository maintainers"
owner: "Docs Guild"
review: "2025-10-01 (Quarterly)"
status: "Active"
---

# Mímir Docsearch

## When to Use This

- Setting up the CLI locally to index and search a documentation corpus
- Pointing teammates to the canonical install and quick start flow

## Prerequisites

- Python 3.11+
- `just` command runner (`brew install just` on macOS)
- Local clone of the repository and access to `docs/`

## Installation

The package is published from GitHub releases. Until the first public
release, install it
from the repository root:

```bash
pip install -e .
```

Optional extras enable vector search and embedding features:

```bash
pip install -e .[full]      # installs FAISS + sentence-transformers
```

## Quick Start

```bash
# Build an index from the docs/ directory
mimir index --docs-path docs/

# Search for content (text output is the default)
mimir search "deployment guide"

# Request a quick summary stitched together from the top matches
mimir ask "How do I configure authentication?"

# Inspect cached metadata
mimir status
```

By default Mímir writes cache files to `.cache/mimir` relative to the project
root.
Everything is controlled through a small YAML configuration file (see
`config/default.yaml`). Pass a custom configuration via the `--config` option:

```bash
mimir --config path/to/mimir.yaml index --docs-path docs/
```

A minimal configuration looks like this:

```yaml
cache_dir: .cache/mimir
vector_search:
  enabled: false
```

## Optional Features

- **Vector search** – install the `full` extra (`pip install -e .[full]`)
  and flip `vector_search.enabled` to `true`. The first run downloads
  embedding models; ensure the machine has network access and enough disk
  space.
- **LLM-assisted answers** – export `OPENROUTER_API_KEY` when advanced
  enrichment is enabled. Optional for the core search workflow.

## Verification

- `mimir status` reports the index path and document count.
- `just docs check` completes without errors (Markdown lint + index rebuild).
- The `docs-quality` GitHub Action succeeds on documentation changes.
- `python -m pytest -q` passes.

## Development

This repository uses `just` recipes for repeatable tasks:

```bash
just install    # editable install with extras
just lint       # Ruff checks
just test       # pytest test-suite
just build      # build wheel + sdist
```

The CI pipeline mirrors these steps and must succeed before publishing a release.

### Release automation

- Merges to `main` run the `CI` workflow first. If it succeeds, the `Release`
  workflow runs automatically.
- The release workflow bumps the version, tags, and publishes a GitHub release
  **only** when the head commit message starts with `feat:` or `fix:` or contains
  `BREAKING`.
- The workflow commits the new version back to `main`, so avoid stacking
  multiple release-triggering commits without letting automation finish.
- If the release workflow is skipped, push a qualifying commit message or
  trigger a rerun after fixing CI.

If you need to install private dependencies (e.g. `llm-cli-tools-core`) inside
CI, follow
the instructions in `docs/PRIVATE_DEPENDENCIES.md` to set up the shared
organization-level GitHub App and workflow snippet.

## Related Docs

- `docs/index.md` — documentation taxonomy, owners, and lifecycle.
- `docs/README.docsearch.md` — full CLI reference.
- `docs/PRIVATE_DEPENDENCIES.md` — CI configuration for private packages.
- `docs/SECURITY_FIXES.md` — security remediation log.
- `CHANGELOG.md` — release history.
