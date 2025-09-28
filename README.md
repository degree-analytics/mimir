# Mímir Docsearch

Mímir is a lightweight documentation indexer and search CLI. It turns a directory of
Markdown, text, or reStructuredText files into a searchable cache that can be queried from
any shell.

## Installation

The package is published from GitHub releases. Until the first public release, install it
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

By default Mímir writes cache files to `.cache/mimir` relative to the project root.
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

- **Vector search** – install the `full` extra (`pip install -e .[full]`) and flip
  `vector_search.enabled` to `true`. The first run will download embedding models; make
  sure the machine has network access and enough disk space.
- **LLM-assisted answers** – the local telemetry helper expects an
  `OPENROUTER_API_KEY` environment variable when advanced enrichment is enabled. This is
  optional and not required for the core search workflow.

## Development

This repository uses `just` recipes for repeatable tasks:

```bash
just install    # editable install with extras
just lint       # Ruff checks
just test       # pytest test-suite
just build      # build wheel + sdist
```

The CI pipeline mirrors these steps and must succeed before publishing a release.

If you need to install private dependencies (e.g. `llm-cli-tools-core`) inside CI, follow
the instructions in `docs/PRIVATE_DEPENDENCIES.md` to set up the shared organization-level
SSH key and workflow snippet.

## Telemetry

Telemetry hooks are currently local to this repository. Once the shared
`llm-cli-tools-core` package is available, Mímir will import the shared telemetry helper
instead of the local compatibility module.
