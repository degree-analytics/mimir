---
purpose: "Overview and quick start for the Mímir documentation search CLI"
audience: "Developers integrating docs search; repository maintainers"
owner: "Docs Guild"
review: "2025-10-01 (Quarterly)"
status: "Active"
---

# Mímir Docsearch

Mímir is a command-line tool that builds a searchable index of your local
documentation tree. It returns precise text snippets, can layer on vector
similarity search for fuzzier matches, and stitches together quick summaries to
help teams answer support questions without digging through pages manually. The
project takes its name from Mímir, the wise counselor from Norse mythology who
guarded a well of knowledge—an apt metaphor for keeping shared docs discoverable.

## When to Use This

- Setting up the CLI locally to index and search a documentation corpus
- Pointing teammates to the canonical install and quick start flow
- Explaining why Mímir accelerates doc lookups for support, enablement, and dev teams

## Prerequisites

- Python 3.11+
- `just` command runner (`brew install just` on macOS)
- Local clone of the repository and access to `docs/`

## Installation

We will publish release artifacts to GitHub once the first stable version ships;
until then, install the CLI straight from the repository root:

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

# Check the installed version
mimir version
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

Need the full command reference? Run `mimir --help` or see
`docs/README.docsearch.md` for subcommand details and usage examples.

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

After the editable install, use the `just` recipes to keep development tasks
fast and repeatable:

```bash
just install    # editable install with extras
just lint       # Ruff checks
just test       # pytest test-suite
just test spacewalker  # real-world integration suite
just build      # build wheel + sdist
```

Branch and PR management flows through Graphite’s `gt` CLI (`gt create`,
`gt modify`, `gt submit`) – avoid raw `git push`/`git rebase`. Stay in the TDD
loop: add a failing test, implement the change, and rerun `just test` until it
passes before submission.

The CI pipeline mirrors these steps and must succeed before publishing a release.

### Release automation

- Merges to `main` run the `CI` workflow first. If it succeeds, the `Release`
  workflow runs automatically.
- The release workflow bumps the version, tags, and publishes a GitHub release
  **only** when the head commit message starts with `feat:` or `fix:` or contains
  `BREAKING`. These markers control the Semantic Versioning bump: `fix:`
  increments the patch number, `feat:` increments the minor number, and
  including `BREAKING` anywhere triggers a major release.
- The workflow commits the new version back to `main`, so avoid stacking
  multiple release-triggering commits without letting automation finish.
- Skip manual edits to `pyproject.toml` or `src/mimir/__version__.py`; the
  workflow updates those files and pushes a `chore: bump version …` commit when
  it runs.
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
