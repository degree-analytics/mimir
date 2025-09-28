---
purpose: "Document the supported Mímir CLI commands and configuration"
audience: "Engineers and operators using the docs search CLI"
owner: "CLI Maintainers"
review: "2025-10-01 (Quarterly)"
status: "Active"
---

# Mímir CLI Reference

## When to Use This

- Look up supported commands, flags, and expected output
- Share canonical examples with teammates rolling out Mímir

## Prerequisites

- Editable install: `pip install -e .[full]` (or run `just install`)
- Access to a docs directory (default `docs/`)

## Commands

### Build an index

```bash
mimir index --docs-path docs/
```

Options:

- `--docs-path` – Directory that contains Markdown/Text documentation
  (default `docs/`).
- `--force` – Rebuild even if cache files already exist.

**Verification**:

Command completes without errors and `mimir status` shows a fresh
timestamp.

### Search documentation

```bash
mimir search "authentication" --limit 5 --format json
```

Options:

- `--limit` – Maximum number of results (default `5`).
- `--format` – Output format (`text`, `json`, or `paths`).
- `--verbose` – Print additional diagnostics while indexing/searching.

**Verification**:

CLI prints ranked matches; `--format json` returns a valid JSON
payload.

### Ask a question

```bash
mimir ask "How do I provision the staging database?" --context-limit 3
```

The command runs a normal search, prints the top results in a bulleted summary,
and is intended for quick reminders while working inside a repository.

**Verification**:

Output contains a summary with source file paths. If no documents
match, the CLI states that relevant docs were not found.

### Inspect index metadata

```bash
mimir status
```

Displays the location of the cached index, the number of documents indexed, and
the most recent build statistics.

**Verification**:

Reports `.cache/mimir` (unless overridden) and shows a non-zero
document count after indexing.

## Configuration

Configuration is expressed as YAML. The most useful settings are:

```yaml
cache_dir: .cache/mimir  # override the location of cached indexes
vector_search:
  enabled: false         # flip to true when embeddings are available locally
telemetry:
  enabled: false         # disable telemetry by default
```

Save the file as `mimir.yaml` (for example) and pass it to the CLI:

```bash
mimir --config mimir.yaml index --docs-path docs/
```

When `vector_search.enabled` is `true`, install optional dependencies
(`pip install mimir-docsearch[full]`) and confirm embedding models exist
locally.

## References

- `docs/index.md` — documentation taxonomy and ownership.
- `../spacewalker/docs/workflows/documentation-search-guide.md` — example
  runbook using `just docs search`.

> **Repo layout**: clone the `spacewalker` repository alongside `mimir`
> (e.g. `/path/to/workspace/mimir` and `/path/to/workspace/spacewalker`). The
> frozen docs snapshot (`tests/data/spacewalker_docs.tar.gz`) is built from that
> sibling checkout.

## Spacewalker Integration Coverage

The integration suite exercises Mímir against a frozen snapshot of the
Spacewalker documentation repo (see `tests/data/spacewalker_docs.tar.gz`). It
lives alongside the rest of the integration tests (`tests/integration/`) and
runs with `just test` by default. We also provide a convenience target for
focused runs:

```bash
just test spacewalker
```

In CI the Spacewalker job runs in parallel after the core lint/unit job,
publishes `spacewalker-metrics.json`, and comments on pull requests with
index/search/telemetry metrics so reviewers can spot regressions in real-world
behaviour quickly.

## Development Workflow

Mímir follows the same Justfile/Graphite workflow used across our internal CLIs:

1. **Create a branch with GT**

   ```bash
   gt create --all -m "feat: short summary"
   ```

2. **Test-driven changes**
   - add or adjust tests first (`tests/`)
   - run `just test` (expect failures), implement, rerun until green
3. **Lint & docs**

   ```bash
   just lint
   just docs check
   ```

4. **Submit via GT**

   ```bash
   gt submit
   ```

5. **Request reviews**
   - comment on the PR tagging `@codex` and `@claude`

Avoid raw `git push`/`git rebase`; if you hit GT conflicts, use `gt modify`
or `gt restack` per the Spacewalker workflow docs.
