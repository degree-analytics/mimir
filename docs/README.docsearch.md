# Mímir CLI Reference

The Mímir CLI wraps the indexing and search helpers provided by the core library. Every
command accepts an optional `--config` flag that points to a YAML configuration file. If a
configuration is not supplied, the packaged default (`config/default.yaml`) is used.

## Commands

### Build an index

```bash
mimir index --docs-path docs/
```

Options:

- `--docs-path` – Directory that contains Markdown/Text documentation (default: `docs/`).
- `--force` – Rebuild even if cache files already exist.

### Search documentation

```bash
mimir search "authentication" --limit 5 --format json
```

Options:

- `--limit` – Maximum number of results (default: `5`).
- `--format` – Output format (`text`, `json`, or `paths`).
- `--verbose` – Print additional diagnostics while indexing/searching.

### Ask a question

```bash
mimir ask "How do I provision the staging database?" --context-limit 3
```

The command runs a normal search and prints the top results in a short bulleted summary.
It is intended for quick reminders while working inside a repository.

### Inspect index metadata

```bash
mimir status
```

Displays the location of the cached index, the number of documents indexed, and the most
recent build statistics.

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

When `vector_search.enabled` is `true`, make sure the optional dependencies are installed
(`pip install mimir-docsearch[full]`) and that the embedding models are available.
