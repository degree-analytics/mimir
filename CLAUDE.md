# Mímir Contributor Guidelines

- Use the `Justfile` commands for every workflow (`just install`, `just lint`, `just test`,
  `just build`). CI mirrors these steps.
- The CLI now exposes four supported commands: `mimir index`, `mimir search`, `mimir ask`,
  and `mimir status`. Keep docs and examples in sync when behaviour changes.
- Configuration lives in small YAML files. Honour the `--config` flag in new features and
  ensure defaults remain safe for offline execution (vector search stays optional).
- Always add or update tests when touching indexing or CLI behaviour. The suite relies on
  the fixtures in `tests/conftest.py`.
- Release automation bumps `pyproject.toml` and `src/mimir/__version__.py`. Conventional
  commit prefixes (`feat:`, `fix:`) decide whether a tag is created.
- Telemetry still uses the local compatibility shim. Plan a follow-up to integrate the
  shared `llm-cli-tools-core` package once it ships.
