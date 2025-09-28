# Changelog

## v1.0.0 - 2025-09-27

- Replaced the experimental CLI with a streamlined, well-tested command set
  (`mimir index`, `mimir search`, `mimir ask`, `mimir status`).
- Added pytest coverage for the indexer, search engine, and CLI workflows.
- Disabled vector search by default while keeping optional extras available when
  embeddings are installed.
- Documented installation, configuration, and usage in the refreshed README and CLI
  reference docs.
- Introduced release automation scaffolding (`__version__`, packaging metadata) in
  preparation for GitHub Actions driven tagging.
