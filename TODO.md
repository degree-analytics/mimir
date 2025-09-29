# Mímir v1.4.0 Release Checklist

## 1. Verify Code & Tests
- [x] Run `just install` (editable install with extras)
- [x] Run `just lint`
- [x] Run `just test`
- [x] Smoke-test CLI (`mimir index`, `mimir search`, `mimir ask`, `mimir status`)

## 2. Packaging & Versioning
- [x] Update `pyproject.toml` metadata and classifiers for the 1.0 release
- [x] Add `CHANGELOG.md` with v1.0 entry
- [x] Ensure `src/mimir/__version__.py` matches `pyproject.toml`
- [x] Configure GitHub release workflow (auto bump + tag)
- [x] Run `just build` and confirm clean artifacts in `dist/`

## 3. Documentation & Examples
- [x] Refresh `README.md` with install + quick-start guidance
- [x] Update `docs/README.docsearch.md` for the streamlined CLI
- [x] Document optional extras / environment variables (e.g. embeddings, API keys)

## 4. Repo Automation & QA
- [x] Update `.github/workflows/ci.yml` to run install/lint/test/build
- [x] Ensure `CLAUDE.md` / `codex.yml` reflect the new workflow
- [x] Mirror release automation approach from `../llm-cli-tools-core`

## 5. Post-Release Follow-up
- [ ] Replace local telemetry shim with `llm-cli-tools-core` once published
- Update SpaceWalker to depend on the new package + remove legacy doc finder scripts

> Tip: use the Justfile for all commands and rely on the packaged configuration unless you
> need to override cache locations or disable vector search.
