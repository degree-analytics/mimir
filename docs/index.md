---
purpose: "Inventory of Mímir documentation, owners, and lifecycle"
audience: "Maintainers, contributors, release managers"
owner: "Docs Guild"
review: "2025-10-01 (Quarterly)"
status: "Active"
---

# Mímir Documentation Index

## When to Use This

- Find canonical docs for a task before writing new content
- Identify stale or missing documentation types

## Prerequisites

- Local checkout of this repository
- `just` command runner installed

## Doc Types

- **Overview** (Active) — `README.md`; Docs—template baseline.
- **CLI** (Active) — `docs/README.docsearch.md`; CLI—release sync.
- **CI/Ops** (Active) — `docs/PRIVATE_DEPENDENCIES.md`; Platform—App flow.
- **Security** (Active) — `docs/SECURITY_FIXES.md`; Security—fix log.
- **Changelog** (Active) — `CHANGELOG.md`; Release—semver history.
- **Contributor** (Active) — `CLAUDE.md`; Docs—conventions checklist.
- **Onboarding** (Planned) — `_Placeholder_`; Docs—use `.docs-templates/how-to.md`.
- **Architecture** (Planned) — `_Placeholder_`; Docs—model after
  `../spacewalker/docs/architecture/system-architecture.md`.

## Review Cadence

- Update metadata `Review:` dates during quarterly planning.
- Run `just docs check` before merging doc PRs (enforced by the `docs-quality`
  workflow).

## Maintenance Checklist

- Run `just docs check` (markdownlint + cspell + lychee) before merging doc PRs
  (enforced by the `docs-quality` workflow).
- Ensure Node.js 18+ and a Rust toolchain (lychee) are available locally to run
  `scripts/docs_phase2.sh`.
- Spell/link automation lives in `scripts/docs_phase2.sh` (shared wrapper).
- Update `owner` and `review` metadata whenever stewardship changes.
- Mark deprecated docs with `status: "Deprecated"` and link to the canonical
  replacement.
- Add new docs using the templates in `.docs-templates/` to keep metadata and
  layout consistent.

## References

- Shared templates in `.docs-templates/`
- Spacewalker exemplars:
  - `../spacewalker/docs/setup/getting-started.md` — comprehensive onboarding
    structure.
  - `../spacewalker/docs/architecture/system-architecture.md` — architecture
    narrative with cross-links.
  - `../spacewalker/docs/workflows/documentation-search-guide.md` — runbook
    with verification patterns.

## Outstanding Gaps

- Add onboarding guide covering installation + first index build (planned
  Q4).
- Publish architecture notes once telemetry refactor lands.
