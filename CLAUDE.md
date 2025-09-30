---
purpose: "Capture contributor guardrails for the Mímir repository"
audience: "Contributors, reviewers, automation authors"
owner: "Docs Guild"
review: "2025-10-01 (Quarterly)"
status: "Active"
---

# Mímir Contributor Guidelines

## When to Use This

- Before authoring code or documentation changes
- During reviews to confirm conventions are followed

## Prerequisites

- `just` installed locally
- Familiarity with `.docs-templates/` and `docs/index.md`
- Node.js 18+ available locally so `just docs check` can run markdownlint

## Core Expectations

- **Justfile-first** – run repo workflows through `just` (`just install`, `just
  lint`, `just test`, `just build`, `just docs check`) rather than direct
  `pytest`, `pip`, or `ruff` commands.
- **GT workflow only** – manage branches/PRs with Graphite’s `gt` CLI (`gt
  create`, `gt modify`, `gt submit`). Avoid raw `git push`, `git rebase`, or
  `git merge`.
- **TDD loop** – write or update tests before implementation, run them to see
  them fail, implement the change, and rerun until green (`red → green →
  refactor`).
- Keep CLI docs (`docs/README.docsearch.md`) synchronized with behavioural
  changes and update `docs/index.md` taxonomy entries when adding new content.
- Honour YAML configuration defaults and the `--config` flag for new features.
- Add or update tests whenever indexing or CLI behaviour changes (fixtures live
  in `tests/conftest.py`).
- Release automation bumps `pyproject.toml` and `src/mimir/__version__.py`;
  conventional commit prefixes (`feat:`, `fix:`) control tagging.

## Documentation Conventions

1. Start every doc with the shared metadata block (Purpose, Audience, Owner,
   Review, Status) followed by **When to Use This** and **Prerequisites**.
2. Use the templates in `.docs-templates/` when adding new documentation.
3. Reference `docs/index.md` to map each doc to its taxonomy entry; mark new
   docs as Planned/Active as needed.
4. Run `just docs check` before sending a PR and ensure the `docs-quality`
   GitHub Action finishes green.
5. Use examples from `../spacewalker/docs/setup/getting-started.md` and
   `../spacewalker/docs/architecture/system-architecture.md` when shaping
   onboarding or architecture docs.

## File Conventions

- **Temp Files**: Always use `.build/tmp/` (never `/tmp/`)
- **Build Artifacts**: `.build/` and `.dist/` are gitignored

## Automation Alignment

- `codex.yml` mirrors CI (install → lint → test → build).
- `just docs check` performs Markdown linting, runs the `mimir index` smoke
  test, and executes `scripts/docs_phase2.sh` (cspell + lychee). Keep Node.js
  18+ and a Rust toolchain available on developer machines and CI runners.

## Follow-Ups

- Plan the onboarding and architecture docs noted in `docs/index.md`.
- Schedule quarterly doc reviews during roadmap planning to keep `Review:`
  metadata accurate.
