---
purpose: "{Explain contributor workflows, guardrails, and automation}"
audience: "{Maintainers, contributors, reviewers}"
owner: "{Docs owner}"
review: "{YYYY-MM-DD (Quarterly)}"
status: draft
---

# {Repo} Contributor Guide

## When to Use This

- {Preparing a change}
- {Running CI locally}

## Prerequisites

- `just` installed
- Access to required GitHub secrets

## Core Principles

1. **{Rule}** – {Reason}
2. **{Rule}** – {Reason}

## Development Workflow

1. Branch from `main` with `just git create-branch`
2. Run lint: `just lint`
3. Run tests: `just test`
4. Submit PR via `just git submit`

## Verification

- CI status ✅ on GitHub
- `just docs check` passes locally

## References

- `[Docs Conventions](docs/index.md)`
- `[Release Process](CHANGELOG.md)`
