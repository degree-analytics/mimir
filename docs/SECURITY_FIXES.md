---
purpose: "Track security fixes, regressions, and verification steps for each release"
audience: "Security reviewers, release engineers"
owner: "Security"
review: "2025-10-01 (Quarterly)"
status: "Active"
---

# Security Fixes and Improvements — Mímir Tool

## When to Use This

- Preparing release notes or audits focused on security and path
  resolution
- Verifying critical fixes before deployment

## Prerequisites

- Local checkout at the tagged release
- Ability to run `python -m pytest`

## v1.0.0 - 2025-09-27

### Critical Path Resolution Bugs (Resolved)

#### Issue 1: Incorrect Project Root Calculation in CLI Entrypoint

- **Problem**: Earlier versions only climbed one directory level when resolving
  project root.
- **Fix**: Updated to rely on `MIMIR_PROJECT_ROOT` and resolve from package
  settings helpers.
- **Impact**: Prevents `.env` loading failures and import path issues.

#### Issue 2: Wrong Cache Directory Path in `indexer.py`

- **Problem**: Cache was created in the wrong location due to incorrect path
  calculation.
- **Fix**: Updated to use `Path(__file__).parent.parent.parent.parent.parent`
  for the correct project root.
- **Impact**: Ensures cache lives in the project root, not random
  directories.

#### Issue 3: Config File Path Resolution in `search.py`

- **Problem**: Looked for `config.yaml` in `src/core/` instead of `mimir/`.
- **Fix**: Changed to `Path(__file__).parent.parent.parent / "config.yaml"`.
- **Impact**: Prevents configuration loading failures.

#### Issue 4: Attribute Access Errors in `search.py`

- **Problem**: Accessed non-existent attributes `self.vectorizer` and
  `self.inverted_index`.
- **Fix**: Updated to use `self.tfidf_strategy.vectorizer` and
  `self.tfidf_strategy.inverted_index`.
- **Impact**: Prevents runtime `AttributeError` exceptions.

#### Issue 5: Duplicate Method Definitions

- **Problem**: Duplicate `_get_exact_match_scores` in `DocumentSearchEngine`.
- **Fix**: Removed the duplicate method referencing non-existent attributes.
- **Impact**: Eliminates code duplication and confusion.

### Enhanced Security Protections

- Added prompt-injection patterns: `\bstop\s+being\b`, `\byou\s+must\s+now\b`,
  `\bchange\s+your\s+instructions\b`, `\bignore\s+all\s+rules\b`,
  `\bbreak\s+character\b`, `\bdeveloper\s+mode\b`,
  `\badmin\s+override\b`, `\bsystem\s+prompt\b`,
  `\bfrom\s+now\s+on\b`, `\bbegin\s+new\s+conversation\b`,
  `\breset\s+conversation\b`.
- **Impact**: Protects against advanced prompt-injection attempts and keeps
  response validation intact.

### Test Coverage

- `tests/test_path_resolution.py` — verifies project root, cache directory,
  config loading, environment files, and path traversal protections.
- `tests/test_security.py` — validates prompt injection defenses, preserves
  legitimate queries, enforces input limits, and escapes special characters.
- `tests/test_integration.py` — covers end-to-end indexing/search workflows,
  error handling, configuration loading, and cache reuse.

## Verification

- `python -m pytest tests/test_security.py -q`
- `python -m pytest tests/test_path_resolution.py -q`
- `just docs check`
- `mimir status` shows cache in `.cache/mimir`

## Deployment Checklist

- [ ] Run full test suite: `python -m pytest tests/`
- [ ] Verify path resolution: `python -c "from src.core.indexer import
  DocumentIndexer"`
- [ ] Test basic functionality: `mimir search "test" --limit 2`
- [ ] Check cache creation in project root
- [ ] Validate environment loading for `.env`
- [ ] Validate prompt injection protections with malicious inputs

## References

- `docs/index.md`
- `../spacewalker/docs/claude-components/verification-standards.md` — example
  of policy-format verification expectations.
