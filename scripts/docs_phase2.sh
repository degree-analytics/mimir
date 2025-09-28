#!/usr/bin/env bash
set -euo pipefail

: "${DOCS_CSPELL_FILES:=README.md CHANGELOG.md CLAUDE.md docs/**/*.md .docs-templates/*.md}"
: "${DOCS_LYCHEE_FILES:=README.md CHANGELOG.md CLAUDE.md docs/**/*.md}"
: "${DOCS_LYCHEE_ARGS:=--max-concurrency 4 --timeout 20 --retry-wait-time 2 --no-progress}"
: "${DOCS_CSPELL_CONFIG:=cspell.config.yaml}"
: "${DOCS_LYCHEE_CONFIG:=lychee.toml}"

if ! command -v npx >/dev/null 2>&1; then
  echo "❌ npx not found. Install Node.js 18+ to run spell checking." >&2
  exit 1
fi

if [ ! -f "$DOCS_CSPELL_CONFIG" ]; then
  echo "ℹ️  Generating default cspell config at $DOCS_CSPELL_CONFIG" >&2
  cat <<'CFG' > "$DOCS_CSPELL_CONFIG"
version: "0.2"
language: en
words:
  - Mímir
  - openrouter
  - pushgateway
  - Pushgateway
  - PUSHGATEWAY
  - overengineering
  - justfile
  - automations
  - dataclass
  - Groundtruth
  - Claude
  - Spacewalker
  - pytest
  - pyproject
  - chadwalters
  - getenv
  - commandname
  - fastapi
  - Runbook
  - anthropics
  - MTOK
  - docsearch
  - docsearcher
  - docindex
flagWords: []
ignorePaths:
  - "**/node_modules/**"
  - "**/.git/**"
  - "**/dist/**"
  - "**/.cache/**"
CFG
fi

echo "🧡 Running cspell…"
npx --yes cspell@7.2.0 lint --config "$DOCS_CSPELL_CONFIG" $DOCS_CSPELL_FILES

echo "🌐 Running lychee…"
if ! command -v lychee >/dev/null 2>&1; then
  if command -v cargo-binstall >/dev/null 2>&1; then
    cargo binstall --no-confirm lychee >/dev/null
  elif command -v cargo >/dev/null 2>&1; then
    cargo install lychee >/dev/null
  else
    echo "❌ lychee not found and cargo unavailable. Install lychee manually or add it to PATH." >&2
    exit 1
  fi
fi

if [ -f "$DOCS_LYCHEE_CONFIG" ]; then
  lychee --config "$DOCS_LYCHEE_CONFIG" $DOCS_LYCHEE_ARGS $DOCS_LYCHEE_FILES
else
  lychee $DOCS_LYCHEE_ARGS $DOCS_LYCHEE_FILES
fi

echo "✅ Phase 2 docs checks complete."
