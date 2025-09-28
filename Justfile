set dotenv-load
set shell := ["bash", "-lc"]

docs_directory := "docs"
docs_templates := ".docs-templates"
docs_root := justfile_directory()

_default:
    @just help

help:
    @echo "Mímir commands"
    @echo "--------------"
    @echo "just install        # install package in editable mode"
    @echo "just lint           # run ruff lint checks"
    @echo "just test           # run pytest suite"
    @echo "just build          # build distribution artifacts"
    @echo "just docs check     # lint docs + rebuild index"

install:
    @python3.11 -m pip install --upgrade pip
    @python3.11 -m pip install -e .[full]

lint:
    @python3.11 -m pip install -q ruff
    @python3.11 -m ruff check src tests

lint-fix:
    @python3.11 -m ruff check --fix src tests

test:
    @python3.11 -m pip install -q pytest
    @python3.11 -m pytest -q

build:
    @python3.11 -m pip install -q build
    @python3.11 -m build

docs action='help' *args='':
    @if [ "{{action}}" = "help" ]; then \
        echo "Usage: just docs check"; \
        exit 0; \
    elif [ "{{action}}" = "check" ]; then \
        set -euo pipefail; \
        FILES="{{docs_root}}/README.md {{docs_root}}/CHANGELOG.md {{docs_root}}/CLAUDE.md"; \
        DOC_LIST=$(find "{{docs_root}}/{{docs_directory}}" -name '*.md' -print 2>/dev/null | tr '\n' ' '); \
        TEMPLATE_LIST=$(find "{{docs_root}}/{{docs_templates}}" -name '*.md' -print 2>/dev/null | tr '\n' ' '); \
        if command -v npx >/dev/null 2>&1; then \
            npx --yes markdownlint-cli2@0.12.1 $FILES $DOC_LIST $TEMPLATE_LIST; \
        else \
            echo "⚠️  npx not found; skipping markdownlint-cli2. Install Node.js to enable docs linting."; \
        fi; \
        python3.11 -m mimir.cli.main index --docs-path {{docs_directory}}/ --force; \
        ./scripts/docs_phase2.sh; \
    else \
        echo "Unknown docs action: {{action}}"; \
        exit 1; \
    fi
