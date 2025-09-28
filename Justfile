set dotenv-load
set shell := ["bash", "-lc"]

_default:
    @just help

help:
    @echo "Mímir commands"
    @echo "--------------"
    @echo "just install   # install package in editable mode"
    @echo "just lint      # run ruff lint checks"
    @echo "just test      # run pytest suite"
    @echo "just build     # build distribution artifacts"

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
