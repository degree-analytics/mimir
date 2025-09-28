"""CLI entry for Mímir."""

from .main import cli


def main() -> None:
    """Console entry point."""
    cli()

__all__ = ["cli", "main"]
