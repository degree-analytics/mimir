#!/usr/bin/env python3
import argparse
import os
import sys
import textwrap
import xml.etree.ElementTree as ET
from datetime import datetime

IDENTIFIER = "<!-- identifier: coverage-report -->"


def parse_line_rate(path: str) -> float | None:
    if not os.path.exists(path):
        return None
    try:
        tree = ET.parse(path)
        root = tree.getroot()
        rate_str = root.get("line-rate")
        if rate_str is None:
            return None
        return float(rate_str) * 100.0
    except Exception as exc:  # pragma: no cover - defensive
        print(f"warning: failed to parse coverage xml ({exc})", file=sys.stderr)
        return None


def build_comment(coverage_pct: float | None, sha: str, coverage_file: str) -> str:
    timestamp = datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S UTC")
    if coverage_pct is None:
        coverage_line = "| Lines | N/A |"
        headline = "⚠️ Coverage data unavailable"
    else:
        rounded = f"{coverage_pct:.1f}%"
        coverage_line = f"| Lines | {rounded} |"
        headline = "📈 Test coverage results"

    body = textwrap.dedent(
        f"""
        {IDENTIFIER}

        ## {headline}

        | Metric | Value |
        | --- | --- |
        {coverage_line}

        - Source: `{os.path.basename(coverage_file)}`
        - Commit: `{sha}`
        - Generated: {timestamp}

        _Run `python -m pytest --cov=src --cov-report=term-missing` locally to reproduce this report._
        """
    ).strip()
    return body + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description="Create a coverage summary comment")
    parser.add_argument("--coverage-file", default="coverage.xml", help="Path to coverage XML file")
    parser.add_argument("--output-file", required=True, help="Path to write the rendered comment")
    args = parser.parse_args()

    sha = os.environ.get("GITHUB_SHA", "unknown")
    coverage_pct = parse_line_rate(args.coverage_file)

    os.makedirs(os.path.dirname(args.output_file), exist_ok=True)
    comment = build_comment(coverage_pct, sha, args.coverage_file)
    with open(args.output_file, "w", encoding="utf-8") as fh:
        fh.write(comment)

    print(f"wrote coverage comment to {args.output_file}")
    if coverage_pct is not None:
        print(f"coverage: {coverage_pct:.2f}%")
    else:
        print("coverage data unavailable")
    return 0


if __name__ == "__main__":
    sys.exit(main())
