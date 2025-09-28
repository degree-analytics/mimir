#!/usr/bin/env python3
"""Render docs-search summaries into a markdown comparison table."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Any, List


def load_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text())


def format_metric_row(name: str, value: float, guardrail: float | None, higher_is_better: bool = True) -> str:
    if guardrail is None:
        status = "ℹ️"
    else:
        if higher_is_better:
            passed = value >= guardrail
        else:
            passed = value <= guardrail
        status = "✅" if passed else "❌"
    guardrail_str = f"{guardrail:.2f}" if guardrail is not None else "—"
    return f"| {name} | {value:.2f} | {guardrail_str} | {status} |"


def render_regression(summary: Dict[str, Any], guardrails: Dict[str, float]) -> List[str]:
    metrics = summary.get("metrics", {})
    latency = summary.get("latency_ms", {})
    highlight = summary.get("highlight_success_rate")

    lines = [f"### {summary.get('config', 'docs-search')}", "", "| Metric | Value | Guardrail | Status |", "|---|---|---|---|"]
    lines.append(format_metric_row("Precision@5", metrics.get("precision_at_5", 0.0), guardrails.get("precision_at_5")))
    lines.append(format_metric_row("Recall@10", metrics.get("recall_at_10", 0.0), guardrails.get("recall_at_10")))
    lines.append(format_metric_row("nDCG@10", metrics.get("ndcg_at_10", 0.0), guardrails.get("ndcg_at_10")))
    if highlight is not None:
        lines.append(format_metric_row("Highlight success", highlight, guardrails.get("highlight_success_rate")))
    for mode, key in (("fast", "fast_p95_ms"), ("smart", "smart_p95_ms"), ("accurate", "accurate_p95_ms")):
        entry = latency.get(mode)
        if entry:
            lines.append(format_metric_row(f"{mode.title()} p95 (ms)", entry.get("p95", 0.0), guardrails.get(key), higher_is_better=False))
    lines.append("")
    return lines


def render_chat(summary: Dict[str, Any], guardrails: Dict[str, float]) -> List[str]:
    success_rate = summary.get("success_rate", 0.0)
    latency = summary.get("latency_ms", {})
    lines = [f"### {summary.get('config', 'docs-search')} – Conversations", "", "| Metric | Value | Guardrail | Status |", "|---|---|---|---|"]
    lines.append(format_metric_row("Conversation success", success_rate, guardrails.get("chat_success_rate")))
    if latency:
        lines.append(format_metric_row("Chat p95 (ms)", latency.get("p95", 0.0), guardrails.get("smart_p95_ms"), higher_is_better=False))
    lines.append("")
    return lines


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inputs", nargs="+", required=True, help="Summary JSON files to render.")
    parser.add_argument("--baseline", help="Optional baseline guardrail JSON to merge.")
    parser.add_argument("--output", help="Optional output markdown file.")
    args = parser.parse_args()

    guardrails: Dict[str, float] = {}
    if args.baseline:
        baseline_path = Path(args.baseline)
        if baseline_path.exists():
            guardrails.update(load_json(baseline_path).get("guardrails", {}))

    markdown_lines: List[str] = []
    for input_path in args.inputs:
        summary_path = Path(input_path)
        if not summary_path.exists():
            raise FileNotFoundError(f"Summary file not found: {summary_path}")
        summary = load_json(summary_path)
        if "metrics" in summary:
            markdown_lines.extend(render_regression(summary, guardrails))
        elif "success_rate" in summary:
            markdown_lines.extend(render_chat(summary, guardrails))

    markdown = "\n".join(markdown_lines).strip() + "\n"
    if args.output:
        Path(args.output).write_text(markdown)
    print(markdown)


if __name__ == "__main__":
    main()
