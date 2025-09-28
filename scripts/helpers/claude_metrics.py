#!/usr/bin/env python3
"""Parse Claude execution metrics and emit GitHub output variables.

This script is intentionally defensive: the execution JSON that Anthropic's
workflow emits can change format, so we attempt to discover the fields we need
(input tokens, output tokens, cost, selected model) across a handful of common
shapes. If a value cannot be located we simply fall back to "unknown" rather
than failing the workflow.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from collections.abc import Iterable
from typing import Any, Dict, Optional

TOKEN_KEYS_IN = {
    "inputtokens",
    "input_tokens",
    "metrics.input_tokens",
    "inputtokenscount",
}
TOKEN_KEYS_OUT = {
    "outputtokens",
    "output_tokens",
    "metrics.output_tokens",
    "outputtokenscount",
}
MODEL_KEYS = {"model", "modelname", "selected_model"}
COST_KEYS = {"estimatedcostusd", "estimated_cost_usd", "cost_usd", "usd_cost"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Emit Claude metrics in GitHub output format")
    parser.add_argument("execution_file", help="Path to Anthropic execution JSON file")
    parser.add_argument("--costs-json", dest="costs_json", default="", help="Inline JSON mapping models to cost data")
    parser.add_argument("--costs-file", dest="costs_file", default="", help="Path to JSON file with cost data")
    parser.add_argument("--fallback-in-rate", dest="fallback_in", default="", help="Fallback input cost per MTok")
    parser.add_argument("--fallback-out-rate", dest="fallback_out", default="", help="Fallback output cost per MTok")
    return parser.parse_args()


def load_json_payload(path: str) -> Optional[Any]:
    if not os.path.exists(path):
        return None
    with open(path, "r", encoding="utf-8") as fh:
        text = fh.read().strip()
    if not text:
        return None
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        # Try newline-delimited JSON as a fallback.
        items = []
        for line in text.splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                items.append(json.loads(line))
            except json.JSONDecodeError:
                continue
        if items:
            return items
        return None


def iter_nodes(obj: Any) -> Iterable[Any]:
    if isinstance(obj, dict):
        yield obj
        for value in obj.values():
            yield from iter_nodes(value)
    elif isinstance(obj, list):
        for item in obj:
            yield from iter_nodes(item)


def normalise_key(key: str) -> str:
    return key.replace("-", "_").replace(" ", "_").lower()


def extract_metrics(payload: Any) -> Dict[str, Any]:
    metrics: Dict[str, Any] = {
        "model": "unknown",
        "in_tokens": 0.0,
        "out_tokens": 0.0,
        "est_cost_usd": None,
    }

    if payload is None:
        return metrics

    for node in iter_nodes(payload):
        if not isinstance(node, dict):
            continue
        for raw_key, value in node.items():
            key = normalise_key(str(raw_key))
            if key in MODEL_KEYS and isinstance(value, str) and metrics["model"] == "unknown":
                metrics["model"] = value
            if key in COST_KEYS and isinstance(value, (int, float)):
                metrics["est_cost_usd"] = float(value)
            if key in TOKEN_KEYS_IN and isinstance(value, (int, float)):
                metrics["in_tokens"] += float(value)
            if key in TOKEN_KEYS_OUT and isinstance(value, (int, float)):
                metrics["out_tokens"] += float(value)

        # Some providers nest tokens underneath a "usage" or "metrics" dict
        if "usage" in node and isinstance(node["usage"], dict):
            usage = node["usage"]
            for key, value in usage.items():
                nkey = normalise_key(str(key))
                if nkey in TOKEN_KEYS_IN and isinstance(value, (int, float)):
                    metrics["in_tokens"] += float(value)
                if nkey in TOKEN_KEYS_OUT and isinstance(value, (int, float)):
                    metrics["out_tokens"] += float(value)
        if "metrics" in node and isinstance(node["metrics"], dict):
            for key, value in node["metrics"].items():
                nkey = normalise_key(str(key))
                if nkey in TOKEN_KEYS_IN and isinstance(value, (int, float)):
                    metrics["in_tokens"] += float(value)
                if nkey in TOKEN_KEYS_OUT and isinstance(value, (int, float)):
                    metrics["out_tokens"] += float(value)

    return metrics


def load_costs(costs_json: str, costs_file: str) -> Dict[str, Any]:
    data: Dict[str, Any] = {}
    sources: list[str] = []
    if costs_json:
        try:
            data.update(json.loads(costs_json))
        except json.JSONDecodeError:
            pass
    if costs_file and os.path.exists(costs_file):
        try:
            with open(costs_file, "r", encoding="utf-8") as fh:
                file_data = json.load(fh)
            if isinstance(file_data, dict):
                data.update(file_data)
        except (json.JSONDecodeError, OSError):
            pass
    return data


def extract_rate(cost_entry: Any, kind: str) -> Optional[float]:
    if not isinstance(cost_entry, dict):
        return None
    candidates = [
        f"{kind}_per_mtok",
        f"{kind}_per_million_tokens",
        f"{kind}_usd_per_mtok",
        f"{kind}_usd_per_token",
        f"{kind}_per_token",
        kind,
    ]
    for key, value in cost_entry.items():
        if not isinstance(value, (int, float)):
            continue
        norm_key = normalise_key(str(key))
        for candidate in candidates:
            if candidate in norm_key:
                return float(value)
    return None


def compute_cost(metrics: Dict[str, Any], costs: Dict[str, Any], fallback_in: str, fallback_out: str) -> Optional[float]:
    if isinstance(metrics.get("est_cost_usd"), (int, float)):
        return float(metrics["est_cost_usd"])

    in_tokens = float(metrics.get("in_tokens") or 0)
    out_tokens = float(metrics.get("out_tokens") or 0)
    if in_tokens == 0 and out_tokens == 0:
        return None

    model = metrics.get("model", "unknown")
    entry = None
    if isinstance(costs, dict):
        entry = costs.get(model) or costs.get(model.lower())

    in_rate = extract_rate(entry, "input") if entry else None
    out_rate = extract_rate(entry, "output") if entry else None

    def parse_rate(value: str) -> Optional[float]:
        if not value:
            return None
        try:
            return float(value)
        except ValueError:
            return None

    if in_rate is None:
        in_rate = parse_rate(fallback_in)
    if out_rate is None:
        out_rate = parse_rate(fallback_out)

    if in_rate is None and out_rate is None:
        return None

    in_cost = (in_tokens / 1_000_000.0) * (in_rate or 0.0)
    out_cost = (out_tokens / 1_000_000.0) * (out_rate or 0.0)
    total = in_cost + out_cost
    return round(total, 6)


def emit_output(metrics: Dict[str, Any]) -> None:
    model = metrics.get("model", "unknown") or "unknown"
    in_tokens = int(round(float(metrics.get("in_tokens") or 0.0)))
    out_tokens = int(round(float(metrics.get("out_tokens") or 0.0)))
    est_cost = metrics.get("est_cost_usd")

    print(f"model={model}")
    print(f"in_tokens={in_tokens}")
    print(f"out_tokens={out_tokens}")
    if isinstance(est_cost, (int, float)):
        print(f"est_cost_usd={est_cost:.6f}")
    else:
        print("est_cost_usd=")


def main() -> int:
    args = parse_args()
    payload = load_json_payload(args.execution_file)
    metrics = extract_metrics(payload)

    costs = load_costs(args.costs_json, args.costs_file)
    est_cost = compute_cost(metrics, costs, args.fallback_in, args.fallback_out)
    if est_cost is not None:
        metrics["est_cost_usd"] = est_cost

    emit_output(metrics)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
