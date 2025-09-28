from __future__ import annotations

from typing import Iterable, Sequence, Dict, Any, List, Tuple
import math


def precision_at_k(results: Sequence[str], relevant: Iterable[str], k: int) -> float:
    relevant_set = set(relevant)
    if not k:
        return 0.0
    hits = sum(1 for path in results[:k] if path in relevant_set)
    return hits / k


def recall_at_k(results: Sequence[str], relevant: Iterable[str], k: int) -> float:
    relevant_set = set(relevant)
    if not relevant_set:
        return 0.0
    hits = sum(1 for path in results[:k] if path in relevant_set)
    return hits / len(relevant_set)


def reciprocal_rank(results: Sequence[str], relevant: Iterable[str]) -> float:
    relevant_set = set(relevant)
    for idx, path in enumerate(results, start=1):
        if path in relevant_set:
            return 1.0 / idx
    return 0.0


def mean_reciprocal_rank(result_sets: Sequence[Tuple[Sequence[str], Iterable[str]]]) -> float:
    if not result_sets:
        return 0.0
    return sum(reciprocal_rank(results, relevant) for results, relevant in result_sets) / len(result_sets)


def ndcg_at_k(results: Sequence[str], relevant: Iterable[str], k: int) -> float:
    relevant_set = set(relevant)
    dcg = 0.0
    for idx, path in enumerate(results[:k], start=1):
        if path in relevant_set:
            dcg += 1.0 / math.log2(idx + 1)
    ideal_hits = min(len(relevant_set), k)
    if ideal_hits == 0:
        return 0.0
    idcg = sum(1.0 / math.log2(idx + 1) for idx in range(1, ideal_hits + 1))
    return dcg / idcg if idcg else 0.0


def overlap_ratio(a: Sequence[str], b: Sequence[str]) -> float:
    if not a or not b:
        return 0.0
    set_a = set(a)
    set_b = set(b)
    return len(set_a & set_b) / len(set_a | set_b)


def kendall_tau(a: Sequence[str], b: Sequence[str]) -> float:
    if len(a) != len(b) or len(a) < 2:
        return 1.0
    index_map = {doc: idx for idx, doc in enumerate(b)}
    concordant = discordant = 0
    for i in range(len(a)):
        for j in range(i + 1, len(a)):
            ai, aj = a[i], a[j]
            bi = index_map.get(ai)
            bj = index_map.get(aj)
            if bi is None or bj is None:
                continue
            if (i < j and bi < bj) or (i > j and bi > bj):
                concordant += 1
            else:
                discordant += 1
    total = concordant + discordant
    if total == 0:
        return 1.0
    return (concordant - discordant) / total


def aggregate_metrics(per_query: List[Dict[str, Any]]) -> Dict[str, float]:
    if not per_query:
        return {"precision_at_5": 0.0, "recall_at_10": 0.0, "ndcg_at_10": 0.0, "mrr": 0.0}

    precision = sum(item["precision_at_5"] for item in per_query) / len(per_query)
    recall = sum(item["recall_at_10"] for item in per_query) / len(per_query)
    ndcg = sum(item["ndcg_at_10"] for item in per_query) / len(per_query)
    mrr = mean_reciprocal_rank(
        [
            (item["result_paths"], item["relevant_paths"])
            for item in per_query
        ]
    )
    return {
        "precision_at_5": precision,
        "recall_at_10": recall,
        "ndcg_at_10": ndcg,
        "mrr": mrr,
    }


def percentile(values: Sequence[float], pct: float) -> float:
    if not values:
        return 0.0
    if pct <= 0:
        return min(values)
    if pct >= 100:
        return max(values)
    sorted_vals = sorted(values)
    k = (len(sorted_vals) - 1) * (pct / 100)
    f = math.floor(k)
    c = math.ceil(k)
    if f == c:
        return sorted_vals[int(k)]
    d0 = sorted_vals[int(f)] * (c - k)
    d1 = sorted_vals[int(c)] * (k - f)
    return d0 + d1
