from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Dict, List

import pytest

from .metrics import percentile

RESULTS_PATH = Path(__file__).parent / "results"
BASELINE_DIR = Path(__file__).parent / "baseline"


@pytest.mark.chat
def test_docs_search_conversations(
    search_engine,
    embedding_config,
    embedding_config_path,
    snapshot_meta,
    conversations_dataset,
    docs_snapshot_dir,
):
    if not getattr(search_engine, "pipeline", None):
        pytest.skip("Pipeline orchestrator not enabled")
    if not getattr(search_engine, "llm_enhancer", None):
        pytest.skip("LLM enhancer unavailable")

    config_meta = embedding_config.get("metadata", {})
    config_name = config_meta.get("name") or embedding_config_path.stem
    baseline_path = BASELINE_DIR / f"{config_name}.json"
    guardrails = {"chat_success_rate": 0.0}
    if baseline_path.exists():
        baseline_data = json.loads(baseline_path.read_text())
        guardrails.update(baseline_data.get("guardrails", {}))

    conversations_summary: List[Dict[str, Any]] = []
    all_latencies_ms: List[float] = []
    pass_count = 0

    for convo in conversations_dataset:
        convo_id = convo["conversation_id"]
        convo_pass = True
        turns_summary: List[Dict[str, Any]] = []

        root = docs_snapshot_dir.resolve()

        for turn in convo.get("turns", []):
            question = turn["question"]
            expected_paths = turn.get("expected_paths", [])

            start = time.perf_counter()
            pipeline_result = search_engine.pipeline.search(question, mode="smart", limit=5)
            latency_s = pipeline_result.execution_time or (time.perf_counter() - start)
            latency_ms = latency_s * 1000
            all_latencies_ms.append(latency_ms)

            results = pipeline_result.results
            if not results:
                results = search_engine.search(question, top_k=5)

            result_paths = []
            for res in results:
                raw_path = Path(res.document_path)
                try:
                    relative = raw_path.resolve().relative_to(root)
                    normalized = str(relative).replace("\\", "/")
                except Exception:
                    normalized = res.document_path
                result_paths.append(normalized)
            if expected_paths:
                retrieval_ok = any(
                    any(path.endswith(exp) or path == exp for path in result_paths)
                    for exp in expected_paths
                )
            else:
                retrieval_ok = bool(result_paths)

            qa_answer = search_engine.llm_enhancer.interactive_qa(
                question,
                [res.to_dict() for res in results],
            )
            expected_answer = turn.get("answer", "").strip().lower()
            answer_ok = expected_answer in (qa_answer or "").lower()

            convo_pass = convo_pass and retrieval_ok and answer_ok
            turns_summary.append(
                {
                    "turn": turn.get("turn"),
                    "question": question,
                    "expected_paths": expected_paths,
                    "result_paths": result_paths,
                    "retrieval_ok": retrieval_ok,
                    "answer": qa_answer,
                    "answer_ok": answer_ok,
                    "latency_ms": latency_ms,
                }
            )

        if convo_pass:
            pass_count += 1

        conversations_summary.append(
            {
                "conversation_id": convo_id,
                "turns": turns_summary,
                "passed": convo_pass,
            }
        )

    success_rate = pass_count / len(conversations_dataset) if conversations_dataset else 1.0

    summary = {
        "config": config_name,
        "commit": snapshot_meta.commit,
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "conversation_count": len(conversations_dataset),
        "success_rate": success_rate,
        "latency_ms": {
            "avg": (sum(all_latencies_ms) / len(all_latencies_ms)) if all_latencies_ms else 0.0,
            "p95": percentile(all_latencies_ms, 95) if all_latencies_ms else 0.0,
        },
        "conversations": conversations_summary,
    }

    results_file = RESULTS_PATH / f"chat_{config_name}.json"
    results_file.write_text(json.dumps(summary, indent=2))

    assert success_rate >= guardrails["chat_success_rate"], (
        f"Conversation success rate {success_rate:.2f} below guardrail {guardrails['chat_success_rate']:.2f}"
    )
