"""Проверка финального анализа на маленьких локальных артефактах."""

from __future__ import annotations

import csv
import json
from pathlib import Path

from rag_playground.reporting.analysis import build_analysis_artifacts
from rag_playground.reporting.report import build_final_report


def write_jsonl(path: Path, row: dict) -> None:
    path.write_text(json.dumps(row) + "\n", encoding="utf-8")


def test_analysis_and_markdown_report_are_built(tmp_path: Path):
    """Таблицы и выводы должны собираться из данных, а не копироваться вручную."""
    metrics = tmp_path / "metrics.json"
    prepared = tmp_path / "prepared.jsonl"
    answers = tmp_path / "answers.jsonl"
    rag_summary = tmp_path / "rag_summary.json"
    manual = tmp_path / "manual.csv"
    analysis_dir = tmp_path / "analysis"
    report = tmp_path / "final.md"

    common_metrics = {
        "recall@1": 0.5,
        "recall@10": 0.8,
        "mrr@10": 0.6,
        "ndcg@10": 0.7,
        "recall@100": 0.9,
    }
    metrics.write_text(
        json.dumps(
            {
                "metrics": {
                    "hybrid_document": {**common_metrics, "ndcg@10": 0.73},
                    "chunk_recursive": {**common_metrics, "ndcg@10": 0.72},
                }
            }
        ),
        encoding="utf-8",
    )
    write_jsonl(
        prepared,
        {
            "query_id": "q1",
            "retrieved_relevant_doc_ids": ["d1"],
            "sources": [{"doc_id": "d1"}],
        },
    )
    write_jsonl(
        answers,
        {
            "query_id": "q1",
            "query_text": "Claim",
            "answer": "Answer [1]",
            "cited_source_numbers": [1],
            "invalid_source_numbers": [],
            "prompt_tokens": 10,
            "completion_tokens": 5,
            "total_tokens": 15,
            "generation_latency_ms": 20.0,
        },
    )
    rag_summary.write_text(
        json.dumps(
            {
                "answers": 1,
                "rag_contexts_with_relevant_document": 1,
                "answers_with_citations": 1,
                "answers_with_invalid_citations": 0,
                "total_prompt_tokens": 10,
                "total_completion_tokens": 5,
                "mean_generation_latency_ms": 20.0,
            }
        ),
        encoding="utf-8",
    )
    with manual.open("w", encoding="utf-8", newline="") as target:
        writer = csv.writer(target)
        writer.writerow(
            [
                "query_id",
                "mode",
                "correctness_0_2",
                "groundedness_0_2",
                "completeness_0_2",
                "citation_quality_0_2",
                "error_type",
                "notes",
            ]
        )
        writer.writerow(["q1", "rag", "2", "2", "2", "2", "none", "ok"])

    paths = build_analysis_artifacts(
        metrics, prepared, answers, rag_summary, manual, analysis_dir
    )
    build_final_report(analysis_dir, report)
    summary = json.loads(paths["summary"].read_text())

    assert summary["retrieval"]["best_ndcg_at_10"]["method"] == "hybrid_document"
    assert summary["manual_mean_scores"]["groundedness_0_2"] == 2.0
    assert summary["langfuse_hobby"]["estimated_import_units"] == 12
    assert "Hybrid обеспечивает" in report.read_text(encoding="utf-8")
