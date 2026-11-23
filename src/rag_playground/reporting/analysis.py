"""Объединение retrieval-метрик, RAG-ответов и ручной оценки."""

from __future__ import annotations

import csv
import json
from collections import Counter
from pathlib import Path
from statistics import fmean
from typing import Any

_MANUAL_SCORE_FIELDS = (
    "correctness_0_2",
    "groundedness_0_2",
    "completeness_0_2",
    "citation_quality_0_2",
)


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8") as source:
        return list(csv.DictReader(source))


def _write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as target:
        writer = csv.DictWriter(target, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def build_analysis_artifacts(
    retrieval_metrics_path: Path,
    prepared_path: Path,
    answers_path: Path,
    rag_summary_path: Path,
    manual_path: Path,
    output_dir: Path,
) -> dict[str, Path]:
    """Построить все машинно-читаемые таблицы шестого этапа."""
    retrieval_payload = json.loads(retrieval_metrics_path.read_text(encoding="utf-8"))
    retrieval_metrics = retrieval_payload["metrics"]
    retrieval_rows = [
        {
            "method": method,
            "recall_at_1": values["recall@1"],
            "recall_at_10": values["recall@10"],
            "mrr_at_10": values["mrr@10"],
            "ndcg_at_10": values["ndcg@10"],
            "recall_at_100": values["recall@100"],
        }
        for method, values in retrieval_metrics.items()
    ]
    prepared_by_id = {str(row["query_id"]): row for row in _read_jsonl(prepared_path)}
    answers_by_id = {str(row["query_id"]): row for row in _read_jsonl(answers_path)}
    manual_rows = _read_csv(manual_path)

    rag_rows: list[dict[str, Any]] = []
    error_rows: list[dict[str, Any]] = []
    for manual in manual_rows:
        query_id = str(manual["query_id"])
        if query_id not in prepared_by_id or query_id not in answers_by_id:
            raise ValueError(f"Missing prepared/answer artifact for query {query_id}")
        prepared = prepared_by_id[query_id]
        answer = answers_by_id[query_id]
        row = {
            "query_id": query_id,
            "relevant_in_context": bool(prepared["retrieved_relevant_doc_ids"]),
            "source_count": len(prepared["sources"]),
            "citation_count": len(answer["cited_source_numbers"]),
            "invalid_citation_count": len(answer["invalid_source_numbers"]),
            "prompt_tokens": answer["prompt_tokens"],
            "completion_tokens": answer["completion_tokens"],
            "total_tokens": answer["total_tokens"],
            "generation_latency_ms": answer["generation_latency_ms"],
            **{field: int(manual[field]) for field in _MANUAL_SCORE_FIELDS},
            "error_type": manual["error_type"],
            "notes": manual["notes"],
        }
        rag_rows.append(row)
        if manual["error_type"] != "none":
            error_rows.append(
                {
                    "query_id": query_id,
                    "query_text": answer["query_text"],
                    "error_type": manual["error_type"],
                    "notes": manual["notes"],
                    "relevant_in_context": row["relevant_in_context"],
                    "answer": answer["answer"],
                }
            )

    rag_summary = json.loads(rag_summary_path.read_text(encoding="utf-8"))
    error_counts = Counter(row["error_type"] for row in rag_rows)
    best_ndcg = max(retrieval_rows, key=lambda row: row["ndcg_at_10"])
    chunk_rows = [row for row in retrieval_rows if row["method"].startswith("chunk_")]
    best_chunk = max(chunk_rows, key=lambda row: row["ndcg_at_10"])
    summary = {
        "retrieval": {
            "methods": len(retrieval_rows),
            "best_ndcg_at_10": {
                "method": best_ndcg["method"],
                "value": best_ndcg["ndcg_at_10"],
            },
            "best_chunk_ndcg_at_10": {
                "method": best_chunk["method"],
                "value": best_chunk["ndcg_at_10"],
            },
        },
        "rag": rag_summary,
        "manual_mean_scores": {
            field: round(fmean(row[field] for row in rag_rows), 3)
            for field in _MANUAL_SCORE_FIELDS
        },
        "error_counts": dict(sorted(error_counts.items())),
        "langfuse_hobby": {
            "estimated_import_units": len(rag_rows) * 12,
            "monthly_included_units": 50000,
            "estimated_allowance_percent": round(len(rag_rows) * 12 / 50000 * 100, 4),
            "data_access_days": 30,
        },
    }

    output_dir.mkdir(parents=True, exist_ok=True)
    retrieval_csv = output_dir / "retrieval_comparison.csv"
    rag_csv = output_dir / "rag_evaluation.csv"
    errors_jsonl = output_dir / "stage_6_error_cases.jsonl"
    summary_json = output_dir / "stage_6_summary.json"
    _write_csv(retrieval_csv, retrieval_rows, list(retrieval_rows[0]))
    _write_csv(rag_csv, rag_rows, list(rag_rows[0]))
    with errors_jsonl.open("w", encoding="utf-8") as target:
        for row in error_rows:
            target.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    summary_json.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return {
        "retrieval_comparison": retrieval_csv,
        "rag_evaluation": rag_csv,
        "error_cases": errors_jsonl,
        "summary": summary_json,
    }
