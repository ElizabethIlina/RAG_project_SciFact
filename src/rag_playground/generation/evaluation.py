"""Автоматическая сводка без субъективной LLM-as-a-judge оценки."""

from __future__ import annotations

import json
from pathlib import Path
from statistics import fmean
from typing import Any


def evaluate_answer_file(path: Path) -> dict[str, Any]:
    """Посчитать токены, ссылки и наличие релевантных документов в контексте."""
    rows = [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if not rows:
        raise ValueError("Answer file must not be empty")
    rag_rows = [row for row in rows if row["mode"] == "rag"]
    return {
        "answers": len(rows),
        "rag_answers": len(rag_rows),
        "baseline_answers": len(rows) - len(rag_rows),
        "answers_with_citations": sum(bool(row["cited_source_numbers"]) for row in rows),
        "answers_with_invalid_citations": sum(
            bool(row["invalid_source_numbers"]) for row in rows
        ),
        "rag_contexts_with_relevant_document": sum(
            bool(row["retrieved_relevant_doc_ids"]) for row in rag_rows
        ),
        "mean_total_tokens": round(fmean(row["total_tokens"] for row in rows), 3),
        "total_prompt_tokens": sum(row["prompt_tokens"] for row in rows),
        "total_completion_tokens": sum(row["completion_tokens"] for row in rows),
        "mean_generation_latency_ms": round(
            fmean(row["generation_latency_ms"] for row in rows), 3
        ),
    }


def save_evaluation(summary: dict[str, Any], path: Path) -> None:
    """Сохранить автоматическую сводку рядом с ответами."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
