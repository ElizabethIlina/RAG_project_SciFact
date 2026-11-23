"""Идемпотентный импорт восьми сохранённых RAG-запусков в Langfuse."""

from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from langfuse import Langfuse


@dataclass(frozen=True, slots=True)
class BackfillReport:
    """Локальная сводка импорта без секретных или пользовательских ключей."""

    imported: int
    skipped: int
    trace_ids: tuple[str, ...]
    estimated_units: int


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    """Прочитать JSONL и сохранить исходный порядок эксперимента."""
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _read_manual_scores(path: Path) -> dict[str, dict[str, str]]:
    """Индексировать ручную оценку по query_id."""
    with path.open(encoding="utf-8") as source:
        rows = list(csv.DictReader(source))
    return {str(row["query_id"]): row for row in rows}


def _read_state(path: Path) -> set[str]:
    """Получить trace IDs, уже подтверждённо отправленные прошлым запуском."""
    if not path.exists():
        return set()
    payload = json.loads(path.read_text(encoding="utf-8"))
    return {str(value) for value in payload.get("imported_trace_ids", [])}


def _write_state(path: Path, trace_ids: set[str]) -> None:
    """Записать компактный state только после успешного flush."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {"imported_trace_ids": sorted(trace_ids)},
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )


def _trace_one_example(
    client: Langfuse,
    prepared: dict[str, Any],
    answer: dict[str, Any],
    manual: dict[str, str],
    trace_id: str,
) -> None:
    """Создать root trace, четыре observations и шесть дешёвых scores."""
    query_id = str(answer["query_id"])
    source_payload = [
        {
            "source_number": source["source_number"],
            "doc_id": source["doc_id"],
            "chunk_id": source["chunk_id"],
            "rank": source["retrieval_rank"],
            "score": source["retrieval_score"],
        }
        for source in prepared["sources"]
    ]
    with client.start_as_current_observation(
        trace_context={"trace_id": trace_id},
        name="scifact-rag-answer",
        as_type="chain",
        input={"query_id": query_id, "claim": answer["query_text"]},
        output={"answer": answer["answer"]},
        metadata={
            "dataset": "SciFact",
            "split": "test",
            "mode": answer["mode"],
            "historical_import": True,
            "error_type": manual["error_type"],
        },
        version="stage-6-v1",
    ) as root:
        root.set_trace_io(
            input={"query_id": query_id, "claim": answer["query_text"]},
            output={"answer": answer["answer"]},
        )
        with root.start_as_current_observation(
            name="recursive-chunk-retrieval",
            as_type="retriever",
            input={"claim": answer["query_text"], "top_k": len(source_payload)},
            output={"sources": source_payload},
            metadata={
                "strategy": prepared["retrieval_strategy"],
                "latency_ms": prepared["retrieval_latency_ms"],
                "relevant_doc_ids": prepared["relevant_doc_ids"],
                "retrieved_relevant_doc_ids": prepared[
                    "retrieved_relevant_doc_ids"
                ],
            },
        ):
            pass
        with root.start_as_current_observation(
            name="context-building",
            as_type="span",
            input={"sources": source_payload},
            output={"prompt": prepared["prompt"]},
            metadata={"source_count": len(source_payload)},
        ):
            pass
        with root.start_as_current_observation(
            name="gigachat-generation",
            as_type="generation",
            input={"prompt": answer["prompt"]},
            output={"answer": answer["answer"]},
            model=answer["model"],
            model_parameters={"temperature": 0.1, "max_tokens": 512},
            usage_details={
                "input": int(answer["prompt_tokens"]),
                "output": int(answer["completion_tokens"]),
                "total": int(answer["total_tokens"]),
            },
            metadata={
                "latency_ms": answer["generation_latency_ms"],
                "finish_reason": answer["finish_reason"],
            },
        ):
            pass
        with root.start_as_current_observation(
            name="deterministic-evaluation",
            as_type="evaluator",
            input={"answer": answer["answer"], "sources": source_payload},
            output={
                "cited_source_numbers": answer["cited_source_numbers"],
                "invalid_source_numbers": answer["invalid_source_numbers"],
                "manual_scores": {
                    name: int(manual[name])
                    for name in (
                        "correctness_0_2",
                        "groundedness_0_2",
                        "completeness_0_2",
                        "citation_quality_0_2",
                    )
                },
            },
            metadata={"error_type": manual["error_type"], "notes": manual["notes"]},
        ):
            pass

        # Scores являются детерминированными: платный LLM-as-a-judge не используется.
        root.score_trace(
            name="retrieval_relevant_present",
            value=float(bool(prepared["retrieved_relevant_doc_ids"])),
            comment="Document-level SciFact qrels hit in generation context",
        )
        root.score_trace(
            name="citations_valid",
            value=float(not answer["invalid_source_numbers"]),
        )
        for field, score_name in (
            ("correctness_0_2", "correctness"),
            ("groundedness_0_2", "groundedness"),
            ("completeness_0_2", "completeness"),
            ("citation_quality_0_2", "citation_quality"),
        ):
            root.score_trace(
                name=score_name,
                value=float(manual[field]),
                comment=manual["notes"],
            )


def backfill_rag_traces(
    client: Langfuse,
    prepared_path: Path,
    answers_path: Path,
    manual_path: Path,
    state_path: Path,
) -> BackfillReport:
    """Импортировать отсутствующие traces и пропустить уже подтверждённые state."""
    prepared_by_id = {str(row["query_id"]): row for row in _read_jsonl(prepared_path)}
    answers = _read_jsonl(answers_path)
    manual_by_id = _read_manual_scores(manual_path)
    imported_before = _read_state(state_path)
    imported_now: list[str] = []
    skipped = 0

    for answer in answers:
        query_id = str(answer["query_id"])
        if query_id not in prepared_by_id or query_id not in manual_by_id:
            raise ValueError(f"Missing prepared/manual data for query {query_id}")
        trace_id = client.create_trace_id(seed=f"rag-playground/stage-5/rag/{query_id}")
        if trace_id in imported_before:
            skipped += 1
            continue
        _trace_one_example(
            client,
            prepared_by_id[query_id],
            answer,
            manual_by_id[query_id],
            trace_id,
        )
        imported_now.append(trace_id)

    # Для короткого CLI обязательно ждём отправки OpenTelemetry batch.
    client.flush()
    all_imported = imported_before.union(imported_now)
    _write_state(state_path, all_imported)
    # Одна trace + root observation + четыре child observations + шесть scores.
    return BackfillReport(
        imported=len(imported_now),
        skipped=skipped,
        trace_ids=tuple(imported_now),
        estimated_units=len(imported_now) * 12,
    )
