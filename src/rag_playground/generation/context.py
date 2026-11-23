"""Подготовка ограниченного и проверяемого контекста из chunk retrieval."""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

from rag_playground.generation.models import (
    ExperimentQuery,
    PreparedRAGExample,
    RAGSource,
)
from rag_playground.generation.prompting import build_rag_prompt
from rag_playground.retrieval.models import ChunkQueryResult


def load_experiment_queries(path: Path) -> list[ExperimentQuery]:
    """Прочитать неизменяемый список примеров и проверить уникальность ID."""
    payload = json.loads(path.read_text(encoding="utf-8"))
    rows = payload.get("examples")
    if not isinstance(rows, list) or not rows:
        raise ValueError("Experiment plan must contain a non-empty examples list")
    queries = [
        ExperimentQuery(
            query_id=str(row["query_id"]),
            query_text=str(row["query_text"]).strip(),
            relevant_doc_ids=tuple(str(value) for value in row["relevant_doc_ids"]),
        )
        for row in rows
    ]
    if len({query.query_id for query in queries}) != len(queries):
        raise ValueError("Experiment query IDs must be unique")
    if not all(query.query_text and query.relevant_doc_ids for query in queries):
        raise ValueError("Every experiment query needs text and relevant documents")
    return queries


def select_experiment_queries(
    queries: list[ExperimentQuery],
    query_ids: list[str] | None,
) -> list[ExperimentQuery]:
    """Выбрать ID в пользовательском порядке и явно отклонить неизвестные."""
    if not query_ids:
        return queries
    by_id = {query.query_id: query for query in queries}
    missing = [query_id for query_id in query_ids if query_id not in by_id]
    if missing:
        raise ValueError(f"Unknown experiment query IDs: {missing}")
    if len(query_ids) != len(set(query_ids)):
        raise ValueError("Query IDs must not be repeated")
    return [by_id[query_id] for query_id in query_ids]


def build_prepared_example(
    experiment_query: ExperimentQuery,
    retrieval: ChunkQueryResult,
    max_documents: int,
    max_context_chars: int,
) -> PreparedRAGExample:
    """Выбрать лучшие неповторяющиеся фрагменты и собрать точный prompt."""
    if retrieval.query_id != experiment_query.query_id:
        raise ValueError("Retrieval query_id does not match experiment query_id")
    if max_documents <= 0 or max_context_chars <= 0:
        raise ValueError("Context limits must be greater than zero")

    sources: list[RAGSource] = []
    seen_texts: set[str] = set()
    used_chars = 0
    for hit in retrieval.document_hits:
        normalized = " ".join(hit.best_chunk_text.lower().split())
        if not normalized or normalized in seen_texts:
            continue
        candidate_chars = len(hit.title) + len(hit.best_chunk_text)
        if sources and used_chars + candidate_chars > max_context_chars:
            break
        if not sources and candidate_chars > max_context_chars:
            raise ValueError("The first source does not fit RAG_CONTEXT_MAX_CHARS")
        sources.append(
            RAGSource(
                source_number=len(sources) + 1,
                doc_id=hit.doc_id,
                chunk_id=hit.best_chunk_id,
                title=hit.title,
                text=hit.best_chunk_text,
                retrieval_rank=hit.rank,
                retrieval_score=hit.score,
            )
        )
        seen_texts.add(normalized)
        used_chars += candidate_chars
        if len(sources) >= max_documents:
            break
    if not sources:
        raise ValueError(f"No usable retrieval sources for query {experiment_query.query_id}")

    relevant = set(experiment_query.relevant_doc_ids)
    retrieved_relevant = tuple(source.doc_id for source in sources if source.doc_id in relevant)
    source_tuple = tuple(sources)
    return PreparedRAGExample(
        query_id=experiment_query.query_id,
        query_text=experiment_query.query_text,
        relevant_doc_ids=experiment_query.relevant_doc_ids,
        retrieval_strategy=retrieval.strategy,
        retrieval_latency_ms=retrieval.latency_ms,
        sources=source_tuple,
        retrieved_relevant_doc_ids=retrieved_relevant,
        prompt=build_rag_prompt(experiment_query.query_text, source_tuple),
    )


def save_prepared_examples(examples: list[PreparedRAGExample], path: Path) -> None:
    """Сохранить exact prompts до генерации для проверки и воспроизводимости."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as target:
        for example in examples:
            target.write(
                json.dumps(asdict(example), ensure_ascii=False, sort_keys=True) + "\n"
            )


def load_prepared_examples(path: Path) -> list[PreparedRAGExample]:
    """Восстановить подготовленные примеры из JSONL без повторного retrieval."""
    examples: list[PreparedRAGExample] = []
    with path.open(encoding="utf-8") as source:
        for line_number, line in enumerate(source, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            try:
                sources = tuple(RAGSource(**item) for item in row["sources"])
                examples.append(
                    PreparedRAGExample(
                        query_id=str(row["query_id"]),
                        query_text=str(row["query_text"]),
                        relevant_doc_ids=tuple(row["relevant_doc_ids"]),
                        retrieval_strategy=str(row["retrieval_strategy"]),
                        retrieval_latency_ms=float(row["retrieval_latency_ms"]),
                        sources=sources,
                        retrieved_relevant_doc_ids=tuple(
                            row["retrieved_relevant_doc_ids"]
                        ),
                        prompt=str(row["prompt"]),
                    )
                )
            except (KeyError, TypeError, ValueError) as exc:
                raise ValueError(
                    f"Malformed prepared example at line {line_number}"
                ) from exc
    return examples
