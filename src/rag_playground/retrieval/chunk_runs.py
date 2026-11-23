"""Пакетный chunk retrieval, evidence и воспроизводимые ranking-файлы."""

from __future__ import annotations

import json
from collections.abc import Iterable
from datetime import UTC, datetime
from pathlib import Path
from statistics import fmean
from typing import Any

from rag_playground.retrieval.chunk_retrieval import search_chunks
from rag_playground.retrieval.chunking import ChunkStrategy
from rag_playground.retrieval.embedding import BGEEmbedder
from rag_playground.retrieval.models import ChunkQueryResult


def run_chunk_queries(
    client: Any,
    collection_name: str,
    strategy: ChunkStrategy,
    embedder: BGEEmbedder,
    query_instruction: str,
    queries: Iterable[tuple[str, str]],
    top_k: int,
    candidate_multiplier: int,
) -> list[ChunkQueryResult]:
    """Выполнить chunk retrieval для запросов в фиксированном порядке."""
    return [
        search_chunks(
            client,
            collection_name,
            strategy,
            embedder,
            query_text,
            query_instruction,
            top_k,
            candidate_multiplier,
            query_id,
        )
        for query_id, query_text in queries
    ]


def save_chunk_run(
    results: Iterable[ChunkQueryResult],
    output_dir: Path,
    run_name: str,
    metadata: dict[str, Any],
) -> dict[str, Path]:
    """Сохранить документный ranking, chunk evidence, query summary и manifest."""
    materialized = list(results)
    output_dir.mkdir(parents=True, exist_ok=True)
    ranking_path = output_dir / f"{run_name}.jsonl"
    evidence_path = output_dir / f"{run_name}_chunks.jsonl"
    queries_path = output_dir / f"{run_name}_queries.jsonl"
    manifest_path = output_dir / f"{run_name}_manifest.json"

    ranking_rows = 0
    with ranking_path.open("w", encoding="utf-8") as target:
        for result in materialized:
            for hit in result.document_hits:
                target.write(
                    json.dumps(
                        {
                            "best_chunk_id": hit.best_chunk_id,
                            "doc_id": hit.doc_id,
                            "matched_chunks": hit.matched_chunks,
                            "query_id": result.query_id,
                            "rank": hit.rank,
                            "retrieval_method": f"chunk_{result.strategy}_max",
                            "score": hit.score,
                        },
                        ensure_ascii=False,
                        sort_keys=True,
                    )
                    + "\n"
                )
                ranking_rows += 1

    evidence_rows = 0
    with evidence_path.open("w", encoding="utf-8") as target:
        for result in materialized:
            for hit in result.chunk_hits:
                target.write(
                    json.dumps(
                        {
                            "chunk_id": hit.chunk_id,
                            "chunk_index": hit.chunk_index,
                            "chunk_rank": hit.chunk_rank,
                            "doc_id": hit.doc_id,
                            "query_id": result.query_id,
                            "score": hit.score,
                        },
                        ensure_ascii=False,
                        sort_keys=True,
                    )
                    + "\n"
                )
                evidence_rows += 1

    with queries_path.open("w", encoding="utf-8") as target:
        for result in materialized:
            target.write(
                json.dumps(
                    {
                        "candidate_chunks": result.candidate_chunks,
                        "documents_returned": len(result.document_hits),
                        "embedding_latency_ms": result.embedding_latency_ms,
                        "latency_ms": result.latency_ms,
                        "query_id": result.query_id,
                        "query_text": result.query_text,
                        "search_latency_ms": result.search_latency_ms,
                        "strategy": result.strategy,
                        "top_k": result.top_k,
                    },
                    ensure_ascii=False,
                    sort_keys=True,
                )
                + "\n"
            )

    manifest = {
        **metadata,
        "created_at_utc": datetime.now(UTC).isoformat(),
        "evidence_rows": evidence_rows,
        "mean_candidate_chunks": round(
            fmean(result.candidate_chunks for result in materialized), 3
        )
        if materialized
        else 0.0,
        "mean_embedding_latency_ms": round(
            fmean(result.embedding_latency_ms for result in materialized), 3
        )
        if materialized
        else 0.0,
        "mean_search_latency_ms": round(
            fmean(result.search_latency_ms for result in materialized), 3
        )
        if materialized
        else 0.0,
        "mean_total_latency_ms": round(
            fmean(result.latency_ms for result in materialized), 3
        )
        if materialized
        else 0.0,
        "queries": len(materialized),
        "ranking_rows": ranking_rows,
    }
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return {
        "ranking": ranking_path,
        "chunks": evidence_path,
        "queries": queries_path,
        "manifest": manifest_path,
    }
