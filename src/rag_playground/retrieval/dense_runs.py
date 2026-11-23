"""Пакетный dense retrieval и сохранение его воспроизводимых артефактов."""

from __future__ import annotations

import json
from collections.abc import Iterable
from datetime import UTC, datetime
from pathlib import Path
from statistics import fmean
from typing import Any

from rag_playground.retrieval.dense import search_dense
from rag_playground.retrieval.embedding import BGEEmbedder
from rag_playground.retrieval.models import DenseQueryResult


def run_dense_queries(
    client: Any,
    collection_name: str,
    embedder: BGEEmbedder,
    query_instruction: str,
    queries: Iterable[tuple[str, str]],
    top_k: int,
) -> list[DenseQueryResult]:
    """Последовательно выполнить dense-поиск для заданного списка запросов."""
    return [
        search_dense(
            client,
            collection_name,
            embedder,
            query_text,
            query_instruction,
            top_k,
            query_id,
        )
        for query_id, query_text in queries
    ]


def save_dense_run(
    results: Iterable[DenseQueryResult],
    output_dir: Path,
    run_name: str,
    metadata: dict[str, Any],
) -> dict[str, Path]:
    """Сохранить ranking, latency по запросам и манифест dense-запуска."""
    materialized = list(results)
    output_dir.mkdir(parents=True, exist_ok=True)
    ranking_path = output_dir / f"{run_name}.jsonl"
    queries_path = output_dir / f"{run_name}_queries.jsonl"
    manifest_path = output_dir / f"{run_name}_manifest.json"

    ranking_rows = 0
    with ranking_path.open("w", encoding="utf-8") as target:
        for result in materialized:
            for hit in result.hits:
                target.write(
                    json.dumps(
                        {
                            "doc_id": hit.doc_id,
                            "query_id": result.query_id,
                            "rank": hit.rank,
                            "retrieval_method": "dense",
                            "score": hit.score,
                        },
                        ensure_ascii=False,
                        sort_keys=True,
                    )
                    + "\n"
                )
                ranking_rows += 1

    with queries_path.open("w", encoding="utf-8") as target:
        for result in materialized:
            target.write(
                json.dumps(
                    {
                        "embedding_latency_ms": result.embedding_latency_ms,
                        "hits_returned": len(result.hits),
                        "latency_ms": result.latency_ms,
                        "query_id": result.query_id,
                        "query_text": result.query_text,
                        "search_latency_ms": result.search_latency_ms,
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
        "retrieval_method": "dense",
    }
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return {"ranking": ranking_path, "queries": queries_path, "manifest": manifest_path}
