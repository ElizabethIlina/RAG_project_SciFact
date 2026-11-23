"""Пакетный BM25-запуск и сохранение воспроизводимых артефактов.

Отдельный файл ranking содержит одну строку на hit. Query summary сохраняет также
запросы с нулевым числом hits, а манифест фиксирует версию Elasticsearch и параметры.
"""

from __future__ import annotations

import json
from collections.abc import Iterable
from datetime import UTC, datetime
from pathlib import Path
from statistics import fmean
from typing import Any

from rag_playground.retrieval.bm25 import search_bm25
from rag_playground.retrieval.models import BM25QueryResult


def read_test_queries(queries_path: Path, qrels_path: Path) -> list[tuple[str, str]]:
    """Выбрать из общего файла только запросы, присутствующие в qrels split.

    Порядок первого появления ``query_id`` в qrels сохраняется. Один запрос может
    иметь несколько релевантных документов, но в batch search он должен попасть один раз.
    """
    query_text_by_id: dict[str, str] = {}
    with queries_path.open(encoding="utf-8") as source:
        for line in source:
            if line.strip():
                row = json.loads(line)
                query_text_by_id[str(row["query_id"])] = str(row["text"])

    ordered_query_ids: list[str] = []
    seen: set[str] = set()
    with qrels_path.open(encoding="utf-8") as source:
        header = source.readline().rstrip("\n").split("\t")
        if header != ["query_id", "doc_id", "score"]:
            raise ValueError(f"Unexpected qrels header: {header!r}")
        for line_number, line in enumerate(source, start=2):
            if not line.strip():
                continue
            fields = line.rstrip("\n").split("\t")
            if len(fields) != 3:
                raise ValueError(f"Malformed qrels row at line {line_number}")
            query_id = fields[0]
            if query_id not in seen:
                ordered_query_ids.append(query_id)
                seen.add(query_id)

    missing = [query_id for query_id in ordered_query_ids if query_id not in query_text_by_id]
    if missing:
        raise ValueError(f"Queries referenced by qrels are missing: {missing[:5]}")
    return [(query_id, query_text_by_id[query_id]) for query_id in ordered_query_ids]


def run_bm25_queries(
    client: Any,
    index_name: str,
    queries: Iterable[tuple[str, str]],
    top_k: int,
) -> list[BM25QueryResult]:
    """Выполнить BM25 для каждого ``(query_id, text)`` в заданном порядке."""
    return [
        search_bm25(client, index_name, query_text, top_k, query_id)
        for query_id, query_text in queries
    ]


def save_bm25_run(
    results: Iterable[BM25QueryResult],
    output_dir: Path,
    run_name: str,
    metadata: dict[str, Any],
) -> dict[str, Path]:
    """Сохранить ranking, query summary и манифест одного запуска.

    ``sort_keys=True`` и фиксированный порядок запросов/hits делают файлы удобными для
    сравнения. Время создания остаётся в манифесте и не влияет на ranking.
    """
    materialized = list(results)
    output_dir.mkdir(parents=True, exist_ok=True)
    ranking_path = output_dir / f"{run_name}.jsonl"
    queries_path = output_dir / f"{run_name}_queries.jsonl"
    manifest_path = output_dir / f"{run_name}_manifest.json"

    hit_count = 0
    with ranking_path.open("w", encoding="utf-8") as ranking_file:
        for result in materialized:
            for hit in result.hits:
                row = {
                    "doc_id": hit.doc_id,
                    "query_id": result.query_id,
                    "rank": hit.rank,
                    "retrieval_method": "bm25",
                    "score": hit.score,
                }
                ranking_file.write(
                    json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n"
                )
                hit_count += 1

    with queries_path.open("w", encoding="utf-8") as queries_file:
        for result in materialized:
            row = {
                "hits_returned": len(result.hits),
                "latency_ms": result.latency_ms,
                "query_id": result.query_id,
                "query_text": result.query_text,
                "top_k": result.top_k,
            }
            queries_file.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")

    latencies = [result.latency_ms for result in materialized]
    manifest = {
        **metadata,
        "created_at_utc": datetime.now(UTC).isoformat(),
        "mean_query_latency_ms": round(fmean(latencies), 3) if latencies else 0.0,
        "queries": len(materialized),
        "ranking_rows": hit_count,
        "retrieval_method": "bm25",
    }
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return {
        "ranking": ranking_path,
        "queries": queries_path,
        "manifest": manifest_path,
    }

