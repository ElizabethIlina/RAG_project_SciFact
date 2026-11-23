"""Единая оценка BM25, dense и hybrid выдач по SciFact qrels."""

from __future__ import annotations

import json
import math
from collections import defaultdict
from pathlib import Path
from statistics import fmean
from typing import Any


def load_qrels(path: Path) -> dict[str, dict[str, int]]:
    """Прочитать qrels TSV в отображение ``query_id -> doc_id -> relevance``."""
    qrels: dict[str, dict[str, int]] = defaultdict(dict)
    with path.open(encoding="utf-8") as source:
        header = source.readline().rstrip("\n").split("\t")
        if header != ["query_id", "doc_id", "score"]:
            raise ValueError(f"Unexpected qrels header: {header!r}")
        for line_number, line in enumerate(source, start=2):
            if not line.strip():
                continue
            fields = line.rstrip("\n").split("\t")
            if len(fields) != 3:
                raise ValueError(f"Malformed qrels row at line {line_number}")
            query_id, doc_id, score = fields
            qrels[query_id][doc_id] = int(score)
    return dict(qrels)


def load_ranking(path: Path) -> dict[str, list[str]]:
    """Прочитать JSONL ranking и вернуть doc_id в порядке возрастания rank."""
    rows: dict[str, list[tuple[int, str]]] = defaultdict(list)
    with path.open(encoding="utf-8") as source:
        for line_number, line in enumerate(source, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            try:
                rows[str(row["query_id"])].append((int(row["rank"]), str(row["doc_id"])))
            except (KeyError, TypeError, ValueError) as exc:
                raise ValueError(f"Malformed ranking row at line {line_number}") from exc

    ranking: dict[str, list[str]] = {}
    for query_id, query_rows in rows.items():
        ordered = sorted(query_rows)
        doc_ids = [doc_id for _rank, doc_id in ordered]
        if len(doc_ids) != len(set(doc_ids)):
            raise ValueError(f"Duplicate doc_id in ranking for query {query_id}")
        ranking[query_id] = doc_ids
    return ranking


def dcg(relevances: list[int]) -> float:
    """Посчитать Discounted Cumulative Gain для списка relevance scores."""
    return sum(
        (2**relevance - 1) / math.log2(rank + 1)
        for rank, relevance in enumerate(relevances, start=1)
    )


def evaluate_ranking(
    ranking: dict[str, list[str]],
    qrels: dict[str, dict[str, int]],
    cutoffs: tuple[int, ...] = (1, 5, 10, 100),
) -> dict[str, Any]:
    """Посчитать macro Recall, Precision, MRR и nDCG на каждом cutoff.

    Усреднение идёт по всем запросам qrels. Если выдача для запроса отсутствует,
    его метрики равны нулю, а не исключаются из среднего.
    """
    if not qrels:
        raise ValueError("qrels must not be empty")
    metrics: dict[str, Any] = {"queries": len(qrels)}
    for cutoff in cutoffs:
        if cutoff <= 0:
            raise ValueError("cutoffs must be greater than zero")
        recalls: list[float] = []
        precisions: list[float] = []
        reciprocal_ranks: list[float] = []
        ndcgs: list[float] = []

        for query_id, relevance_by_doc in qrels.items():
            relevant = {doc_id for doc_id, score in relevance_by_doc.items() if score > 0}
            retrieved = ranking.get(query_id, [])[:cutoff]
            relevant_retrieved = [doc_id for doc_id in retrieved if doc_id in relevant]
            recalls.append(len(relevant_retrieved) / len(relevant) if relevant else 0.0)
            precisions.append(len(relevant_retrieved) / cutoff)

            first_relevant_rank = next(
                (rank for rank, doc_id in enumerate(retrieved, start=1) if doc_id in relevant),
                None,
            )
            reciprocal_ranks.append(
                1.0 / first_relevant_rank if first_relevant_rank is not None else 0.0
            )

            actual = [relevance_by_doc.get(doc_id, 0) for doc_id in retrieved]
            ideal = sorted(relevance_by_doc.values(), reverse=True)[:cutoff]
            ideal_dcg = dcg(ideal)
            ndcgs.append(dcg(actual) / ideal_dcg if ideal_dcg else 0.0)

        metrics[f"recall@{cutoff}"] = round(fmean(recalls), 6)
        metrics[f"precision@{cutoff}"] = round(fmean(precisions), 6)
        metrics[f"mrr@{cutoff}"] = round(fmean(reciprocal_ranks), 6)
        metrics[f"ndcg@{cutoff}"] = round(fmean(ndcgs), 6)
    return metrics


def evaluate_run_file(
    ranking_path: Path,
    qrels_path: Path,
    cutoffs: tuple[int, ...] = (1, 5, 10, 100),
) -> dict[str, Any]:
    """Удобный файловый интерфейс для CLI и ноутбука."""
    return evaluate_ranking(load_ranking(ranking_path), load_qrels(qrels_path), cutoffs)
