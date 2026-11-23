"""Reciprocal Rank Fusion для объединения BM25 и dense retrieval."""

from __future__ import annotations

import json
from collections import defaultdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from rag_playground.retrieval.evaluation import load_ranking


def reciprocal_rank_fusion(
    rankings: list[list[str]],
    rrf_k: int = 60,
    top_k: int = 100,
) -> list[tuple[str, float]]:
    """Объединить несколько ранжированных списков без сравнения сырых scores."""
    if rrf_k <= 0 or top_k <= 0:
        raise ValueError("rrf_k and top_k must be greater than zero")
    scores: dict[str, float] = defaultdict(float)
    for ranking in rankings:
        # set защищает один список от повторного начисления баллов одному документу.
        seen: set[str] = set()
        for rank, doc_id in enumerate(ranking, start=1):
            if doc_id in seen:
                continue
            seen.add(doc_id)
            scores[doc_id] += 1.0 / (rrf_k + rank)

    # doc_id — детерминированный tie-breaker для полностью воспроизводимого JSONL.
    ordered = sorted(scores.items(), key=lambda item: (-item[1], item[0]))
    return ordered[:top_k]


def fuse_run_files(
    bm25_path: Path,
    dense_path: Path,
    output_path: Path,
    rrf_k: int,
    top_k: int,
) -> dict[str, Any]:
    """Объединить два ranking-файла и сохранить hybrid ranking JSONL."""
    bm25 = load_ranking(bm25_path)
    dense = load_ranking(dense_path)
    query_ids = list(dict.fromkeys([*bm25.keys(), *dense.keys()]))
    output_path.parent.mkdir(parents=True, exist_ok=True)

    rows = 0
    with output_path.open("w", encoding="utf-8") as target:
        for query_id in query_ids:
            fused = reciprocal_rank_fusion(
                [bm25.get(query_id, []), dense.get(query_id, [])],
                rrf_k=rrf_k,
                top_k=top_k,
            )
            for rank, (doc_id, score) in enumerate(fused, start=1):
                target.write(
                    json.dumps(
                        {
                            "doc_id": doc_id,
                            "query_id": query_id,
                            "rank": rank,
                            "retrieval_method": "hybrid_rrf",
                            "score": score,
                        },
                        ensure_ascii=False,
                        sort_keys=True,
                    )
                    + "\n"
                )
                rows += 1
    return {
        "created_at_utc": datetime.now(UTC).isoformat(),
        "queries": len(query_ids),
        "ranking_rows": rows,
        "retrieval_method": "hybrid_rrf",
        "rrf_k": rrf_k,
        "top_k": top_k,
    }
