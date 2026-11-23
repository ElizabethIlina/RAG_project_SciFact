"""Индексация чанков в Qdrant и агрегация chunk hits обратно в документы."""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from time import perf_counter
from typing import Any
from uuid import NAMESPACE_URL, uuid5

from qdrant_client import models

from rag_playground.retrieval.bm25 import build_content
from rag_playground.retrieval.chunking import ChunkStrategy, read_chunks
from rag_playground.retrieval.embedding import BGEEmbedder, batched
from rag_playground.retrieval.models import (
    ChunkDocumentHit,
    ChunkHit,
    ChunkIndexingReport,
    ChunkQueryResult,
)


def chunk_collection_name(prefix: str, strategy: ChunkStrategy, chunk_size: int) -> str:
    """Собрать явное имя коллекции, фиксирующее стратегию и token budget."""
    return f"{prefix}-{strategy}-s{chunk_size}"


def point_id_for_chunk(chunk_id: str) -> str:
    """Преобразовать стабильный chunk_id в допустимый Qdrant UUID."""
    return str(uuid5(NAMESPACE_URL, f"rag-playground/scifact/chunk/{chunk_id}"))


def iter_existing_chunk_ids(client: Any, collection_name: str):
    """Постранично вернуть chunk_id, уже сохранённые в коллекции."""
    offset = None
    while True:
        points, offset = client.scroll(
            collection_name=collection_name,
            limit=256,
            offset=offset,
            with_payload=["chunk_id"],
            with_vectors=False,
        )
        for point in points:
            if point.payload and point.payload.get("chunk_id") is not None:
                yield str(point.payload["chunk_id"])
        if offset is None:
            break


def index_chunk_dataset(
    client: Any,
    collection_name: str,
    chunks_path: Path,
    embedder: BGEEmbedder,
    embedding_batch_size: int,
    upsert_batch_size: int,
) -> ChunkIndexingReport:
    """Возобновляемо вычислить embeddings отсутствующих чанков."""
    started = perf_counter()
    chunks = read_chunks(chunks_path)
    existing = set(iter_existing_chunk_ids(client, collection_name))
    pending = [chunk for chunk in chunks if chunk.chunk_id not in existing]
    embedded = 0
    upserted = 0

    for chunk_batch in batched(pending, upsert_batch_size):
        contents = [build_content(chunk.title, chunk.text) for chunk in chunk_batch]
        vectors = embedder.embed_documents(contents, embedding_batch_size)
        embedded += len(vectors)
        points = [
            models.PointStruct(
                id=point_id_for_chunk(chunk.chunk_id),
                vector=vector,
                payload={
                    "chunk_id": chunk.chunk_id,
                    "doc_id": chunk.doc_id,
                    "chunk_index": chunk.chunk_index,
                    "strategy": chunk.strategy,
                    "title": chunk.title,
                    "text": chunk.text,
                    "token_count": chunk.token_count,
                    "embedding_model": embedder.model_name,
                },
            )
            for chunk, vector in zip(chunk_batch, vectors, strict=True)
        ]
        client.upsert(collection_name=collection_name, points=points, wait=True)
        upserted += len(points)

    final_count = int(client.count(collection_name, exact=True).count)
    return ChunkIndexingReport(
        collection_name=collection_name,
        chunks_read=len(chunks),
        chunks_skipped=len(existing.intersection(chunk.chunk_id for chunk in chunks)),
        chunks_embedded=embedded,
        chunks_upserted=upserted,
        duration_seconds=round(perf_counter() - started, 6),
        final_point_count=final_count,
    )


def parse_chunk_points(points: list[Any]) -> tuple[ChunkHit, ...]:
    """Преобразовать Qdrant points в стабильные типизированные chunk hits."""
    hits: list[ChunkHit] = []
    for rank, point in enumerate(points, start=1):
        payload = point.payload or {}
        hits.append(
            ChunkHit(
                chunk_id=str(payload.get("chunk_id", "")),
                doc_id=str(payload.get("doc_id", "")),
                chunk_index=int(payload.get("chunk_index", 0)),
                title=str(payload.get("title", "")),
                text=str(payload.get("text", "")),
                chunk_rank=rank,
                score=float(point.score),
            )
        )
    return tuple(hits)


def aggregate_chunks_max_score(
    chunk_hits: tuple[ChunkHit, ...],
    top_k: int,
) -> tuple[ChunkDocumentHit, ...]:
    """Оставить один лучший чанк на doc_id и посчитать число совпавших чанков."""
    by_document: dict[str, list[ChunkHit]] = defaultdict(list)
    for hit in chunk_hits:
        by_document[hit.doc_id].append(hit)

    best_and_count = [
        (min(hits, key=lambda hit: (-hit.score, hit.chunk_rank)), len(hits))
        for hits in by_document.values()
    ]
    best_and_count.sort(key=lambda item: (-item[0].score, item[0].chunk_rank, item[0].doc_id))
    return tuple(
        ChunkDocumentHit(
            doc_id=best.doc_id,
            title=best.title,
            rank=rank,
            score=best.score,
            best_chunk_id=best.chunk_id,
            best_chunk_index=best.chunk_index,
            best_chunk_text=best.text,
            matched_chunks=matched_chunks,
        )
        for rank, (best, matched_chunks) in enumerate(best_and_count[:top_k], start=1)
    )


def search_chunks(
    client: Any,
    collection_name: str,
    strategy: ChunkStrategy,
    embedder: BGEEmbedder,
    query_text: str,
    query_instruction: str,
    top_k: int,
    candidate_multiplier: int,
    query_id: str = "manual",
) -> ChunkQueryResult:
    """Найти чанки и вернуть top-K уникальных документов по max chunk score."""
    if top_k <= 0 or candidate_multiplier <= 0:
        raise ValueError("top_k and candidate_multiplier must be greater than zero")

    embedding_started = perf_counter()
    query_vector = embedder.embed_query(query_text, query_instruction)
    embedding_latency_ms = (perf_counter() - embedding_started) * 1000

    total_points = int(client.count(collection_name, exact=True).count)
    if total_points == 0:
        raise ValueError(f"Chunk collection {collection_name!r} is empty")
    candidate_limit = min(max(top_k * candidate_multiplier, top_k), total_points)
    search_started = perf_counter()
    points: list[Any] = []
    document_hits: tuple[ChunkDocumentHit, ...] = ()
    while True:
        response = client.query_points(
            collection_name=collection_name,
            query=query_vector,
            limit=candidate_limit,
            with_payload=True,
            with_vectors=False,
        )
        points = list(response.points)
        chunk_hits = parse_chunk_points(points)
        document_hits = aggregate_chunks_max_score(chunk_hits, top_k)
        if len(document_hits) >= top_k or candidate_limit >= total_points:
            break
        candidate_limit = min(candidate_limit * 2, total_points)
    search_latency_ms = (perf_counter() - search_started) * 1000

    return ChunkQueryResult(
        query_id=str(query_id),
        query_text=query_text.strip(),
        strategy=strategy,
        top_k=top_k,
        candidate_chunks=len(points),
        embedding_latency_ms=round(embedding_latency_ms, 3),
        search_latency_ms=round(search_latency_ms, 3),
        chunk_hits=chunk_hits,
        document_hits=document_hits,
    )
