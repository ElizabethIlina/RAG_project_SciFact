"""Документный dense retrieval через локальные BGE Small и Qdrant."""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import asdict
from pathlib import Path
from time import perf_counter
from typing import Any
from uuid import NAMESPACE_URL, uuid5

from qdrant_client import QdrantClient, models

from rag_playground.retrieval.bm25 import build_content, read_corpus
from rag_playground.retrieval.embedding import BGEEmbedder, batched
from rag_playground.retrieval.models import DenseHit, DenseIndexingReport, DenseQueryResult


def create_qdrant_client(url: str, timeout: int) -> QdrantClient:
    """Создать HTTP-клиент локального Qdrant."""
    return QdrantClient(url=url, timeout=timeout, prefer_grpc=False)


def point_id_for_doc_id(doc_id: str) -> str:
    """Получить стабильный UUID, допустимый как Qdrant point ID.

    Исходный SciFact ``doc_id`` всё равно сохраняется в payload. UUID нужен потому,
    что Qdrant принимает только целые числа или UUID, а проект хранит ID как строки.
    """
    return str(uuid5(NAMESPACE_URL, f"rag-playground/scifact/document/{doc_id}"))


def create_dense_collection(
    client: Any,
    collection_name: str,
    vector_size: int,
    recreate: bool = False,
) -> bool:
    """Создать cosine-коллекцию, не удаляя её без явного ``recreate``."""
    exists = bool(client.collection_exists(collection_name))
    if exists and not recreate:
        info = client.get_collection(collection_name)
        existing_size = int(info.config.params.vectors.size)
        if existing_size != vector_size:
            raise ValueError(
                f"Collection vector size is {existing_size}, expected {vector_size}; "
                "use a new collection name or --recreate"
            )
        return False
    if exists:
        # Удаляется только точное имя коллекции, без wildcard.
        client.delete_collection(collection_name)
    client.create_collection(
        collection_name=collection_name,
        vectors_config=models.VectorParams(
            size=vector_size,
            distance=models.Distance.COSINE,
        ),
    )
    return True


def iter_existing_doc_ids(client: Any, collection_name: str) -> Iterator[str]:
    """Постранично прочитать уже загруженные doc_id для возобновления индексации."""
    offset = None
    while True:
        points, offset = client.scroll(
            collection_name=collection_name,
            limit=256,
            offset=offset,
            with_payload=["doc_id"],
            with_vectors=False,
        )
        for point in points:
            if point.payload and point.payload.get("doc_id") is not None:
                yield str(point.payload["doc_id"])
        if offset is None:
            break


def index_dense_corpus(
    client: Any,
    collection_name: str,
    corpus_path: Path,
    embedder: BGEEmbedder,
    embedding_batch_size: int,
    upsert_batch_size: int,
) -> DenseIndexingReport:
    """Создать embeddings отсутствующих документов и пакетно записать их в Qdrant.

    Перед началом читаются существующие ``doc_id``. Поэтому после остановки команда
    продолжает работу с первого отсутствующего документа, не вычисляя готовые векторы снова.
    """
    started = perf_counter()
    documents = list(read_corpus(corpus_path))
    existing = set(iter_existing_doc_ids(client, collection_name))
    pending = [document for document in documents if document["doc_id"] not in existing]
    truncated_documents = 0
    documents_embedded = 0
    documents_upserted = 0

    # Upsert batch может быть больше embedding batch. Внешний цикл ограничивает
    # объём payload одного HTTP-запроса, внутренний параметр — память ONNX inference.
    for document_batch in batched(pending, upsert_batch_size):
        contents = [
            build_content(document["title"], document["text"])
            for document in document_batch
        ]
        token_counts = [embedder.count_tokens(content) for content in contents]
        truncated_documents += sum(
            token_count > embedder.max_length for token_count in token_counts
        )
        vectors = embedder.embed_documents(contents, embedding_batch_size)
        documents_embedded += len(vectors)

        points = []
        for document, vector, token_count in zip(
            document_batch, vectors, token_counts, strict=True
        ):
            payload = {
                **document,
                "embedding_model": embedder.model_name,
                "original_token_count": token_count,
                "truncated": token_count > embedder.max_length,
            }
            points.append(
                models.PointStruct(
                    id=point_id_for_doc_id(document["doc_id"]),
                    vector=vector,
                    payload=payload,
                )
            )
        client.upsert(collection_name=collection_name, points=points, wait=True)
        documents_upserted += len(points)

    final_count = int(client.count(collection_name=collection_name, exact=True).count)
    return DenseIndexingReport(
        collection_name=collection_name,
        documents_read=len(documents),
        documents_skipped=len(existing.intersection(document["doc_id"] for document in documents)),
        documents_embedded=documents_embedded,
        documents_upserted=documents_upserted,
        truncated_documents=truncated_documents,
        duration_seconds=round(perf_counter() - started, 6),
        final_point_count=final_count,
    )


def search_dense(
    client: Any,
    collection_name: str,
    embedder: BGEEmbedder,
    query_text: str,
    query_instruction: str,
    top_k: int,
    query_id: str = "manual",
) -> DenseQueryResult:
    """Векторизовать запрос, найти ближайшие документы и раздельно замерить время."""
    if top_k <= 0:
        raise ValueError("top_k must be greater than zero")

    embedding_started = perf_counter()
    query_vector = embedder.embed_query(query_text, query_instruction)
    embedding_latency_ms = (perf_counter() - embedding_started) * 1000

    search_started = perf_counter()
    response = client.query_points(
        collection_name=collection_name,
        query=query_vector,
        limit=top_k,
        with_payload=True,
        with_vectors=False,
    )
    search_latency_ms = (perf_counter() - search_started) * 1000

    hits: list[DenseHit] = []
    for rank, point in enumerate(response.points, start=1):
        payload = point.payload or {}
        hits.append(
            DenseHit(
                doc_id=str(payload.get("doc_id", "")),
                title=str(payload.get("title", "")),
                text=str(payload.get("text", "")),
                rank=rank,
                score=float(point.score),
            )
        )
    return DenseQueryResult(
        query_id=str(query_id),
        query_text=query_text.strip(),
        top_k=top_k,
        embedding_latency_ms=round(embedding_latency_ms, 3),
        search_latency_ms=round(search_latency_ms, 3),
        hits=tuple(hits),
    )


def get_dense_collection_info(client: Any, collection_name: str) -> dict[str, Any]:
    """Вернуть компактную и JSON-совместимую сводку коллекции."""
    exists = bool(client.collection_exists(collection_name))
    summary: dict[str, Any] = {
        "collection_name": collection_name,
        "collection_exists": exists,
    }
    if exists:
        info = client.get_collection(collection_name)
        summary.update(
            {
                "point_count": int(client.count(collection_name, exact=True).count),
                "vector_size": int(info.config.params.vectors.size),
                "distance": str(info.config.params.vectors.distance),
                "status": str(info.status),
            }
        )
    return summary


def report_as_dict(report: DenseIndexingReport) -> dict[str, Any]:
    """Сохранить единое место преобразования отчёта для CLI и ноутбука."""
    return asdict(report)
