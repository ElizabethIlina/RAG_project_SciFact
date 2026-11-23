"""Типизированные модели BM25-индексации и поиска.

Отдельные dataclass-модели не дают формату JSON-ответа Elasticsearch распространяться
по всему проекту. В следующих этапах эти модели можно будет сравнивать с
результатами Qdrant, не меняя остальную логику.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class BM25Hit:
    """Один документ в ранжированном ответе BM25.

    ``rank`` всегда начинается с 1. ``score`` сравним только внутри одного
    индекса и запроса: для сравнения методов позже будут использоваться ranks и
    метрики, а не сырые score разных движков.
    """

    doc_id: str
    title: str
    text: str
    rank: int
    score: float


@dataclass(frozen=True, slots=True)
class BM25QueryResult:
    """Полный результат одного запроса с задержкой и списком hits."""

    query_id: str
    query_text: str
    top_k: int
    latency_ms: float
    hits: tuple[BM25Hit, ...]


@dataclass(frozen=True, slots=True)
class IndexingReport:
    """Сводка пакетной индексации корпуса."""

    index_name: str
    documents_read: int
    documents_indexed: int
    documents_failed: int
    duration_seconds: float
    final_document_count: int


@dataclass(frozen=True, slots=True)
class DenseHit:
    """Один документ в выдаче семантического поиска Qdrant."""

    doc_id: str
    title: str
    text: str
    rank: int
    score: float


@dataclass(frozen=True, slots=True)
class DenseQueryResult:
    """Результат dense-поиска с раздельным временем embedding и Qdrant."""

    query_id: str
    query_text: str
    top_k: int
    embedding_latency_ms: float
    search_latency_ms: float
    hits: tuple[DenseHit, ...]

    @property
    def latency_ms(self) -> float:
        """Вернуть полную клиентскую задержку одного запроса."""
        return round(self.embedding_latency_ms + self.search_latency_ms, 3)


@dataclass(frozen=True, slots=True)
class DenseIndexingReport:
    """Сводка возобновляемой индексации документов в Qdrant."""

    collection_name: str
    documents_read: int
    documents_skipped: int
    documents_embedded: int
    documents_upserted: int
    truncated_documents: int
    duration_seconds: float
    final_point_count: int


@dataclass(frozen=True, slots=True)
class ChunkHit:
    """Один чанк в сырой выдаче Qdrant до документной агрегации."""

    chunk_id: str
    doc_id: str
    chunk_index: int
    title: str
    text: str
    chunk_rank: int
    score: float


@dataclass(frozen=True, slots=True)
class ChunkDocumentHit:
    """Уникальный документ после агрегации его найденных чанков."""

    doc_id: str
    title: str
    rank: int
    score: float
    best_chunk_id: str
    best_chunk_index: int
    best_chunk_text: str
    matched_chunks: int


@dataclass(frozen=True, slots=True)
class ChunkQueryResult:
    """Результат chunk retrieval с сырыми чанками и документным рейтингом."""

    query_id: str
    query_text: str
    strategy: str
    top_k: int
    candidate_chunks: int
    embedding_latency_ms: float
    search_latency_ms: float
    chunk_hits: tuple[ChunkHit, ...]
    document_hits: tuple[ChunkDocumentHit, ...]

    @property
    def latency_ms(self) -> float:
        """Вернуть полную задержку embedding + Qdrant."""
        return round(self.embedding_latency_ms + self.search_latency_ms, 3)


@dataclass(frozen=True, slots=True)
class ChunkIndexingReport:
    """Сводка возобновляемой индексации одного набора чанков."""

    collection_name: str
    chunks_read: int
    chunks_skipped: int
    chunks_embedded: int
    chunks_upserted: int
    duration_seconds: float
    final_point_count: int

