"""Быстрые тесты dense retrieval без загрузки настоящей BGE-модели."""

from __future__ import annotations

import json
from pathlib import Path

from qdrant_client import QdrantClient

from rag_playground.retrieval.dense import (
    create_dense_collection,
    index_dense_corpus,
    point_id_for_doc_id,
    search_dense,
)


class FakeEmbedder:
    """Детерминированная маленькая модель для проверки Qdrant-потока."""

    model_name = "fake-bge"
    max_length = 5
    dimension = 2

    def count_tokens(self, text: str) -> int:
        return len(text.split())

    def embed_documents(self, texts, batch_size: int):
        del batch_size
        return [self._vector(text) for text in texts]

    def embed_query(self, text: str, instruction: str):
        del instruction
        return self._vector(text)

    @staticmethod
    def _vector(text: str) -> list[float]:
        # Тексты про vitamin направляются по первой оси, остальные — по второй.
        return [1.0, 0.0] if "vitamin" in text.lower() else [0.0, 1.0]


def write_corpus(path: Path) -> None:
    """Создать минимальный корпус в реальном формате проекта."""
    rows = [
        {"doc_id": "1", "title": "Vitamin D", "text": "Bone health and vitamin D"},
        {"doc_id": "2", "title": "Ocean", "text": "Water currents"},
    ]
    path.write_text(
        "".join(json.dumps(row) + "\n" for row in rows),
        encoding="utf-8",
    )


def test_point_id_is_stable_uuid():
    """Один doc_id всегда должен получать один и тот же допустимый point ID."""
    assert point_id_for_doc_id("42") == point_id_for_doc_id("42")
    assert point_id_for_doc_id("42") != point_id_for_doc_id("43")


def test_dense_index_can_resume_without_reembedding(tmp_path: Path):
    """Повторный запуск должен пропустить уже записанные документы."""
    client = QdrantClient(":memory:")
    embedder = FakeEmbedder()
    corpus_path = tmp_path / "corpus.jsonl"
    write_corpus(corpus_path)
    assert create_dense_collection(client, "docs", embedder.dimension) is True

    first = index_dense_corpus(client, "docs", corpus_path, embedder, 2, 2)
    second = index_dense_corpus(client, "docs", corpus_path, embedder, 2, 2)

    assert first.documents_upserted == 2
    assert first.final_point_count == 2
    assert first.truncated_documents == 1
    assert second.documents_skipped == 2
    assert second.documents_embedded == 0
    assert second.final_point_count == 2


def test_dense_search_returns_semantically_nearest_document(tmp_path: Path):
    """Qdrant должен поставить документ с совпадающим fake-вектором первым."""
    client = QdrantClient(":memory:")
    embedder = FakeEmbedder()
    corpus_path = tmp_path / "corpus.jsonl"
    write_corpus(corpus_path)
    create_dense_collection(client, "docs", embedder.dimension)
    index_dense_corpus(client, "docs", corpus_path, embedder, 2, 2)

    result = search_dense(client, "docs", embedder, "vitamin deficiency", "instruction", 2)
    assert result.hits[0].doc_id == "1"
    assert result.hits[0].rank == 1
    assert result.latency_ms >= 0
