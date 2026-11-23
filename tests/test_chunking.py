"""Unit-тесты chunking и документной агрегации без настоящей BGE-модели."""

from __future__ import annotations

import json
import re
from dataclasses import asdict
from pathlib import Path

from qdrant_client import QdrantClient

from rag_playground.retrieval.chunk_retrieval import (
    aggregate_chunks_max_score,
    chunk_collection_name,
    index_chunk_dataset,
    search_chunks,
)
from rag_playground.retrieval.chunking import (
    CHUNK_STRATEGIES,
    Chunk,
    chunk_document,
    prepare_chunk_dataset,
)
from rag_playground.retrieval.dense import create_dense_collection
from rag_playground.retrieval.models import ChunkHit


class WordEmbedder:
    """Простой обратимый word-tokenizer и двумерные тестовые embeddings."""

    model_name = "word-test-model"
    dimension = 2
    max_length = 512

    def __init__(self) -> None:
        self._token_to_id: dict[str, int] = {}
        self._id_to_token: dict[int, str] = {}

    @staticmethod
    def _words(text: str) -> list[str]:
        return re.findall(r"[A-Za-z0-9]+|[^\w\s]", text)

    def encode_text(self, text: str) -> list[int]:
        ids = []
        for token in self._words(text):
            if token not in self._token_to_id:
                token_id = len(self._token_to_id) + 1
                self._token_to_id[token] = token_id
                self._id_to_token[token_id] = token
            ids.append(self._token_to_id[token])
        return ids

    def decode_tokens(self, token_ids) -> str:
        return " ".join(self._id_to_token[token_id] for token_id in token_ids)

    def count_tokens(self, text: str) -> int:
        # Два дополнительных токена имитируют CLS и SEP настоящей BGE.
        return len(self.encode_text(text)) + 2

    def embed_documents(self, texts, batch_size: int):
        del batch_size
        return [self._vector(text) for text in texts]

    def embed_query(self, text: str, instruction: str):
        del instruction
        return self._vector(text)

    @staticmethod
    def _vector(text: str) -> list[float]:
        return [1.0, 0.0] if "vitamin" in text.lower() else [0.0, 1.0]


def sample_document() -> dict[str, str]:
    """Вернуть текст с несколькими предложениями и достаточной длиной."""
    return {
        "doc_id": "42",
        "title": "Scientific title",
        "text": (
            "First sentence explains vitamin effects. "
            "Second sentence contains several additional scientific words. "
            "Third sentence reports the final experimental conclusion."
        ),
    }


def test_all_strategies_produce_stable_nonempty_chunks_within_limit():
    """Ни одна стратегия не должна создавать пустой или слишком длинный чанк."""
    embedder = WordEmbedder()
    for strategy in CHUNK_STRATEGIES:
        first = chunk_document(sample_document(), strategy, embedder, 16, 3, 1)
        second = chunk_document(sample_document(), strategy, embedder, 16, 3, 1)

        assert first
        assert [chunk.chunk_id for chunk in first] == [chunk.chunk_id for chunk in second]
        assert all(chunk.text.strip() for chunk in first)
        assert all(chunk.token_count <= 16 for chunk in first)
        assert [chunk.chunk_index for chunk in first] == list(range(len(first)))


def test_fixed_chunks_have_token_overlap():
    """Последние токены окна должны повторяться в начале следующего окна."""
    embedder = WordEmbedder()
    chunks = chunk_document(sample_document(), "fixed", embedder, 14, 3, 1)
    assert len(chunks) > 1
    first_ids = embedder.encode_text(chunks[0].text)
    second_ids = embedder.encode_text(chunks[1].text)
    assert first_ids[-3:] == second_ids[:3]


def test_sentence_chunks_repeat_one_sentence():
    """Sentence overlap должен переносить последнее предложение прошлого чанка."""
    embedder = WordEmbedder()
    chunks = chunk_document(sample_document(), "sentence", embedder, 23, 3, 1)
    assert len(chunks) > 1
    assert "Second sentence" in chunks[0].text
    assert "Second sentence" in chunks[1].text


def test_prepare_dataset_writes_jsonl_and_manifest(tmp_path: Path):
    """Подготовка должна сохранить все документы и нулевое число нарушений лимита."""
    corpus = tmp_path / "corpus.jsonl"
    documents = [
        sample_document(),
        {"doc_id": "43", "title": "Ocean", "text": "Ocean currents move water."},
    ]
    corpus.write_text(
        "".join(json.dumps(document) + "\n" for document in documents),
        encoding="utf-8",
    )
    chunks_path, manifest_path, report = prepare_chunk_dataset(
        corpus, tmp_path / "chunks", "recursive", WordEmbedder(), 16, 3, 1
    )

    assert chunks_path.exists()
    assert manifest_path.exists()
    assert report.source_documents == 2
    assert report.empty_chunks == 0
    assert report.over_limit_chunks == 0


def test_max_score_aggregation_returns_each_document_once():
    """Несколько чанков одного doc_id должны превратиться в одну строку ranking."""
    chunk_hits = (
        ChunkHit("a0", "a", 0, "A", "first", 1, 0.8),
        ChunkHit("a1", "a", 1, "A", "best", 2, 0.9),
        ChunkHit("b0", "b", 0, "B", "other", 3, 0.7),
    )
    documents = aggregate_chunks_max_score(chunk_hits, top_k=10)

    assert [hit.doc_id for hit in documents] == ["a", "b"]
    assert documents[0].best_chunk_id == "a1"
    assert documents[0].matched_chunks == 2


def test_chunk_index_and_search_work_with_in_memory_qdrant(tmp_path: Path):
    """Подготовленный JSONL должен пройти полный index-search-aggregate поток."""
    embedder = WordEmbedder()
    chunks = [
        Chunk("1:fixed:0000", "1", 0, "fixed", "Vitamin", "healthy bones", 5),
        Chunk("2:fixed:0000", "2", 0, "fixed", "Ocean", "warm current", 5),
    ]
    chunks_path = tmp_path / "fixed.jsonl"
    chunks_path.write_text(
        "".join(json.dumps(asdict(chunk)) + "\n" for chunk in chunks),
        encoding="utf-8",
    )
    client = QdrantClient(":memory:")
    collection = chunk_collection_name("test", "fixed", 16)
    create_dense_collection(client, collection, embedder.dimension)
    report = index_chunk_dataset(client, collection, chunks_path, embedder, 2, 2)
    result = search_chunks(
        client, collection, "fixed", embedder, "vitamin bones", "instruction", 2, 2
    )

    assert report.final_point_count == 2
    assert result.document_hits[0].doc_id == "1"
    assert len({hit.doc_id for hit in result.document_hits}) == 2
