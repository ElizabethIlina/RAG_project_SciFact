"""Интеграционный тест chunk retrieval на реальном локальном Qdrant."""

from __future__ import annotations

import json
import os
import uuid
from dataclasses import asdict

import pytest

from rag_playground.config import Settings
from rag_playground.retrieval.chunk_retrieval import (
    index_chunk_dataset,
    search_chunks,
)
from rag_playground.retrieval.chunking import Chunk
from rag_playground.retrieval.dense import create_dense_collection, create_qdrant_client
from rag_playground.retrieval.embedding import BGEEmbedder

pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(
        os.getenv("RUN_QDRANT_INTEGRATION") != "1",
        reason="set RUN_QDRANT_INTEGRATION=1 to use local Qdrant and cached BGE",
    ),
]


def test_real_chunk_index_search_and_document_aggregation(tmp_path):
    """Проверить полный путь от chunk JSONL до уникальных документов в выдаче."""
    settings = Settings.from_env()
    client = create_qdrant_client(settings.qdrant_url, settings.qdrant_timeout)
    embedder = BGEEmbedder(
        settings.dense_model_name,
        settings.dense_max_length,
        settings.dense_device,
        settings.dense_cache_dir,
    )
    collection = f"rag-chunks-integration-{uuid.uuid4().hex}"
    chunks_path = tmp_path / "fixed.jsonl"

    # У первого документа два фрагмента: тест одновременно проверяет индексацию
    # каждого чанка и последующее объединение результатов по одному doc_id.
    chunks = [
        Chunk(
            chunk_id="bones:fixed:0000",
            doc_id="bones",
            chunk_index=0,
            strategy="fixed",
            title="Vitamin D and bone health",
            text="Vitamin D supports calcium absorption and healthy bones.",
            token_count=16,
        ),
        Chunk(
            chunk_id="bones:fixed:0001",
            doc_id="bones",
            chunk_index=1,
            strategy="fixed",
            title="Vitamin D and bone health",
            text="Low vitamin D may reduce bone mineral density.",
            token_count=16,
        ),
        Chunk(
            chunk_id="ocean:fixed:0000",
            doc_id="ocean",
            chunk_index=0,
            strategy="fixed",
            title="Ocean circulation",
            text="Ocean currents move warm water across the planet.",
            token_count=13,
        ),
    ]
    chunks_path.write_text(
        "".join(json.dumps(asdict(chunk)) + "\n" for chunk in chunks),
        encoding="utf-8",
    )

    try:
        create_dense_collection(client, collection, embedder.dimension)
        first_report = index_chunk_dataset(
            client,
            collection,
            chunks_path,
            embedder,
            embedding_batch_size=3,
            upsert_batch_size=3,
        )
        # Повторный вызов обязан быть возобновляемым и не вычислять те же embeddings.
        second_report = index_chunk_dataset(
            client,
            collection,
            chunks_path,
            embedder,
            embedding_batch_size=3,
            upsert_batch_size=3,
        )
        result = search_chunks(
            client,
            collection,
            "fixed",
            embedder,
            "vitamin D and bone density",
            settings.dense_query_instruction,
            top_k=2,
            candidate_multiplier=2,
        )

        assert first_report.chunks_upserted == 3
        assert second_report.chunks_skipped == 3
        assert second_report.chunks_embedded == 0
        assert result.document_hits[0].doc_id == "bones"
        assert len({hit.doc_id for hit in result.document_hits}) == len(result.document_hits)
        assert result.document_hits[0].matched_chunks == 2
    finally:
        # Временная коллекция удаляется даже при падении assertion, поэтому тест
        # не засоряет Qdrant и безопасен для многократного запуска.
        if client.collection_exists(collection):
            client.delete_collection(collection)
