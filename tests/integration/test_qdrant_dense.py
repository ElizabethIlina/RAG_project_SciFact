"""Интеграционный тест dense retrieval на реальном локальном Qdrant."""

from __future__ import annotations

import os
import uuid

import pytest

from rag_playground.config import Settings
from rag_playground.retrieval.dense import (
    create_dense_collection,
    create_qdrant_client,
    search_dense,
)
from rag_playground.retrieval.embedding import BGEEmbedder

pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(
        os.getenv("RUN_QDRANT_INTEGRATION") != "1",
        reason="set RUN_QDRANT_INTEGRATION=1 to use local Qdrant and cached BGE",
    ),
]


def test_real_bge_and_qdrant_rank_relevant_text_first():
    """Настоящие BGE embeddings и cosine search должны работать вместе."""
    settings = Settings.from_env()
    client = create_qdrant_client(settings.qdrant_url, settings.qdrant_timeout)
    embedder = BGEEmbedder(
        settings.dense_model_name,
        settings.dense_max_length,
        settings.dense_device,
        settings.dense_cache_dir,
    )
    collection = f"rag-dense-integration-{uuid.uuid4().hex}"
    try:
        create_dense_collection(client, collection, embedder.dimension)
        from qdrant_client import models

        texts = [
            "Vitamin D supports calcium absorption and healthy bones.",
            "Ocean currents move warm water across the planet.",
        ]
        vectors = embedder.embed_documents(texts, batch_size=2)
        client.upsert(
            collection,
            points=[
                models.PointStruct(
                    id=index,
                    vector=vector,
                    payload={"doc_id": str(index), "title": "", "text": text},
                )
                for index, (text, vector) in enumerate(zip(texts, vectors, strict=True))
            ],
            wait=True,
        )
        result = search_dense(
            client,
            collection,
            embedder,
            "vitamin D and bone health",
            settings.dense_query_instruction,
            2,
        )
        assert result.hits[0].doc_id == "0"
    finally:
        if client.collection_exists(collection):
            client.delete_collection(collection)
