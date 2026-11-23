"""Интеграционная проверка BM25 на реальном Elasticsearch."""

from __future__ import annotations

import os
import uuid

import pytest
from elasticsearch import helpers

from rag_playground.config import Settings
from rag_playground.retrieval.bm25 import (
    create_elasticsearch_client,
    create_index,
    iter_bulk_actions,
    search_bm25,
)

pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(
        os.getenv("RUN_ELASTICSEARCH_INTEGRATION") != "1",
        reason="set RUN_ELASTICSEARCH_INTEGRATION=1 to use local Elasticsearch",
    ),
]


def test_relevant_document_is_ranked_first():
    """Реальный English analyzer и BM25 должны поднять документ с точными терминами."""
    settings = Settings.from_env()
    client = create_elasticsearch_client(
        settings.elasticsearch_url, settings.elasticsearch_request_timeout
    )
    # UUID не даёт параллельным тестам испортить индексы друг друга.
    index_name = f"rag-bm25-integration-{uuid.uuid4().hex}"
    try:
        create_index(client, index_name)
        documents = [
            {
                "doc_id": "relevant",
                "title": "Vitamin D and bone density",
                "text": "Vitamin D improves calcium absorption and supports bone density.",
            },
            {
                "doc_id": "irrelevant",
                "title": "Ocean currents",
                "text": "Ocean currents transport warm water across the planet.",
            },
        ]
        helpers.bulk(client, iter_bulk_actions(documents, index_name), refresh="wait_for")
        result = search_bm25(client, index_name, "vitamin D bone density", 2, "q")
        assert result.hits[0].doc_id == "relevant"
    finally:
        # Удаляем только точное случайное имя, без wildcard и без затрагивания основного индекса.
        if client.indices.exists(index=index_name):
            client.indices.delete(index=index_name)

