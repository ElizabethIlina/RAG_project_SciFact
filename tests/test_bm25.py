"""Быстрые unit-тесты BM25 без запуска Elasticsearch."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from rag_playground.retrieval.bm25 import (
    build_content,
    build_index_definition,
    build_search_query,
    create_index,
    index_corpus,
    iter_bulk_actions,
    parse_search_response,
    search_bm25,
)
from rag_playground.retrieval.models import BM25Hit, BM25QueryResult
from rag_playground.retrieval.runs import read_test_queries, save_bm25_run


class FakeIndices:
    """Минимальная подмена ``client.indices`` для проверки lifecycle индекса."""

    def __init__(self, exists: bool = False) -> None:
        self.exists_value = exists
        self.created: list[dict] = []
        self.deleted: list[str] = []
        self.refreshed: list[str] = []

    def exists(self, *, index: str) -> bool:
        return self.exists_value

    def delete(self, *, index: str) -> None:
        self.deleted.append(index)
        self.exists_value = False

    def create(self, **kwargs) -> None:
        self.created.append(kwargs)
        self.exists_value = True

    def refresh(self, *, index: str) -> None:
        self.refreshed.append(index)


class FakeClient:
    """Подмена клиента с программируемым search-ответом и count."""

    def __init__(self, *, exists: bool = False, count: int = 0, response=None) -> None:
        self.indices = FakeIndices(exists)
        self.count_value = count
        self.response = response or {"hits": {"hits": []}}
        self.search_calls: list[dict] = []

    def count(self, *, index: str) -> dict[str, int]:
        return {"count": self.count_value}

    def search(self, **kwargs):
        self.search_calls.append(kwargs)
        return self.response


def test_content_and_mapping_define_one_comparable_search_field():
    """Заголовок и текст должны попадать в англоязычное ``content``."""
    assert build_content(" Title ", " Body ") == "Title\n\nBody"
    assert build_content("", " Body ") == "Body"
    definition = build_index_definition()
    assert definition["mappings"]["properties"]["content"] == {
        "type": "text",
        "analyzer": "english",
    }
    assert definition["mappings"]["dynamic"] == "strict"


def test_create_index_never_deletes_without_explicit_recreate():
    """Существующий индекс без ``recreate`` остаётся нетронутым."""
    client = FakeClient(exists=True)
    assert create_index(client, "scifact-documents-v1") is False
    assert client.indices.deleted == []
    assert client.indices.created == []

    assert create_index(client, "scifact-documents-v1", recreate=True) is True
    assert client.indices.deleted == ["scifact-documents-v1"]
    assert client.indices.created[0]["index"] == "scifact-documents-v1"


def test_bulk_actions_use_doc_id_as_elasticsearch_id():
    """Повторная индексация того же doc_id не должна создавать дубликат."""
    actions = list(
        iter_bulk_actions(
            [{"doc_id": "42", "title": "A title", "text": "A body"}],
            "scifact-documents-v1",
        )
    )
    assert actions[0]["_id"] == "42"
    assert actions[0]["_source"]["content"] == "A title\n\nA body"


def test_index_corpus_counts_successes_and_failures(tmp_path: Path):
    """Отчёт индексации должен разделять успешные и ошибочные bulk actions."""
    corpus_path = tmp_path / "corpus.jsonl"
    corpus_path.write_text(
        '\n'.join(
            [
                json.dumps({"doc_id": "1", "title": "One", "text": "Text one"}),
                json.dumps({"doc_id": "2", "title": "Two", "text": "Text two"}),
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    client = FakeClient(count=1)

    def fake_streaming_bulk(_client, actions, **kwargs):
        materialized = list(actions)
        assert kwargs["chunk_size"] == 50
        yield True, {"index": {"_id": materialized[0]["_id"]}}
        yield False, {"index": {"_id": materialized[1]["_id"]}}

    report = index_corpus(
        client,
        "test-index",
        corpus_path,
        batch_size=50,
        streaming_bulk_fn=fake_streaming_bulk,
    )
    assert report.documents_read == 2
    assert report.documents_indexed == 1
    assert report.documents_failed == 1
    assert report.final_document_count == 1
    assert client.indices.refreshed == ["test-index"]


def test_search_response_gets_stable_one_based_ranks():
    """Ранги ответа Elasticsearch начинаются с 1, а ID всегда строковый."""
    response = {
        "hits": {
            "hits": [
                {"_id": 7, "_score": 3.5, "_source": {"title": "First", "text": "A"}},
                {
                    "_id": "8",
                    "_score": 2.0,
                    "_source": {"doc_id": "8", "title": "Second", "text": "B"},
                },
            ]
        }
    }
    hits = parse_search_response(response)
    assert [hit.rank for hit in hits] == [1, 2]
    assert [hit.doc_id for hit in hits] == ["7", "8"]

    client = FakeClient(response=response)
    result = search_bm25(client, "index", " scientific claim ", 2, "q1")
    assert result.query_text == "scientific claim"
    assert result.query_id == "q1"
    assert client.search_calls[0]["query"] == build_search_query("scientific claim")


@pytest.mark.parametrize("query", ["", "   "])
def test_empty_query_is_rejected(query: str):
    """Пустой запрос не должен отправляться в Elasticsearch."""
    with pytest.raises(ValueError):
        build_search_query(query)


def test_qrels_select_each_query_once(tmp_path: Path):
    """Несколько relevant docs не должны дублировать запрос в batch run."""
    queries_path = tmp_path / "queries.jsonl"
    queries_path.write_text(
        '{"query_id":"1","text":"first"}\n{"query_id":"2","text":"second"}\n',
        encoding="utf-8",
    )
    qrels_path = tmp_path / "test.tsv"
    qrels_path.write_text(
        "query_id\tdoc_id\tscore\n1\td1\t1\n1\td2\t1\n2\td3\t1\n",
        encoding="utf-8",
    )
    assert read_test_queries(queries_path, qrels_path) == [
        ("1", "first"),
        ("2", "second"),
    ]


def test_saved_run_contains_ranking_queries_and_manifest(tmp_path: Path):
    """Три артефакта запуска сохраняют ranking, latency и метаданны."""
    result = BM25QueryResult(
        query_id="q1",
        query_text="claim",
        top_k=10,
        latency_ms=4.2,
        hits=(BM25Hit("d1", "Title", "Text", 1, 3.5),),
    )
    paths = save_bm25_run(
        [result],
        tmp_path,
        "demo",
        {"index_name": "index", "top_k": 10},
    )
    ranking = json.loads(paths["ranking"].read_text(encoding="utf-8"))
    manifest = json.loads(paths["manifest"].read_text(encoding="utf-8"))
    assert ranking["retrieval_method"] == "bm25"
    assert ranking["rank"] == 1
    assert manifest["queries"] == 1
    assert manifest["ranking_rows"] == 1
    assert manifest["mean_query_latency_ms"] == 4.2

