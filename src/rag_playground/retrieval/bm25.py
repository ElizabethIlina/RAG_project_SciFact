"""Документный BM25 baseline на Elasticsearch.

Модуль отвечает только за жизненный цикл индекса и поиск. Расчёт Recall/nDCG
будет добавлен отдельно, чтобы BM25, dense и hybrid оценивались одним и тем же кодом.
"""

from __future__ import annotations

import json
from collections.abc import Iterable, Iterator, Mapping
from pathlib import Path
from time import perf_counter
from typing import Any

from elasticsearch import Elasticsearch, helpers

from rag_playground.retrieval.models import BM25Hit, BM25QueryResult, IndexingReport


def create_elasticsearch_client(url: str, request_timeout: int) -> Elasticsearch:
    """Создать клиент Elasticsearch для локального учебного стенда.

    Повтор включён только для timeout: сервис может ещё запускаться. Ошибки
    запроса не повторяются бесконечно и будут видны вызывающему коду.
    """
    return Elasticsearch(
        url,
        request_timeout=request_timeout,
        max_retries=2,
        retry_on_timeout=True,
    )


def build_content(title: str, text: str) -> str:
    """Детерминированно объединить заголовок и текст для поиска.

    Между непустыми частями ставятся два переноса. Позже такой же ``content`` можно
    передавать embedding-модели, чтобы сравнивать BM25 и dense на одинаковом тексте.
    """
    parts = [part.strip() for part in (title, text) if part and part.strip()]
    return "\n\n".join(parts)


def build_index_definition() -> dict[str, Any]:
    """Вернуть фиксированные settings и mapping документного индекса.

    Один shard и ноль replicas достаточны для локального корпуса из 5 183 документов.
    ``english`` выбран, потому что SciFact англоязычный. Параметры BM25 ``k1`` и ``b`` не
    переопределяются: первый прогон должен быть baseline без подгонки под test qrels.
    """
    return {
        "settings": {
            "number_of_shards": 1,
            "number_of_replicas": 0,
        },
        "mappings": {
            "dynamic": "strict",
            "properties": {
                "doc_id": {"type": "keyword"},
                "title": {"type": "text", "index": False},
                "text": {"type": "text", "index": False},
                "content": {"type": "text", "analyzer": "english"},
            },
        },
    }


def create_index(client: Any, index_name: str, recreate: bool = False) -> bool:
    """Создать индекс, не удаляя существующие данные без явного флага.

    Функция вернёт ``True``, если индекс был создан. Без ``recreate`` она вернёт
    ``False`` и оставит существующий индекс нетронутым. Удаление всегда адресовано
    точному имени без wildcard.
    """
    exists = bool(client.indices.exists(index=index_name))
    if exists and not recreate:
        return False
    if exists:
        client.indices.delete(index=index_name)

    definition = build_index_definition()
    client.indices.create(
        index=index_name,
        settings=definition["settings"],
        mappings=definition["mappings"],
    )
    return True


def read_corpus(corpus_path: Path) -> Iterator[dict[str, str]]:
    """Построчно прочитать нормализованный ``corpus.jsonl``.

    Валидация повторяется на границе индексации: повреждённый или
    изменённый вручную файл не должен попасть в Elasticsearch.
    """
    if not corpus_path.exists():
        raise FileNotFoundError(f"Corpus not found: {corpus_path}")

    with corpus_path.open(encoding="utf-8") as source:
        for line_number, line in enumerate(source, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            doc_id = str(row.get("doc_id", "")).strip()
            title = str(row.get("title", "")).strip()
            text = str(row.get("text", "")).strip()
            if not doc_id or not text:
                raise ValueError(
                    f"Invalid corpus row at line {line_number}: doc_id and text are required"
                )
            yield {"doc_id": doc_id, "title": title, "text": text}


def iter_bulk_actions(
    documents: Iterable[Mapping[str, str]], index_name: str
) -> Iterator[dict[str, Any]]:
    """Превратить документы в ленивый поток Bulk API actions.

    ``doc_id`` одновременно становится Elasticsearch ``_id``. Повторная индексация
    того же корпуса обновит те же записи, а не создаст дубликаты.
    """
    for document in documents:
        doc_id = str(document["doc_id"])
        title = str(document.get("title", ""))
        text = str(document["text"])
        yield {
            "_op_type": "index",
            "_index": index_name,
            "_id": doc_id,
            "_source": {
                "doc_id": doc_id,
                "title": title,
                "text": text,
                "content": build_content(title, text),
            },
        }


def index_corpus(
    client: Any,
    index_name: str,
    corpus_path: Path,
    batch_size: int,
    streaming_bulk_fn: Any = helpers.streaming_bulk,
) -> IndexingReport:
    """Пакетно загрузить корпус, обновить index и проверить итоговый count.

    ``streaming_bulk`` выбран вместо ``bulk``, чтобы видеть успех или ошибку каждого
    документа и не хранить всю пачку в памяти. Функция-аргумент позволяет тестам
    подменить сетевую часть без запуска Elasticsearch.
    """
    started = perf_counter()
    documents_read = 0
    documents_indexed = 0
    documents_failed = 0

    def counted_actions() -> Iterator[dict[str, Any]]:
        nonlocal documents_read
        for action in iter_bulk_actions(read_corpus(corpus_path), index_name):
            documents_read += 1
            yield action

    for ok, _item in streaming_bulk_fn(
        client,
        counted_actions(),
        chunk_size=batch_size,
        raise_on_error=False,
        raise_on_exception=False,
    ):
        if ok:
            documents_indexed += 1
        else:
            documents_failed += 1

    # Refresh делает все успешные bulk-записи видимыми для count и поиска.
    client.indices.refresh(index=index_name)
    final_count = int(client.count(index=index_name)["count"])
    duration = perf_counter() - started

    return IndexingReport(
        index_name=index_name,
        documents_read=documents_read,
        documents_indexed=documents_indexed,
        documents_failed=documents_failed,
        duration_seconds=round(duration, 6),
        final_document_count=final_count,
    )


def build_search_query(query_text: str) -> dict[str, Any]:
    """Построить явный BM25 match-запрос к полю ``content``."""
    normalized = query_text.strip()
    if not normalized:
        raise ValueError("Query text must not be empty")
    return {"match": {"content": {"query": normalized}}}


def parse_search_response(response: Mapping[str, Any]) -> tuple[BM25Hit, ...]:
    """Превратить Elasticsearch hits в стабильные модели проекта."""
    raw_hits = response.get("hits", {}).get("hits", [])
    hits: list[BM25Hit] = []
    for rank, raw_hit in enumerate(raw_hits, start=1):
        source = raw_hit.get("_source", {})
        hits.append(
            BM25Hit(
                doc_id=str(source.get("doc_id") or raw_hit.get("_id", "")),
                title=str(source.get("title", "")),
                text=str(source.get("text", "")),
                rank=rank,
                score=float(raw_hit.get("_score") or 0.0),
            )
        )
    return tuple(hits)


def search_bm25(
    client: Any,
    index_name: str,
    query_text: str,
    top_k: int,
    query_id: str = "manual",
) -> BM25QueryResult:
    """Выполнить один BM25-запрос и измерить client-side latency.

    Задержка включает HTTP round trip и преобразование ответа. Именно это время
    видит клиент учебного стенда. Оно может отличаться от поля ``took`` Elasticsearch.
    """
    if top_k <= 0:
        raise ValueError("top_k must be greater than zero")
    query = build_search_query(query_text)
    started = perf_counter()
    response = client.search(
        index=index_name,
        query=query,
        size=top_k,
        source_includes=["doc_id", "title", "text"],
        track_total_hits=True,
    )
    hits = parse_search_response(response)
    latency_ms = (perf_counter() - started) * 1000
    return BM25QueryResult(
        query_id=str(query_id),
        query_text=query_text.strip(),
        top_k=top_k,
        latency_ms=round(latency_ms, 3),
        hits=hits,
    )


def get_index_info(client: Any, index_name: str) -> dict[str, Any]:
    """Получить компактную сводку о версии сервера, mapping и числе документов."""
    server_info = client.info()
    exists = bool(client.indices.exists(index=index_name))
    summary: dict[str, Any] = {
        "elasticsearch_version": server_info.get("version", {}).get("number", "unknown"),
        "index_name": index_name,
        "index_exists": exists,
    }
    if exists:
        summary["document_count"] = int(client.count(index=index_name)["count"])
        summary["mapping"] = client.indices.get_mapping(index=index_name)[index_name]["mappings"]
    return summary

