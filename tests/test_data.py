"""Тесты границы между форматом BEIR и внутренней схемой проекта."""

import pytest

from rag_playground.data import normalize_document, normalize_qrel, normalize_query


def test_normalize_document_uses_stable_schema():
    """Документ получает единообразные поля, строковый ID и очищенный текст."""
    assert normalize_document({"_id": 42, "title": "Title", "text": " Body "}) == {
        "doc_id": "42",
        "title": "Title",
        "text": "Body",
    }


def test_normalize_query_accepts_beir_fields():
    """Стандартные поля SciFact преобразуются в ``query_id`` и ``text``."""
    assert normalize_query({"_id": "q1", "text": "claim"}) == {
        "query_id": "q1",
        "text": "claim",
    }


def test_normalize_qrel_preserves_score():
    """Нормализация qrel не должна менять исходную оценку релевантности."""
    assert normalize_qrel({"query-id": "q1", "corpus-id": "d1", "score": 2}) == (
        "q1",
        "d1",
        2,
    )


def test_empty_document_is_rejected():
    """Документ без текста не должен незаметно попадать в будущий индекс."""
    with pytest.raises(ValueError):
        normalize_document({"_id": "d1", "text": ""})

