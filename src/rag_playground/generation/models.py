"""Типизированные модели воспроизводимого RAG-эксперимента."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class ExperimentQuery:
    """Один заранее выбранный запрос и его document-level разметка."""

    query_id: str
    query_text: str
    relevant_doc_ids: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class RAGSource:
    """Один пронумерованный фрагмент, который увидит модель генерации."""

    source_number: int
    doc_id: str
    chunk_id: str
    title: str
    text: str
    retrieval_rank: int
    retrieval_score: float


@dataclass(frozen=True, slots=True)
class PreparedRAGExample:
    """Полностью подготовленный prompt до любого обращения к GigaChat."""

    query_id: str
    query_text: str
    relevant_doc_ids: tuple[str, ...]
    retrieval_strategy: str
    retrieval_latency_ms: float
    sources: tuple[RAGSource, ...]
    retrieved_relevant_doc_ids: tuple[str, ...]
    prompt: str


@dataclass(frozen=True, slots=True)
class ProviderGeneration:
    """Минимальный безопасный результат одного ответа провайдера."""

    answer: str
    model: str
    finish_reason: str | None
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int


@dataclass(frozen=True, slots=True)
class GeneratedRAGAnswer:
    """Ответ вместе с retrieval-контекстом, токенами и проверкой цитат."""

    query_id: str
    query_text: str
    mode: str
    model: str
    prompt: str
    answer: str
    sources: tuple[RAGSource, ...]
    relevant_doc_ids: tuple[str, ...]
    retrieved_relevant_doc_ids: tuple[str, ...]
    cited_source_numbers: tuple[int, ...]
    invalid_source_numbers: tuple[int, ...]
    finish_reason: str | None
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    generation_latency_ms: float

