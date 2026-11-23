"""Unit-тесты RAG-потока без обращения к GigaChat и расходования токенов."""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import pytest

from rag_playground.generation import client as client_module
from rag_playground.generation.client import require_safe_generation_batch
from rag_playground.generation.context import (
    build_prepared_example,
    load_experiment_queries,
    load_prepared_examples,
    save_prepared_examples,
    select_experiment_queries,
)
from rag_playground.generation.evaluation import evaluate_answer_file
from rag_playground.generation.models import (
    ExperimentQuery,
    PreparedRAGExample,
    ProviderGeneration,
    RAGSource,
)
from rag_playground.generation.prompting import (
    build_rag_prompt,
    extract_citations,
    invalid_citations,
)
from rag_playground.generation.run import (
    append_generated_answers,
    ensure_answers_do_not_exist,
    generate_examples,
)
from rag_playground.retrieval.models import ChunkDocumentHit, ChunkQueryResult


def make_retrieval() -> ChunkQueryResult:
    """Создать детерминированную выдачу с релевантным и нерелевантным документом."""
    hits = (
        ChunkDocumentHit(
            doc_id="relevant",
            title="Vitamin D",
            rank=1,
            score=0.9,
            best_chunk_id="relevant:recursive:0000",
            best_chunk_index=0,
            best_chunk_text="Vitamin D supports calcium absorption and bone health.",
            matched_chunks=2,
        ),
        ChunkDocumentHit(
            doc_id="other",
            title="Ocean",
            rank=2,
            score=0.2,
            best_chunk_id="other:recursive:0000",
            best_chunk_index=0,
            best_chunk_text="Ocean currents transport warm water.",
            matched_chunks=1,
        ),
    )
    return ChunkQueryResult(
        query_id="q1",
        query_text="Vitamin D improves bone health.",
        strategy="recursive",
        top_k=2,
        candidate_chunks=10,
        embedding_latency_ms=1.0,
        search_latency_ms=2.0,
        chunk_hits=(),
        document_hits=hits,
    )


def make_prepared() -> PreparedRAGExample:
    """Собрать валидный prepared example через настоящий context builder."""
    query = ExperimentQuery(
        query_id="q1",
        query_text="Vitamin D improves bone health.",
        relevant_doc_ids=("relevant",),
    )
    return build_prepared_example(query, make_retrieval(), 2, 5000)


class FakeGenerator:
    """Вернуть предсказуемый ответ, не создавая сетевых соединений."""

    def generate(self, prompt: str):
        assert "Vitamin D" in prompt
        return (
            ProviderGeneration(
                answer="ВЫВОД: ПОДТВЕРЖДАЕТСЯ\nОбъяснение [1].\nИСТОЧНИКИ: [1]",
                model="GigaChat-2",
                finish_reason="stop",
                prompt_tokens=100,
                completion_tokens=20,
                total_tokens=120,
            ),
            12.5,
        )


def test_prompt_contains_numbered_sources_and_strict_rules():
    """Модель должна видеть только пронумерованный контекст и запрет на выдумки."""
    prepared = make_prepared()

    assert "используя ТОЛЬКО" in prepared.prompt
    assert "[1] doc_id=relevant" in prepared.prompt
    assert "[2] doc_id=other" in prepared.prompt
    assert prepared.retrieved_relevant_doc_ids == ("relevant",)


def test_citation_parser_finds_unique_and_invalid_numbers():
    """Повторная корректная ссылка не должна считаться новой или ошибочной."""
    answer = "Факт [1], ещё факт [1], неизвестный источник [7]."
    assert extract_citations(answer) == (1, 7)
    assert invalid_citations(answer, 2) == (7,)


def test_generation_batch_is_hard_limited_to_five():
    """Даже неверная конфигурация не должна разрешить шестой API-запрос."""
    assert len(require_safe_generation_batch(range(5), configured_limit=99)) == 5
    with pytest.raises(ValueError, match="1..5"):
        require_safe_generation_batch(range(6), configured_limit=99)
    with pytest.raises(ValueError, match="1..5"):
        require_safe_generation_batch([], configured_limit=5)


def test_fake_generation_preserves_usage_and_validates_citations(tmp_path: Path):
    """Полный pipeline должен работать с fake-клиентом и сохранять ответ JSONL."""
    prepared = make_prepared()
    answers = generate_examples(FakeGenerator(), [prepared], "rag", batch_limit=5)
    output = tmp_path / "answers.jsonl"
    append_generated_answers(answers, output)

    row = json.loads(output.read_text(encoding="utf-8"))
    assert row["model"] == "GigaChat-2"
    assert row["total_tokens"] == 120
    assert row["cited_source_numbers"] == [1]
    assert row["invalid_source_numbers"] == []
    assert evaluate_answer_file(output)["rag_contexts_with_relevant_document"] == 1


def test_duplicate_answer_is_rejected_before_generation(tmp_path: Path):
    """Повторный batch не должен зря расходовать токены на уже готовый query ID."""
    output = tmp_path / "answers.jsonl"
    answers = generate_examples(FakeGenerator(), [make_prepared()], "rag", 5)
    append_generated_answers(answers, output)

    with pytest.raises(ValueError, match="already exist"):
        ensure_answers_do_not_exist(output, ["q1"], "rag")


def test_prepared_jsonl_round_trip(tmp_path: Path):
    """Exact prompt и источники должны сохраняться без повторного retrieval."""
    path = tmp_path / "prepared.jsonl"
    save_prepared_examples([make_prepared()], path)
    loaded = load_prepared_examples(path)

    assert loaded == [make_prepared()]


def test_experiment_plan_selection_validates_ids(tmp_path: Path):
    """Неизвестный ID не должен молча заменяться другим примером."""
    path = tmp_path / "plan.json"
    path.write_text(
        json.dumps(
            {
                "examples": [
                    {
                        "query_id": "q1",
                        "query_text": "Claim",
                        "relevant_doc_ids": ["d1"],
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    queries = load_experiment_queries(path)

    assert select_experiment_queries(queries, ["q1"]) == queries
    with pytest.raises(ValueError, match="Unknown"):
        select_experiment_queries(queries, ["missing"])


def test_context_budget_rejects_oversized_first_source():
    """Нельзя незаметно отправлять prompt крупнее заданного бюджета."""
    query = ExperimentQuery("q1", "Claim", ("relevant",))
    with pytest.raises(ValueError, match="first source"):
        build_prepared_example(query, make_retrieval(), 2, 10)


def test_baseline_has_no_sources_or_citations():
    """Контрольный режим не должен притворяться retrieval-ответом."""
    answer = generate_examples(FakeGenerator(), [make_prepared()], "baseline", 5)[0]
    assert answer.sources == ()
    assert answer.invalid_source_numbers == (1,)


def test_build_prompt_rejects_empty_source_list():
    """RAG без контекста должен завершиться до обращения к провайдеру."""
    with pytest.raises(ValueError, match="at least one"):
        build_rag_prompt("Claim", ())


def test_replaced_example_keeps_dataclass_immutable():
    """Тестовый helper подтверждает, что experiment artifacts не меняются на месте."""
    original = make_prepared()
    changed = replace(original, query_id="q2")
    assert original.query_id == "q1"
    assert changed.query_id == "q2"


def test_source_dataclass_has_explicit_number():
    """Номер источника является данными, а не зависит от позиции после загрузки."""
    source = RAGSource(1, "d", "c", "T", "X", 1, 0.5)
    assert source.source_number == 1


def test_sdk_client_uses_exact_gigachat_2_model(monkeypatch):
    """Клиент должен собирать Chat ровно с моделью, указанной пользователем."""
    from types import SimpleNamespace

    captured = {}

    class FakeSDKClient:
        def __init__(self, **kwargs):
            captured["client_kwargs"] = kwargs

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, traceback):
            return None

        def chat(self, chat):
            captured["chat"] = chat
            return SimpleNamespace(
                choices=[
                    SimpleNamespace(
                        message=SimpleNamespace(content="Ответ"),
                        finish_reason="stop",
                    )
                ],
                model="GigaChat-2",
                usage=SimpleNamespace(
                    prompt_tokens=10,
                    completion_tokens=2,
                    total_tokens=12,
                ),
            )

    settings = SimpleNamespace(
        gigachat_credentials="secret-for-test",
        gigachat_model="GigaChat-2",
        rag_generation_max_tokens=512,
        gigachat_base_url="https://api.giga.chat/v1",
        gigachat_scope="GIGACHAT_API_PERS",
        gigachat_verify_ssl_certs=False,
        gigachat_ca_bundle_file=None,
    )
    monkeypatch.setattr(client_module, "GigaChat", FakeSDKClient)

    with client_module.GigaChatGenerator(settings) as generator:
        result, latency_ms = generator.generate("PROMPT")

    assert captured["chat"].model == "GigaChat-2"
    assert captured["chat"].messages[0].content == "PROMPT"
    assert captured["client_kwargs"]["verify_ssl_certs"] is False
    assert result.total_tokens == 12
    assert latency_ms >= 0
