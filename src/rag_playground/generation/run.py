"""Batch-генерация с лимитом пять запросов и сохранением всех артефактов."""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path
from typing import Protocol

from rag_playground.generation.client import require_safe_generation_batch
from rag_playground.generation.models import (
    GeneratedRAGAnswer,
    PreparedRAGExample,
    ProviderGeneration,
)
from rag_playground.generation.prompting import (
    build_baseline_prompt,
    extract_citations,
    invalid_citations,
)


class TextGenerator(Protocol):
    """Маленький интерфейс позволяет тестировать pipeline без внешнего API."""

    def generate(self, prompt: str) -> tuple[ProviderGeneration, float]: ...


def generate_examples(
    generator: TextGenerator,
    examples: list[PreparedRAGExample],
    mode: str,
    batch_limit: int,
) -> list[GeneratedRAGAnswer]:
    """Сделать не больше пяти последовательных вызовов и проверить цитаты."""
    if mode not in {"rag", "baseline"}:
        raise ValueError("mode must be 'rag' or 'baseline'")
    safe_examples = require_safe_generation_batch(examples, batch_limit)
    answers: list[GeneratedRAGAnswer] = []
    for item in safe_examples:
        if not isinstance(item, PreparedRAGExample):
            raise TypeError("Expected PreparedRAGExample")
        prompt = item.prompt if mode == "rag" else build_baseline_prompt(item.query_text)
        provider_result, latency_ms = generator.generate(prompt)
        citations = extract_citations(provider_result.answer)
        source_count = len(item.sources) if mode == "rag" else 0
        answers.append(
            GeneratedRAGAnswer(
                query_id=item.query_id,
                query_text=item.query_text,
                mode=mode,
                model=provider_result.model,
                prompt=prompt,
                answer=provider_result.answer,
                sources=item.sources if mode == "rag" else (),
                relevant_doc_ids=item.relevant_doc_ids,
                retrieved_relevant_doc_ids=item.retrieved_relevant_doc_ids,
                cited_source_numbers=citations,
                invalid_source_numbers=invalid_citations(
                    provider_result.answer, source_count
                ),
                finish_reason=provider_result.finish_reason,
                prompt_tokens=provider_result.prompt_tokens,
                completion_tokens=provider_result.completion_tokens,
                total_tokens=provider_result.total_tokens,
                generation_latency_ms=latency_ms,
            )
        )
    return answers


def append_generated_answers(
    answers: list[GeneratedRAGAnswer],
    path: Path,
) -> None:
    """Дописывать разрешённые batches, не стирая результаты прошлых запусков."""
    path.parent.mkdir(parents=True, exist_ok=True)
    existing_ids: set[tuple[str, str]] = set()
    if path.exists():
        with path.open(encoding="utf-8") as source:
            for line in source:
                if line.strip():
                    row = json.loads(line)
                    existing_ids.add((str(row["query_id"]), str(row["mode"])))
    duplicates = [
        answer.query_id
        for answer in answers
        if (answer.query_id, answer.mode) in existing_ids
    ]
    if duplicates:
        raise ValueError(f"Answers already exist for query IDs: {duplicates}")
    with path.open("a", encoding="utf-8") as target:
        for answer in answers:
            target.write(
                json.dumps(asdict(answer), ensure_ascii=False, sort_keys=True) + "\n"
            )


def ensure_answers_do_not_exist(
    path: Path,
    query_ids: list[str],
    mode: str,
) -> None:
    """До платных вызовов убедиться, что batch не дублирует готовые ответы."""
    if not path.exists():
        return
    requested = {(query_id, mode) for query_id in query_ids}
    existing: set[tuple[str, str]] = set()
    with path.open(encoding="utf-8") as source:
        for line in source:
            if line.strip():
                row = json.loads(line)
                existing.add((str(row["query_id"]), str(row["mode"])))
    duplicates = sorted(query_id for query_id, _mode in requested.intersection(existing))
    if duplicates:
        raise ValueError(f"Answers already exist for query IDs: {duplicates}")
