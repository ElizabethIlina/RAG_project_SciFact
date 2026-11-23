"""Unit-тесты Langfuse backfill без сети и реального Cloud-проекта."""

from __future__ import annotations

import csv
import hashlib
import json
from contextlib import contextmanager
from pathlib import Path

from rag_playground.observability.backfill import backfill_rag_traces
from rag_playground.observability.redaction import redact_payload


class FakeObservation:
    """Записать структуру observations/scores вместо отправки в Langfuse."""

    def __init__(self, client, payload):
        self.client = client
        self.payload = payload

    def set_trace_io(self, **payload):
        self.client.trace_io.append(payload)
        return self

    @contextmanager
    def start_as_current_observation(self, **payload):
        self.client.observations.append(payload)
        yield FakeObservation(self.client, payload)

    def score_trace(self, **payload):
        self.client.scores.append(payload)


class FakeLangfuse:
    """Реализовать только интерфейс, используемый backfill-кодом."""

    def __init__(self):
        self.observations = []
        self.scores = []
        self.trace_io = []
        self.flushes = 0

    @staticmethod
    def create_trace_id(*, seed: str) -> str:
        return hashlib.sha256(seed.encode()).hexdigest()[:32]

    @contextmanager
    def start_as_current_observation(self, **payload):
        self.observations.append(payload)
        yield FakeObservation(self, payload)

    def flush(self):
        self.flushes += 1


def write_backfill_inputs(tmp_path: Path) -> tuple[Path, Path, Path, Path]:
    """Создать минимальный, но полный набор stage-5 артефактов."""
    prepared = tmp_path / "prepared.jsonl"
    answers = tmp_path / "answers.jsonl"
    manual = tmp_path / "manual.csv"
    state = tmp_path / "state.json"
    prepared.write_text(
        json.dumps(
            {
                "query_id": "q1",
                "query_text": "Claim",
                "relevant_doc_ids": ["d1"],
                "retrieval_strategy": "recursive",
                "retrieval_latency_ms": 3.0,
                "retrieved_relevant_doc_ids": ["d1"],
                "prompt": "PROMPT",
                "sources": [
                    {
                        "source_number": 1,
                        "doc_id": "d1",
                        "chunk_id": "c1",
                        "title": "Title",
                        "text": "Evidence",
                        "retrieval_rank": 1,
                        "retrieval_score": 0.9,
                    }
                ],
            }
        )
        + "\n",
        encoding="utf-8",
    )
    answers.write_text(
        json.dumps(
            {
                "query_id": "q1",
                "query_text": "Claim",
                "mode": "rag",
                "model": "GigaChat-2",
                "prompt": "PROMPT",
                "answer": "ВЫВОД: ПОДТВЕРЖДАЕТСЯ [1]",
                "cited_source_numbers": [1],
                "invalid_source_numbers": [],
                "prompt_tokens": 100,
                "completion_tokens": 20,
                "total_tokens": 120,
                "generation_latency_ms": 50.0,
                "finish_reason": "stop",
            }
        )
        + "\n",
        encoding="utf-8",
    )
    with manual.open("w", encoding="utf-8", newline="") as target:
        writer = csv.DictWriter(
            target,
            fieldnames=[
                "query_id",
                "mode",
                "correctness_0_2",
                "groundedness_0_2",
                "completeness_0_2",
                "citation_quality_0_2",
                "error_type",
                "notes",
            ],
        )
        writer.writeheader()
        writer.writerow(
            {
                "query_id": "q1",
                "mode": "rag",
                "correctness_0_2": "2",
                "groundedness_0_2": "2",
                "completeness_0_2": "2",
                "citation_quality_0_2": "2",
                "error_type": "none",
                "notes": "ok",
            }
        )
    return prepared, answers, manual, state


def test_redaction_masks_sensitive_keys_and_exact_secrets():
    """Credentials не должны попасть в trace даже внутри вложенной структуры."""
    payload = {
        "authorization": "Bearer secret",
        "nested": {"text": "prefix-real-key-suffix", "safe": "value"},
        "access_token": "token",
    }
    redacted = redact_payload(payload, ("real-key",))

    assert redacted["authorization"] == "[REDACTED]"
    assert redacted["nested"]["text"] == "prefix-[REDACTED]-suffix"
    assert redacted["nested"]["safe"] == "value"
    assert redacted["access_token"] == "[REDACTED]"


def test_backfill_creates_expected_structure_and_is_idempotent(tmp_path: Path):
    """Повторный импорт должен пропустить trace и не создавать новые units."""
    prepared, answers, manual, state = write_backfill_inputs(tmp_path)
    first_client = FakeLangfuse()
    first = backfill_rag_traces(first_client, prepared, answers, manual, state)

    assert first.imported == 1
    assert first.estimated_units == 12
    assert len(first_client.observations) == 5  # root + четыре child observation
    assert len(first_client.scores) == 6
    assert first_client.flushes == 1
    assert len(json.loads(state.read_text())["imported_trace_ids"]) == 1

    second_client = FakeLangfuse()
    second = backfill_rag_traces(second_client, prepared, answers, manual, state)
    assert second.imported == 0
    assert second.skipped == 1
    assert second.estimated_units == 0
    assert second_client.observations == []


def test_trace_id_is_stable_for_same_query():
    """Seed гарантирует один trace для одного режима и query ID."""
    client = FakeLangfuse()
    seed = "rag-playground/stage-5/rag/q1"
    assert client.create_trace_id(seed=seed) == client.create_trace_id(seed=seed)
