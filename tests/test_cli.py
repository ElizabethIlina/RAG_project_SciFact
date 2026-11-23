"""Регрессионные тесты маршрутизации командного интерфейса."""

from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

from rag_playground import cli
from rag_playground.retrieval.embedding import TokenLengthReport


def test_dense_analyze_calls_embedder_helper(monkeypatch, tmp_path: Path, capsys):
    """Локальная переменная другой команды не должна затенять функцию `_embedder`."""
    fake_settings = SimpleNamespace(data_dir=tmp_path)
    fake_embedder = object()
    report = TokenLengthReport(
        model_name="fake-bge",
        max_length=512,
        documents=1,
        min_tokens=10,
        mean_tokens=10.0,
        median_tokens=10.0,
        p95_tokens=10.0,
        max_tokens=10,
        truncated_documents=0,
        truncated_percent=0.0,
        longest_documents=(),
    )

    monkeypatch.setattr(cli.Settings, "from_env", lambda: fake_settings)
    monkeypatch.setattr(cli, "_embedder", lambda settings: fake_embedder)
    monkeypatch.setattr(
        cli,
        "analyze_corpus_lengths",
        lambda corpus_path, embedder: report,
    )
    monkeypatch.setattr(cli, "save_length_report", lambda result, output: None)
    monkeypatch.setattr(sys, "argv", ["rag-playground", "dense-analyze"])

    assert cli.main() == 0
    assert '"model_name": "fake-bge"' in capsys.readouterr().out
