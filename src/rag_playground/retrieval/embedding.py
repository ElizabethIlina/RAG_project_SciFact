"""Локальные embeddings BGE Small и анализ длины корпуса.

FastEmbed запускает ONNX-версию модели на CPU. После первого скачивания модель
берётся из локального кэша; API-ключ и отправка текстов во внешний сервис не нужны.
"""

from __future__ import annotations

import json
from collections.abc import Iterable, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from statistics import fmean, median
from typing import Any

import numpy as np
from fastembed import TextEmbedding
from tokenizers import Tokenizer

from rag_playground.retrieval.bm25 import build_content, read_corpus


@dataclass(frozen=True, slots=True)
class TokenLengthReport:
    """Статистика длины документов в токенах BGE до обрезания."""

    model_name: str
    max_length: int
    documents: int
    min_tokens: int
    mean_tokens: float
    median_tokens: float
    p95_tokens: float
    max_tokens: int
    truncated_documents: int
    truncated_percent: float
    longest_documents: tuple[dict[str, Any], ...]


class BGEEmbedder:
    """Небольшая оболочка над FastEmbed с едиными правилами проекта.

    Отдельный tokenizer без truncation нужен для статистики: tokenizer,
    используемый ONNX-моделью, ограничивает вход 512 токенами и не может показать,
    насколько длинным был исходный документ.
    """

    def __init__(
        self,
        model_name: str,
        max_length: int,
        device: str = "cpu",
        cache_dir: Path | None = None,
    ) -> None:
        if max_length <= 0:
            raise ValueError("max_length must be greater than zero")
        if device != "cpu":
            raise ValueError("FastEmbed configuration currently supports DENSE_DEVICE=cpu")

        self.model_name = model_name
        self.max_length = max_length
        # CPUExecutionProvider фиксирует одинаковый и доступный на Mac backend.
        self._model = TextEmbedding(
            model_name=model_name,
            cache_dir=str(cache_dir) if cache_dir is not None else None,
            providers=["CPUExecutionProvider"],
        )
        self.dimension = int(self._model.model.model_description.dim)

        # FastEmbed уже скачал tokenizer.json рядом с ONNX-моделью. Загружаем
        # независимую копию и снимаем truncation только для анализа исходной длины.
        model_dir = Path(self._model.model._model_dir)  # noqa: SLF001
        self._length_tokenizer = Tokenizer.from_file(str(model_dir / "tokenizer.json"))
        self._length_tokenizer.no_truncation()

    def count_tokens(self, text: str) -> int:
        """Посчитать токены вместе со специальными токенами BGE."""
        return len(self._length_tokenizer.encode(text).ids)

    def encode_text(self, text: str) -> list[int]:
        """Преобразовать текст в token IDs без служебных CLS/SEP.

        Этот интерфейс используется chunking-кодом. Служебные токены добавит
        embedding-модель после сборки полного ``title + chunk``.
        """
        return list(self._length_tokenizer.encode(text, add_special_tokens=False).ids)

    def decode_tokens(self, token_ids: Sequence[int]) -> str:
        """Преобразовать token IDs обратно в нормализованный непустой текст."""
        return self._length_tokenizer.decode(list(token_ids), skip_special_tokens=True).strip()

    def embed_documents(self, texts: Sequence[str], batch_size: int) -> list[list[float]]:
        """Создать нормализованные векторы документов.

        FastEmbed сам обрезает вход по лимиту модели. Тексты формируются как
        ``title + abstract``, поэтому заголовок всегда остаётся в начале.
        """
        if not texts:
            return []
        vectors = self._model.embed(texts, batch_size=batch_size)
        return [np.asarray(vector, dtype=np.float32).tolist() for vector in vectors]

    def embed_query(self, text: str, instruction: str) -> list[float]:
        """Создать вектор запроса с рекомендованной retrieval-инструкцией BGE."""
        normalized = text.strip()
        if not normalized:
            raise ValueError("Query text must not be empty")
        query_text = f"{instruction.strip()} {normalized}".strip()
        return self.embed_documents([query_text], batch_size=1)[0]


def percentile(values: Sequence[int], percentile_value: float) -> float:
    """Посчитать линейный percentile без зависимости от pandas."""
    if not values:
        raise ValueError("values must not be empty")
    return float(np.percentile(np.asarray(values), percentile_value))


def analyze_corpus_lengths(
    corpus_path: Path,
    embedder: BGEEmbedder,
    longest_limit: int = 10,
) -> TokenLengthReport:
    """Измерить длину каждого документа тем же tokenizer, что использует BGE."""
    rows: list[tuple[int, str, str]] = []
    for document in read_corpus(corpus_path):
        content = build_content(document["title"], document["text"])
        rows.append(
            (
                embedder.count_tokens(content),
                document["doc_id"],
                document["title"],
            )
        )
    if not rows:
        raise ValueError("Corpus must contain at least one document")

    lengths = [row[0] for row in rows]
    truncated = sum(length > embedder.max_length for length in lengths)
    longest = sorted(rows, reverse=True)[:longest_limit]
    return TokenLengthReport(
        model_name=embedder.model_name,
        max_length=embedder.max_length,
        documents=len(rows),
        min_tokens=min(lengths),
        mean_tokens=round(fmean(lengths), 3),
        median_tokens=round(float(median(lengths)), 3),
        p95_tokens=round(percentile(lengths, 95), 3),
        max_tokens=max(lengths),
        truncated_documents=truncated,
        truncated_percent=round(truncated / len(rows) * 100, 3),
        longest_documents=tuple(
            {"doc_id": doc_id, "title": title, "tokens": tokens}
            for tokens, doc_id, title in longest
        ),
    )


def save_length_report(report: TokenLengthReport, output_path: Path) -> None:
    """Сохранить статистику корпуса как читаемый JSON."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(asdict(report), ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def batched(items: Sequence[Any], batch_size: int) -> Iterable[Sequence[Any]]:
    """Разбить последовательность на непустые пакеты фиксированного размера."""
    if batch_size <= 0:
        raise ValueError("batch_size must be greater than zero")
    for start in range(0, len(items), batch_size):
        yield items[start : start + batch_size]
