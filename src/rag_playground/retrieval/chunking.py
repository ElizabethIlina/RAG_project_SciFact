"""Три воспроизводимые стратегии разбиения SciFact на токен-безопасные чанки."""

from __future__ import annotations

import json
import re
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from statistics import fmean, median
from typing import Literal

from rag_playground.retrieval.bm25 import build_content, read_corpus
from rag_playground.retrieval.embedding import BGEEmbedder, percentile

ChunkStrategy = Literal["fixed", "recursive", "sentence"]
CHUNK_STRATEGIES: tuple[ChunkStrategy, ...] = ("fixed", "recursive", "sentence")


@dataclass(frozen=True, slots=True)
class Chunk:
    """Один фрагмент документа с достаточными данными для поиска и диагностики."""

    chunk_id: str
    doc_id: str
    chunk_index: int
    strategy: ChunkStrategy
    title: str
    text: str
    token_count: int


@dataclass(frozen=True, slots=True)
class ChunkingReport:
    """Статистика одного подготовленного набора чанков."""

    strategy: ChunkStrategy
    source_documents: int
    chunks: int
    documents_with_multiple_chunks: int
    mean_chunks_per_document: float
    min_tokens: int
    mean_tokens: float
    median_tokens: float
    p95_tokens: float
    max_tokens: int
    empty_chunks: int
    over_limit_chunks: int
    chunk_size: int
    overlap_tokens: int
    sentence_overlap: int
    embedding_model: str


def _body_budget(title: str, embedder: BGEEmbedder, chunk_size: int) -> int:
    """Оценить число body-токенов, оставшееся после заголовка.

    Итог каждого кандидата дополнительно проверяется точным ``count_tokens``:
    tokenization на границе заголовка и текста может отличаться на один токен.
    """
    budget = chunk_size - embedder.count_tokens(title)
    if budget <= 0:
        raise ValueError(f"Title alone does not fit chunk_size={chunk_size}: {title!r}")
    return budget


def _fits(title: str, text: str, embedder: BGEEmbedder, chunk_size: int) -> bool:
    """Проверить полный вход ``title + text`` тем же tokenizer, что и BGE."""
    return embedder.count_tokens(build_content(title, text)) <= chunk_size


def _fixed_token_parts(
    title: str,
    text: str,
    embedder: BGEEmbedder,
    chunk_size: int,
    overlap_tokens: int,
) -> list[str]:
    """Разрезать текст токен-окнами, сохраняя заданное перекрытие."""
    if overlap_tokens >= chunk_size:
        raise ValueError("overlap_tokens must be smaller than chunk_size")
    token_ids = embedder.encode_text(text)
    if not token_ids:
        return []
    body_budget = _body_budget(title, embedder, chunk_size)
    if overlap_tokens >= body_budget:
        raise ValueError("overlap_tokens leaves no room for new body tokens")

    parts: list[str] = []
    start = 0
    while start < len(token_ids):
        end = min(start + body_budget, len(token_ids))
        candidate_ids = token_ids[start:end]
        candidate = embedder.decode_tokens(candidate_ids)
        # Из-за повторной tokenization уменьшаем конец, пока полный вход не поместится.
        while candidate_ids and not _fits(title, candidate, embedder, chunk_size):
            candidate_ids = candidate_ids[:-1]
            end -= 1
            candidate = embedder.decode_tokens(candidate_ids)
        if not candidate:
            raise ValueError("Unable to create a non-empty chunk within token budget")
        parts.append(candidate)
        if end >= len(token_ids):
            break
        next_start = end - overlap_tokens
        if next_start <= start:
            raise ValueError("Chunking made no progress; reduce overlap_tokens")
        start = next_start
    return parts


def split_sentences(text: str) -> list[str]:
    """Разбить английский abstract по наиболее надёжным границам предложений.

    Регулярное выражение не претендует на лингвистический парсер, но сохраняет
    сокращения лучше, чем простой ``split('.')``, и не требует сетевых ресурсов NLTK.
    """
    normalized = re.sub(r"\s+", " ", text).strip()
    if not normalized:
        return []
    parts = re.split(r"(?<=[.!?])\s+(?=[A-Z0-9])", normalized)
    return [part.strip() for part in parts if part.strip()]


def _expand_oversized_units(
    title: str,
    units: list[str],
    embedder: BGEEmbedder,
    chunk_size: int,
) -> list[str]:
    """Без потерь разрезать единицы, которые не помещаются даже поодиночке."""
    expanded: list[str] = []
    for unit in units:
        if _fits(title, unit, embedder, chunk_size):
            expanded.append(unit)
        else:
            expanded.extend(
                _fixed_token_parts(title, unit, embedder, chunk_size, overlap_tokens=0)
            )
    return expanded


def _pack_units(
    title: str,
    units: list[str],
    embedder: BGEEmbedder,
    chunk_size: int,
    separator: str,
    overlap_units: int = 0,
) -> list[str]:
    """Упаковать соседние смысловые единицы, не превышая token budget."""
    units = _expand_oversized_units(title, units, embedder, chunk_size)
    if not units:
        return []

    chunks: list[str] = []
    current: list[str] = []
    for unit in units:
        candidate = separator.join([*current, unit]).strip()
        if current and not _fits(title, candidate, embedder, chunk_size):
            chunks.append(separator.join(current).strip())
            overlap = current[-overlap_units:] if overlap_units else []
            # Если overlap вместе с новой единицей не помещается, сначала уменьшаем
            # только overlap: новая информация никогда не отбрасывается.
            while overlap and not _fits(
                title, separator.join([*overlap, unit]), embedder, chunk_size
            ):
                overlap = overlap[1:]
            current = [*overlap, unit]
        else:
            current.append(unit)
    if current:
        chunks.append(separator.join(current).strip())
    return chunks


def fixed_chunks(
    title: str,
    text: str,
    embedder: BGEEmbedder,
    chunk_size: int,
    overlap_tokens: int,
) -> list[str]:
    """Создать окна фиксированного размера в токенах BGE."""
    return _fixed_token_parts(title, text, embedder, chunk_size, overlap_tokens)


def recursive_chunks(
    title: str,
    text: str,
    embedder: BGEEmbedder,
    chunk_size: int,
    overlap_tokens: int,
) -> list[str]:
    """Сначала сохранить абзацы/предложения, затем применить token fallback."""
    paragraphs = [part.strip() for part in re.split(r"\n\s*\n", text) if part.strip()]
    units: list[str] = []
    for paragraph in paragraphs or [text]:
        if _fits(title, paragraph, embedder, chunk_size):
            units.append(paragraph)
        else:
            units.extend(split_sentences(paragraph))
    base_chunks = _pack_units(title, units, embedder, chunk_size, separator=" ")
    if overlap_tokens <= 0 or len(base_chunks) <= 1:
        return base_chunks

    overlapped = [base_chunks[0]]
    for previous, current in zip(base_chunks, base_chunks[1:], strict=False):
        tail_ids = embedder.encode_text(previous)[-overlap_tokens:]
        # Добавляем столько tail-токенов, сколько помещается; текущий чанк не режем.
        while tail_ids:
            candidate = f"{embedder.decode_tokens(tail_ids)} {current}".strip()
            if _fits(title, candidate, embedder, chunk_size):
                current = candidate
                break
            tail_ids = tail_ids[1:]
        overlapped.append(current)
    return overlapped


def sentence_chunks(
    title: str,
    text: str,
    embedder: BGEEmbedder,
    chunk_size: int,
    sentence_overlap: int,
) -> list[str]:
    """Упаковать полные предложения с overlap в одно или несколько предложений."""
    return _pack_units(
        title,
        split_sentences(text),
        embedder,
        chunk_size,
        separator=" ",
        overlap_units=sentence_overlap,
    )


def chunk_document(
    document: dict[str, str],
    strategy: ChunkStrategy,
    embedder: BGEEmbedder,
    chunk_size: int,
    overlap_tokens: int,
    sentence_overlap: int,
) -> list[Chunk]:
    """Разбить один документ выбранной стратегией и присвоить стабильные ID."""
    title = document["title"]
    text = document["text"]
    if strategy == "fixed":
        parts = fixed_chunks(title, text, embedder, chunk_size, overlap_tokens)
    elif strategy == "recursive":
        parts = recursive_chunks(title, text, embedder, chunk_size, overlap_tokens)
    elif strategy == "sentence":
        parts = sentence_chunks(title, text, embedder, chunk_size, sentence_overlap)
    else:
        raise ValueError(f"Unknown chunk strategy: {strategy}")

    chunks = []
    for index, part in enumerate(parts):
        token_count = embedder.count_tokens(build_content(title, part))
        chunks.append(
            Chunk(
                chunk_id=f"{document['doc_id']}:{strategy}:{index:04d}",
                doc_id=document["doc_id"],
                chunk_index=index,
                strategy=strategy,
                title=title,
                text=part,
                token_count=token_count,
            )
        )
    return chunks


def build_chunk_report(
    chunks: list[Chunk],
    source_documents: int,
    strategy: ChunkStrategy,
    chunk_size: int,
    overlap_tokens: int,
    sentence_overlap: int,
    embedding_model: str,
) -> ChunkingReport:
    """Проверить набор и посчитать статистику без сокрытия ошибок."""
    counts = Counter(chunk.doc_id for chunk in chunks)
    token_counts = [chunk.token_count for chunk in chunks]
    if source_documents and len(counts) != source_documents:
        raise ValueError(
            f"Only {len(counts)} of {source_documents} documents produced chunks"
        )
    if len({chunk.chunk_id for chunk in chunks}) != len(chunks):
        raise ValueError("Chunk IDs must be unique")
    if not token_counts:
        raise ValueError("Chunk dataset must not be empty")

    return ChunkingReport(
        strategy=strategy,
        source_documents=source_documents,
        chunks=len(chunks),
        documents_with_multiple_chunks=sum(count > 1 for count in counts.values()),
        mean_chunks_per_document=round(fmean(counts.values()), 3),
        min_tokens=min(token_counts),
        mean_tokens=round(fmean(token_counts), 3),
        median_tokens=round(float(median(token_counts)), 3),
        p95_tokens=round(percentile(token_counts, 95), 3),
        max_tokens=max(token_counts),
        empty_chunks=sum(not chunk.text.strip() for chunk in chunks),
        over_limit_chunks=sum(count > chunk_size for count in token_counts),
        chunk_size=chunk_size,
        overlap_tokens=overlap_tokens if strategy != "sentence" else 0,
        sentence_overlap=sentence_overlap if strategy == "sentence" else 0,
        embedding_model=embedding_model,
    )


def prepare_chunk_dataset(
    corpus_path: Path,
    output_dir: Path,
    strategy: ChunkStrategy,
    embedder: BGEEmbedder,
    chunk_size: int,
    overlap_tokens: int,
    sentence_overlap: int,
) -> tuple[Path, Path, ChunkingReport]:
    """Подготовить JSONL и манифест одной стратегии."""
    documents = list(read_corpus(corpus_path))
    chunks = [
        chunk
        for document in documents
        for chunk in chunk_document(
            document,
            strategy,
            embedder,
            chunk_size,
            overlap_tokens,
            sentence_overlap,
        )
    ]
    report = build_chunk_report(
        chunks,
        len(documents),
        strategy,
        chunk_size,
        overlap_tokens,
        sentence_overlap,
        embedder.model_name,
    )
    if report.empty_chunks or report.over_limit_chunks:
        raise ValueError(f"Invalid chunks detected: {asdict(report)}")

    output_dir.mkdir(parents=True, exist_ok=True)
    chunks_path = output_dir / f"{strategy}.jsonl"
    manifest_path = output_dir / f"{strategy}_manifest.json"
    with chunks_path.open("w", encoding="utf-8") as target:
        for chunk in chunks:
            target.write(json.dumps(asdict(chunk), ensure_ascii=False, sort_keys=True) + "\n")
    manifest_path.write_text(
        json.dumps(asdict(report), ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return chunks_path, manifest_path, report


def read_chunks(path: Path) -> list[Chunk]:
    """Прочитать подготовленный JSONL и восстановить типизированные чанки."""
    chunks: list[Chunk] = []
    with path.open(encoding="utf-8") as source:
        for line_number, line in enumerate(source, start=1):
            if not line.strip():
                continue
            try:
                chunks.append(Chunk(**json.loads(line)))
            except (TypeError, KeyError, json.JSONDecodeError) as exc:
                raise ValueError(f"Malformed chunk at line {line_number}") from exc
    return chunks


def group_chunks_by_document(chunks: list[Chunk]) -> dict[str, list[Chunk]]:
    """Сгруппировать чанки в исходном порядке для ноутбука и тестов."""
    grouped: dict[str, list[Chunk]] = defaultdict(list)
    for chunk in chunks:
        grouped[chunk.doc_id].append(chunk)
    return dict(grouped)
