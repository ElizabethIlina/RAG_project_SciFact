"""Строгие prompts и проверка ссылок на переданные источники."""

from __future__ import annotations

import re

from rag_playground.generation.models import RAGSource

_CITATION_PATTERN = re.compile(r"\[(\d+)]")


def format_sources(sources: tuple[RAGSource, ...]) -> str:
    """Собрать читаемый контекст с однозначными номерами источников."""
    if not sources:
        raise ValueError("RAG prompt requires at least one source")
    return "\n\n".join(
        (
            f"[{source.source_number}] doc_id={source.doc_id}; "
            f"chunk_id={source.chunk_id}\n"
            f"TITLE: {source.title}\n"
            f"TEXT: {source.text}"
        )
        for source in sources
    )


def build_rag_prompt(query_text: str, sources: tuple[RAGSource, ...]) -> str:
    """Попросить проверить утверждение только по найденным научным фрагментам."""
    claim = query_text.strip()
    if not claim:
        raise ValueError("Query text must not be empty")
    context = format_sources(sources)
    return f"""Ты — аккуратный научный ассистент. Проверь утверждение, используя ТОЛЬКО
предоставленные ниже источники. Не используй сведения из памяти и не выдумывай факты.

Правила ответа:
1. Пиши по-русски и кратко.
2. Первая строка должна иметь один из видов:
   ВЫВОД: ПОДТВЕРЖДАЕТСЯ
   ВЫВОД: ОПРОВЕРГАЕТСЯ
   ВЫВОД: НЕДОСТАТОЧНО ДАННЫХ
3. Затем дай объяснение и поставь ссылки [1], [2] только после утверждений,
   которые действительно поддерживаются соответствующим источником.
4. Не ссылайся на номера, которых нет в контексте.
5. В конце напиши: ИСТОЧНИКИ: и перечисли использованные номера.

НАУЧНОЕ УТВЕРЖДЕНИЕ:
{claim}

ИСТОЧНИКИ:
{context}
""".strip()


def build_baseline_prompt(query_text: str) -> str:
    """Собрать контрольный prompt без retrieval-контекста."""
    claim = query_text.strip()
    if not claim:
        raise ValueError("Query text must not be empty")
    return f"""Ты — научный ассистент. Кратко оцени следующее утверждение по-русски.
Начни с ПОДТВЕРЖДАЕТСЯ, ОПРОВЕРГАЕТСЯ или НЕДОСТАТОЧНО ДАННЫХ, затем объясни вывод.
Не придумывай библиографические ссылки.

УТВЕРЖДЕНИЕ:
{claim}
""".strip()


def extract_citations(answer: str) -> tuple[int, ...]:
    """Извлечь уникальные номера ссылок в порядке первого появления."""
    ordered: list[int] = []
    seen: set[int] = set()
    for raw_number in _CITATION_PATTERN.findall(answer):
        number = int(raw_number)
        if number not in seen:
            ordered.append(number)
            seen.add(number)
    return tuple(ordered)


def invalid_citations(answer: str, source_count: int) -> tuple[int, ...]:
    """Вернуть ссылки на источники, которых модель не получала."""
    if source_count < 0:
        raise ValueError("source_count must not be negative")
    return tuple(
        number for number in extract_citations(answer) if not 1 <= number <= source_count
    )
