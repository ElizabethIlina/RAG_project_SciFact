"""Генерация итогового Markdown-отчёта без внешней модели."""

from __future__ import annotations

import csv
import json
from pathlib import Path


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8") as source:
        return list(csv.DictReader(source))


def build_final_report(
    analysis_dir: Path,
    output_path: Path,
) -> None:
    """Собрать проверяемый отчёт из готовых чисел без ручного копирования."""
    summary = json.loads(
        (analysis_dir / "stage_6_summary.json").read_text(encoding="utf-8")
    )
    retrieval = _read_csv(analysis_dir / "retrieval_comparison.csv")
    rag_rows = _read_csv(analysis_dir / "rag_evaluation.csv")
    retrieval_table = "\n".join(
        "| {method} | {recall_at_1:.4f} | {recall_at_10:.4f} | "
        "{mrr_at_10:.4f} | {ndcg_at_10:.4f} | {recall_at_100:.4f} |".format(
            method=row["method"],
            **{
                key: float(row[key])
                for key in (
                    "recall_at_1",
                    "recall_at_10",
                    "mrr_at_10",
                    "ndcg_at_10",
                    "recall_at_100",
                )
            },
        )
        for row in retrieval
    )
    rag_table = "\n".join(
        f"| {row['query_id']} | {row['relevant_in_context']} | "
        f"{row['correctness_0_2']} | {row['groundedness_0_2']} | "
        f"{row['citation_quality_0_2']} | {row['error_type']} |"
        for row in rag_rows
    )
    manual = summary["manual_mean_scores"]
    content = f"""# Итоговый отчёт: RAG Playground

## Цель

Проект сравнивает документный и chunk-level retrieval на SciFact, а затем использует найденные
научные фрагменты как проверяемый контекст для `GigaChat-2`.

## Данные и воспроизводимость

- Корпус: 5 183 документа SciFact.
- Test split: 300 запросов с неизменённым document-level qrels.
- Dense-модель: локальная `BAAI/bge-small-en-v1.5`.
- Генерация: 8 заранее выбранных запросов, два подтверждённых batch по 4.
- Параметры chunking и retrieval зафиксированы до просмотра test-метрик.

## Сравнение retrieval

| Метод | Recall@1 | Recall@10 | MRR@10 | nDCG@10 | Recall@100 |
|---|---:|---:|---:|---:|---:|
{retrieval_table}

Лучший общий nDCG@10: **{summary['retrieval']['best_ndcg_at_10']['method']}**
({summary['retrieval']['best_ndcg_at_10']['value']:.4f}). Лучший chunk-вариант:
**{summary['retrieval']['best_chunk_ndcg_at_10']['method']}**
({summary['retrieval']['best_chunk_ndcg_at_10']['value']:.4f}).

Hybrid обеспечивает наибольшую полноту, а recursive chunking хорошо ранжирует релевантный evidence
в верхней части выдачи. Практическое продолжение — hybrid document retrieval с последующим выбором
recursive-чанков внутри найденных документов.

## RAG с GigaChat-2

- Ответов: {summary['rag']['answers']}.
- Контекстов с документом из qrels: {summary['rag']['rag_contexts_with_relevant_document']} из 8.
- Ответов с цитатами: {summary['rag']['answers_with_citations']} из 8.
- Ответов с несуществующими номерами: {summary['rag']['answers_with_invalid_citations']}.
- Prompt tokens: {summary['rag']['total_prompt_tokens']}.
- Completion tokens: {summary['rag']['total_completion_tokens']}.
- Средняя generation latency: {summary['rag']['mean_generation_latency_ms']:.1f} мс.

Средние ручные оценки по шкале 0–2:

- correctness: {manual['correctness_0_2']:.3f};
- groundedness: {manual['groundedness_0_2']:.3f};
- completeness: {manual['completeness_0_2']:.3f};
- citation quality: {manual['citation_quality_0_2']:.3f}.

| Query | Relevant in context | Correctness | Groundedness | Citation quality | Error type |
|---|---|---:|---:|---:|---|
{rag_table}

## Анализ ошибок

- `format_noncompliance`: корректный ответ не завершён требуемой строкой `ИСТОЧНИКИ`.
- `generation_overclaim`: дополнительный источник интерпретирован сильнее, чем позволяет его текст.
- `retrieval_qrels_miss`: документ из qrels не попал в четыре переданных источника.
- Невалидных номеров цитат не обнаружено.

Query `295` показывает важность раздельной диагностики: документ qrels был на девятом месте, однако
другой top-1 фрагмент семантически поддерживал утверждение. Это может быть как retrieval miss по
официальной разметке, так и проявление неполноты qrels.

## Langfuse Cloud Hobby

В Langfuse импортируются только существующие восемь запусков; новые генерации не выполняются.
Оценка объёма: {summary['langfuse_hobby']['estimated_import_units']} units — около
{summary['langfuse_hobby']['estimated_allowance_percent']:.4f}% от бесплатных 50 000 units в месяц.

## Ограничения

- SciFact qrels размечает релевантность документов, но не является эталоном готового ответа.
- Набор генерации мал: 8 примеров.
- Baseline без контекста не запускался для экономии токенов, поэтому причинный эффект RAG
  не измерен.
- Ручные оценки субъективны и выставлены одним оценщиком.

## Итог

Для поиска с максимальной полнотой лучше всего показал себя hybrid. Для формирования компактного
научного контекста выбран recursive chunking. GigaChat обычно следует evidence и корректно
использует номера источников, но требует дополнительного контроля полноты формата и чрезмерных
обобщений.
"""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(content, encoding="utf-8")
