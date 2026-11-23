"""Загрузка и нормализация BEIR SciFact без изменения qrels.

Внешний датасет сразу приводится к внутренней схеме проекта. Остальным
этапам не нужно знать, как именно Hugging Face называет поля идентификаторов.
Разметка релевантности сохраняется как есть: мы меняем только имена колонок.
"""

from __future__ import annotations

import csv
import json
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any


def normalize_document(row: Mapping[str, Any]) -> dict[str, str]:
    """Привести один документ BEIR к внутренней схеме корпуса.

    Идентификатор приводится к строке, чтобы Elasticsearch, Qdrant и таблицы
    метрик впоследствии сравнивали ``doc_id`` одинаково. Заголовок может быть
    пустым, но без идентификатора или текста документ индексировать нельзя.
    """
    # Поддерживаем оба распространённых имени поля ID, хотя текущий SciFact
    # использует ``_id``. Это делает границу импорта немного устойчивее.
    doc_id = str(row.get("_id") or row.get("id") or "").strip()
    title = str(row.get("title") or "").strip()
    text = str(row.get("text") or "").strip()
    if not doc_id or not text:
        raise ValueError("Every corpus row must have a non-empty id and text")
    return {"doc_id": doc_id, "title": title, "text": text}


def normalize_query(row: Mapping[str, Any]) -> dict[str, str]:
    """Привести один запрос BEIR к паре ``query_id`` и ``text``.

    Поле ``title`` оставлено как резервный источник текста для совместимости
    с другими BEIR-наборами. В SciFact основной текст находится в ``text``.
    """
    query_id = str(row.get("_id") or row.get("id") or "").strip()
    text = str(row.get("text") or row.get("title") or "").strip()
    if not query_id or not text:
        raise ValueError("Every query row must have a non-empty id and text")
    return {"query_id": query_id, "text": text}


def normalize_qrel(row: Mapping[str, Any]) -> tuple[str, str, int]:
    """Преобразовать одну строку qrels без изменения оценки релевантности.

    Результат — тройка ``query_id``, ``doc_id``, ``score``. Здесь нет фильтрации,
    дедупликации или изменения score: эталонная разметка должна остаться нетронутой.
    """
    query_id = str(
        row.get("query-id") or row.get("query_id") or row.get("query-id") or ""
    ).strip()
    doc_id = str(
        row.get("corpus-id") or row.get("corpus_id") or row.get("doc_id") or ""
    ).strip()
    score = int(row.get("score", 0))
    if not query_id or not doc_id:
        raise ValueError(f"Malformed qrel row: {dict(row)!r}")
    return query_id, doc_id, score


def _write_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> int:
    """Записать поток словарей в JSON Lines и вернуть число строк.

    JSONL позволяет позже читать корпус построчно, не загружая весь набор в память.
    ``ensure_ascii=False`` сохраняет Unicode-текст читаемым, а сортировка ключей делает
    файл детерминированным.
    """
  
    path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with path.open("w", encoding="utf-8") as output:
        for row in rows:
            output.write(json.dumps(dict(row), ensure_ascii=False, sort_keys=True) + "\n")
            count += 1
    return count


def _write_qrels(path: Path, rows: Iterable[tuple[str, str, int]]) -> int:
    """Записать qrels в TSV и вернуть количество связей запрос–документ.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with path.open("w", encoding="utf-8", newline="") as output:
        writer = csv.writer(output, delimiter="\t", lineterminator="\n")
        writer.writerow(["query_id", "doc_id", "score"])
        for row in rows:
            writer.writerow(row)
            count += 1
    return count


def prepare_scifact(output_dir: Path, split: str = "test") -> dict[str, int | str]:
    """Скачать SciFact с Hugging Face и сохранить его в стабильной локальной схеме.

    Корпус и запросы живут в ``BeIR/scifact``, а qrels — в отдельном
    ``BeIR/scifact-qrels``. В конце создаётся манифест: он фиксирует источники,
    split и фактическое количество записей для проверки воспроизводимости.
    """
    
    try:
        from datasets import load_dataset
    except ImportError as exc:  # pragma: no cover - user-facing environment error
        raise RuntimeError("Install the project dependencies before downloading SciFact") from exc

    # У корпуса и запросов свои конфигурации и split-ы. Для qrels выбираем train/test,
    # переданный пользователем.
    corpus = load_dataset("BeIR/scifact", "corpus", split="corpus")
    queries = load_dataset("BeIR/scifact", "queries", split="queries")
    qrels = load_dataset("BeIR/scifact-qrels", split=split)

    # map создаёт ленивый итератор: записи нормализуются по мере записи, а не
    # копятся целиком в новый список в памяти.
    target = output_dir / "raw" / "scifact"
    corpus_count = _write_jsonl(target / "corpus.jsonl", map(normalize_document, corpus))
    query_count = _write_jsonl(target / "queries.jsonl", map(normalize_query, queries))
    qrel_count = _write_qrels(target / "qrels" / f"{split}.tsv", map(normalize_qrel, qrels))

    # Манифест не дублирует сами данные. Он хранит минимальную информацию, по
    # которой можно быстро понять, что именно было скачано.
    manifest: dict[str, int | str] = {
        "dataset": "BeIR/scifact",
        "qrels_dataset": "BeIR/scifact-qrels",
        "split": split,
        "corpus_documents": corpus_count,
        "queries": query_count,
        "qrels": qrel_count,
    }
    (target / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return manifest

