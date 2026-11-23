"""Командный интерфейс для воспроизводимых шагов эксперимента.

"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

from rag_playground.config import Settings
from rag_playground.data import prepare_scifact
from rag_playground.generation.client import (
    GigaChatGenerator,
    require_safe_generation_batch,
)
from rag_playground.generation.context import (
    build_prepared_example,
    load_experiment_queries,
    load_prepared_examples,
    save_prepared_examples,
    select_experiment_queries,
)
from rag_playground.generation.evaluation import evaluate_answer_file, save_evaluation
from rag_playground.generation.run import (
    append_generated_answers,
    ensure_answers_do_not_exist,
    generate_examples,
)
from rag_playground.health import check_gigachat, check_local_services
from rag_playground.observability.backfill import backfill_rag_traces
from rag_playground.observability.langfuse_client import (
    check_langfuse_connection,
    create_langfuse_client,
)
from rag_playground.reporting.analysis import build_analysis_artifacts
from rag_playground.reporting.report import build_final_report
from rag_playground.retrieval.bm25 import (
    create_elasticsearch_client,
    create_index,
    get_index_info,
    index_corpus,
    search_bm25,
)
from rag_playground.retrieval.chunk_retrieval import (
    chunk_collection_name,
    index_chunk_dataset,
    search_chunks,
)
from rag_playground.retrieval.chunk_runs import run_chunk_queries, save_chunk_run
from rag_playground.retrieval.chunking import CHUNK_STRATEGIES, prepare_chunk_dataset
from rag_playground.retrieval.dense import (
    create_dense_collection,
    create_qdrant_client,
    get_dense_collection_info,
    index_dense_corpus,
    search_dense,
)
from rag_playground.retrieval.dense_runs import run_dense_queries, save_dense_run
from rag_playground.retrieval.embedding import (
    BGEEmbedder,
    analyze_corpus_lengths,
    save_length_report,
)
from rag_playground.retrieval.evaluation import evaluate_run_file
from rag_playground.retrieval.fusion import fuse_run_files
from rag_playground.retrieval.runs import read_test_queries, run_bm25_queries, save_bm25_run


def _build_parser() -> argparse.ArgumentParser:
    """Описать доступные команды и их аргументы.

    Парсер создаётся в отдельной функции, чтобы структуру CLI можно было тестировать
    без запуска отдельного процесса.
    """
    parser = argparse.ArgumentParser(prog="rag-playground")
    # required=True не даёт молча запустить CLI без выбора действия.
    subparsers = parser.add_subparsers(dest="command", required=True)

    # Выбор split ограничен реально существующими в qrels значениями. Так опечатка
    # обнаружится до сетевого запроса к Hugging Face.
    prepare = subparsers.add_parser("prepare-data", help="download and normalize SciFact")
    prepare.add_argument("--split", choices=("train", "test"), default="test")

    subparsers.add_parser("check-services", help="check local Qdrant and Elasticsearch")
    subparsers.add_parser("check-gigachat", help="authenticate and list available GigaChat models")

    # BM25 разбит на несколько команд: это позволяет пошагово
    # проверить индекс, один запрос и лишь затем запустить весь split.
    bm25_index = subparsers.add_parser(
        "bm25-index", help="create the Elasticsearch index and load the SciFact corpus"
    )
    bm25_index.add_argument(
        "--recreate",
        action="store_true",
        help="delete only the configured BM25 index before rebuilding it",
    )

    bm25_search = subparsers.add_parser(
        "bm25-search", help="run one BM25 query against the document index"
    )
    bm25_search.add_argument("query", help="English SciFact-style search query")
    bm25_search.add_argument("--top-k", type=int, default=None)

    bm25_run = subparsers.add_parser(
        "bm25-run", help="run BM25 for all queries referenced by a qrels split"
    )
    bm25_run.add_argument("--split", choices=("train", "test"), default="test")
    bm25_run.add_argument("--top-k", type=int, default=None)
    bm25_run.add_argument(
        "--output-dir", type=Path, default=Path("reports/results/runs")
    )
    bm25_run.add_argument("--run-name", default=None)

    subparsers.add_parser("bm25-info", help="show Elasticsearch and BM25 index information")

    # Dense retrieval использует бесплатную локальную BGE Small. Анализ длины
    # вынесен отдельно, чтобы до индексации было видно, сколько текстов обрежется.
    dense_analyze = subparsers.add_parser(
        "dense-analyze", help="analyze SciFact token lengths with the BGE tokenizer"
    )
    dense_analyze.add_argument(
        "--output",
        type=Path,
        default=Path("reports/results/dataset/bge_small_length_stats.json"),
    )

    dense_index = subparsers.add_parser(
        "dense-index", help="embed SciFact documents and load them into Qdrant"
    )
    dense_index.add_argument(
        "--recreate",
        action="store_true",
        help="delete only the configured dense collection before rebuilding it",
    )

    subparsers.add_parser("dense-info", help="show Qdrant dense collection information")

    dense_search = subparsers.add_parser(
        "dense-search", help="run one semantic search query against Qdrant"
    )
    dense_search.add_argument("query", help="English SciFact-style search query")
    dense_search.add_argument("--top-k", type=int, default=None)

    dense_run = subparsers.add_parser(
        "dense-run", help="run dense retrieval for all queries in a qrels split"
    )
    dense_run.add_argument("--split", choices=("train", "test"), default="test")
    dense_run.add_argument("--top-k", type=int, default=None)
    dense_run.add_argument(
        "--output-dir", type=Path, default=Path("reports/results/runs")
    )
    dense_run.add_argument("--run-name", default=None)

    hybrid_run = subparsers.add_parser(
        "hybrid-run", help="fuse BM25 and dense rankings with RRF and evaluate all runs"
    )
    hybrid_run.add_argument("--split", choices=("train", "test"), default="test")
    hybrid_run.add_argument("--top-k", type=int, default=None)
    hybrid_run.add_argument("--bm25-ranking", type=Path, default=None)
    hybrid_run.add_argument("--dense-ranking", type=Path, default=None)
    hybrid_run.add_argument("--output", type=Path, default=None)
    hybrid_run.add_argument(
        "--metrics-output",
        type=Path,
        default=Path("reports/results/metrics/document_retrieval_test.json"),
    )

    chunk_prepare = subparsers.add_parser(
        "chunk-prepare", help="prepare fixed, recursive, or sentence chunk datasets"
    )
    chunk_prepare.add_argument(
        "--strategy", choices=(*CHUNK_STRATEGIES, "all"), default="all"
    )

    chunk_index = subparsers.add_parser(
        "chunk-index", help="embed one prepared chunk dataset into Qdrant"
    )
    chunk_index.add_argument("--strategy", choices=CHUNK_STRATEGIES, required=True)
    chunk_index.add_argument("--recreate", action="store_true")

    chunk_info = subparsers.add_parser(
        "chunk-info", help="show one chunk collection and its manifest"
    )
    chunk_info.add_argument("--strategy", choices=CHUNK_STRATEGIES, required=True)

    chunk_search = subparsers.add_parser(
        "chunk-search", help="search chunks and aggregate them into unique documents"
    )
    chunk_search.add_argument("query", help="English SciFact-style search query")
    chunk_search.add_argument("--strategy", choices=CHUNK_STRATEGIES, required=True)
    chunk_search.add_argument("--top-k", type=int, default=None)

    chunk_run = subparsers.add_parser(
        "chunk-run", help="run one chunk strategy for every query in a qrels split"
    )
    chunk_run.add_argument("--strategy", choices=CHUNK_STRATEGIES, required=True)
    chunk_run.add_argument("--split", choices=("train", "test"), default="test")
    chunk_run.add_argument("--top-k", type=int, default=None)
    chunk_run.add_argument(
        "--output-dir", type=Path, default=Path("reports/results/runs")
    )

    chunk_compare = subparsers.add_parser(
        "chunk-compare", help="evaluate all chunk rankings against document baselines"
    )
    chunk_compare.add_argument("--split", choices=("train", "test"), default="test")
    chunk_compare.add_argument("--top-k", type=int, default=None)
    chunk_compare.add_argument(
        "--output",
        type=Path,
        default=Path("reports/results/metrics/chunking_test.json"),
    )

    # Генерация разделена на локальную подготовку и явный API-вызов. Это позволяет
    # проверить prompts до расходования токенов и запросить подтверждение пользователя.
    rag_prepare = subparsers.add_parser(
        "rag-prepare", help="retrieve sources and save exact prompts without generation"
    )
    rag_prepare.add_argument(
        "--plan",
        type=Path,
        default=Path("data/experiments/stage_5_queries.json"),
    )
    rag_prepare.add_argument("--query-id", action="append", default=None)
    rag_prepare.add_argument(
        "--output",
        type=Path,
        default=Path("reports/results/rag/stage_5_prepared.jsonl"),
    )

    rag_generate = subparsers.add_parser(
        "rag-generate", help="send an explicitly confirmed batch of at most five prompts"
    )
    rag_generate.add_argument(
        "--input",
        type=Path,
        default=Path("reports/results/rag/stage_5_prepared.jsonl"),
    )
    rag_generate.add_argument("--query-id", action="append", required=True)
    rag_generate.add_argument("--mode", choices=("rag", "baseline"), default="rag")
    rag_generate.add_argument(
        "--output",
        type=Path,
        default=Path("reports/results/rag/stage_5_answers.jsonl"),
    )
    rag_generate.add_argument(
        "--confirm-generation",
        action="store_true",
        help="confirm that this command may spend GigaChat generation tokens",
    )

    rag_evaluate = subparsers.add_parser(
        "rag-evaluate", help="summarize saved answers without calling GigaChat"
    )
    rag_evaluate.add_argument(
        "--input",
        type=Path,
        default=Path("reports/results/rag/stage_5_answers.jsonl"),
    )
    rag_evaluate.add_argument(
        "--output",
        type=Path,
        default=Path("reports/results/rag/stage_5_summary.json"),
    )

    subparsers.add_parser(
        "langfuse-check", help="validate Langfuse Cloud credentials without creating traces"
    )

    langfuse_backfill = subparsers.add_parser(
        "langfuse-backfill", help="import saved stage-5 RAG runs without new generation"
    )
    langfuse_backfill.add_argument(
        "--prepared",
        type=Path,
        default=Path("reports/results/rag/stage_5_prepared.jsonl"),
    )
    langfuse_backfill.add_argument(
        "--answers",
        type=Path,
        default=Path("reports/results/rag/stage_5_answers.jsonl"),
    )
    langfuse_backfill.add_argument(
        "--manual",
        type=Path,
        default=Path("data/experiments/stage_5_manual_evaluation.csv"),
    )
    langfuse_backfill.add_argument(
        "--state",
        type=Path,
        default=Path("reports/results/langfuse/backfill_state.json"),
    )

    final_report = subparsers.add_parser(
        "final-report", help="build stage-6 analysis tables and final Markdown report"
    )
    final_report.add_argument(
        "--analysis-dir",
        type=Path,
        default=Path("reports/results/analysis"),
    )
    final_report.add_argument(
        "--output",
        type=Path,
        default=Path("reports/final_report.md"),
    )
    return parser


def _bm25_client(settings: Settings):
    """Создать Elasticsearch-клиент из единой конфигурации проекта.

    URL и timeout не дублируются в каждой CLI-команде, поэтому изменение ``.env``
    одинаково влияет на индексацию, одиночный поиск и пакетный запуск.
    """
    return create_elasticsearch_client(
        settings.elasticsearch_url, settings.elasticsearch_request_timeout
    )


def _embedder(settings: Settings) -> BGEEmbedder:
    """Загрузить BGE без ненужного подключения к Qdrant."""
    return BGEEmbedder(
        settings.dense_model_name,
        settings.dense_max_length,
        settings.dense_device,
        settings.dense_cache_dir,
    )


def _dense_components(settings: Settings):
    """Создать локальную BGE-модель и Qdrant-клиент из общей конфигурации."""
    embedder = _embedder(settings)
    client = create_qdrant_client(settings.qdrant_url, settings.qdrant_timeout)
    return embedder, client


def _chunks_dir(settings: Settings) -> Path:
    """Вернуть единый каталог подготовленных chunk JSONL и manifests."""
    return settings.data_dir / "processed" / "scifact" / "chunks"


def _chunk_collection(settings: Settings, strategy: str) -> str:
    """Получить имя коллекции из общей конфигурации без дублирования формата."""
    return chunk_collection_name(
        settings.chunk_collection_prefix,
        strategy,
        settings.chunk_size,
    )


def main() -> int:
    """Выполнить выбранную команду и вернуть shell-совместимый код завершения.

    ``0`` означает успех, ``1`` — что хотя бы один health check не прошёл. Благодаря этому
    команду можно позже без изменений включить в CI-проверку.
    """
    args = _build_parser().parse_args()
    # Все команды читают настройки одинаково и не дублируют логику работы с .env.
    settings = Settings.from_env()

    if args.command == "prepare-data":
        manifest = prepare_scifact(settings.data_dir, split=args.split)
        print(json.dumps(manifest, ensure_ascii=False, indent=2))
        return 0

    if args.command == "bm25-index":
        client = _bm25_client(settings)
        # Без --recreate существующий индекс не удаляется. Повторная индексация
        # обновляет документы по стабильному ``_id`` и не создаёт дубликаты.
        created = create_index(client, settings.elasticsearch_index, recreate=args.recreate)
        if not created:
            print(
                f"Index {settings.elasticsearch_index!r} already exists; "
                "documents will be updated without deleting the index."
            )
        report = index_corpus(
            client=client,
            index_name=settings.elasticsearch_index,
            corpus_path=settings.data_dir / "raw" / "scifact" / "corpus.jsonl",
            batch_size=settings.bm25_batch_size,
        )
        print(json.dumps(asdict(report), ensure_ascii=False, indent=2))
        return 0 if report.documents_failed == 0 else 1

    if args.command == "bm25-search":
        top_k = args.top_k or settings.bm25_default_top_k
        result = search_bm25(
            _bm25_client(settings),
            settings.elasticsearch_index,
            args.query,
            top_k,
        )
        # Dataclass преобразуется в обычный JSON и для последующей программной обработки результата.
        payload = {
            "query": result.query_text,
            "latency_ms": result.latency_ms,
            "hits": [asdict(hit) for hit in result.hits],
        }
        print(json.dumps(payload, ensure_ascii=False, indent=2))
        return 0

    if args.command == "bm25-run":
        top_k = args.top_k or settings.bm25_run_top_k
        qrels_path = settings.data_dir / "raw" / "scifact" / "qrels" / f"{args.split}.tsv"
        queries = read_test_queries(
            settings.data_dir / "raw" / "scifact" / "queries.jsonl",
            qrels_path,
        )
        client = _bm25_client(settings)
        results = run_bm25_queries(
            client, settings.elasticsearch_index, queries, top_k
        )
        info = get_index_info(client, settings.elasticsearch_index)
        run_name = args.run_name or f"bm25_{args.split}_top{top_k}"
        paths = save_bm25_run(
            results,
            args.output_dir,
            run_name,
            metadata={
                "analyzer": "english",
                "elasticsearch_version": info["elasticsearch_version"],
                "index_name": settings.elasticsearch_index,
                "indexed_documents": info.get("document_count", 0),
                "split": args.split,
                "top_k": top_k,
            },
        )
        print(
            json.dumps(
                {name: str(path) for name, path in paths.items()},
                ensure_ascii=False,
                indent=2,
            )
        )
        return 0

    if args.command == "bm25-info":
        info = get_index_info(_bm25_client(settings), settings.elasticsearch_index)
        print(json.dumps(info, ensure_ascii=False, indent=2))
        return 0

    if args.command == "dense-analyze":
        embedder = _embedder(settings)
        report = analyze_corpus_lengths(
            settings.data_dir / "raw" / "scifact" / "corpus.jsonl",
            embedder,
        )
        save_length_report(report, args.output)
        print(json.dumps(asdict(report), ensure_ascii=False, indent=2))
        print(f"Saved: {args.output}")
        return 0

    if args.command == "dense-index":
        embedder, client = _dense_components(settings)
        created = create_dense_collection(
            client,
            settings.dense_collection,
            embedder.dimension,
            recreate=args.recreate,
        )
        if not created:
            print(
                f"Collection {settings.dense_collection!r} already exists; "
                "already indexed documents will be skipped."
            )
        report = index_dense_corpus(
            client=client,
            collection_name=settings.dense_collection,
            corpus_path=settings.data_dir / "raw" / "scifact" / "corpus.jsonl",
            embedder=embedder,
            embedding_batch_size=settings.dense_batch_size,
            upsert_batch_size=settings.dense_upsert_batch_size,
        )
        print(json.dumps(asdict(report), ensure_ascii=False, indent=2))
        return 0

    if args.command == "dense-info":
        # Для просмотра метаданных модель не нужна: не тратим время и память на ONNX.
        client = create_qdrant_client(settings.qdrant_url, settings.qdrant_timeout)
        info = get_dense_collection_info(client, settings.dense_collection)
        info["embedding_model"] = settings.dense_model_name
        info["max_length"] = settings.dense_max_length
        print(json.dumps(info, ensure_ascii=False, indent=2))
        return 0

    if args.command == "dense-search":
        top_k = args.top_k or settings.dense_default_top_k
        embedder, client = _dense_components(settings)
        result = search_dense(
            client,
            settings.dense_collection,
            embedder,
            args.query,
            settings.dense_query_instruction,
            top_k,
        )
        print(
            json.dumps(
                {
                    "query": result.query_text,
                    "embedding_latency_ms": result.embedding_latency_ms,
                    "search_latency_ms": result.search_latency_ms,
                    "latency_ms": result.latency_ms,
                    "hits": [asdict(hit) for hit in result.hits],
                },
                ensure_ascii=False,
                indent=2,
            )
        )
        return 0

    if args.command == "dense-run":
        top_k = args.top_k or settings.dense_run_top_k
        qrels_path = settings.data_dir / "raw" / "scifact" / "qrels" / f"{args.split}.tsv"
        queries = read_test_queries(
            settings.data_dir / "raw" / "scifact" / "queries.jsonl",
            qrels_path,
        )
        embedder, client = _dense_components(settings)
        results = run_dense_queries(
            client,
            settings.dense_collection,
            embedder,
            settings.dense_query_instruction,
            queries,
            top_k,
        )
        run_name = args.run_name or f"dense_bge_small_{args.split}_top{top_k}"
        paths = save_dense_run(
            results,
            args.output_dir,
            run_name,
            metadata={
                "collection_name": settings.dense_collection,
                "embedding_model": settings.dense_model_name,
                "max_length": settings.dense_max_length,
                "query_instruction": settings.dense_query_instruction,
                "split": args.split,
                "top_k": top_k,
            },
        )
        print(
            json.dumps(
                {name: str(path) for name, path in paths.items()},
                ensure_ascii=False,
                indent=2,
            )
        )
        return 0

    if args.command == "hybrid-run":
        top_k = args.top_k or settings.dense_run_top_k
        bm25_path = args.bm25_ranking or Path(
            f"reports/results/runs/bm25_{args.split}_top{top_k}.jsonl"
        )
        dense_path = args.dense_ranking or Path(
            f"reports/results/runs/dense_bge_small_{args.split}_top{top_k}.jsonl"
        )
        output_path = args.output or Path(
            f"reports/results/runs/hybrid_rrf_{args.split}_top{top_k}.jsonl"
        )
        manifest = fuse_run_files(
            bm25_path,
            dense_path,
            output_path,
            rrf_k=settings.rrf_k,
            top_k=top_k,
        )
        manifest_path = output_path.with_name(f"{output_path.stem}_manifest.json")
        manifest_path.write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )

        qrels_path = settings.data_dir / "raw" / "scifact" / "qrels" / f"{args.split}.tsv"
        metrics = {
            "bm25": evaluate_run_file(bm25_path, qrels_path),
            "dense_bge_small": evaluate_run_file(dense_path, qrels_path),
            "hybrid_rrf": evaluate_run_file(output_path, qrels_path),
        }
        args.metrics_output.parent.mkdir(parents=True, exist_ok=True)
        args.metrics_output.write_text(
            json.dumps(metrics, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        print(
            json.dumps(
                {
                    "hybrid_ranking": str(output_path),
                    "hybrid_manifest": str(manifest_path),
                    "metrics": str(args.metrics_output),
                    "results": metrics,
                },
                ensure_ascii=False,
                indent=2,
            )
        )
        return 0

    if args.command == "chunk-prepare":
        embedder = _embedder(settings)
        strategies = CHUNK_STRATEGIES if args.strategy == "all" else (args.strategy,)
        prepared = {}
        for strategy in strategies:
            chunks_path, manifest_path, report = prepare_chunk_dataset(
                corpus_path=settings.data_dir / "raw" / "scifact" / "corpus.jsonl",
                output_dir=_chunks_dir(settings),
                strategy=strategy,
                embedder=embedder,
                chunk_size=settings.chunk_size,
                overlap_tokens=settings.chunk_overlap_tokens,
                sentence_overlap=settings.chunk_sentence_overlap,
            )
            prepared[strategy] = {
                "chunks": str(chunks_path),
                "manifest": str(manifest_path),
                "report": asdict(report),
            }
        print(json.dumps(prepared, ensure_ascii=False, indent=2))
        return 0

    if args.command == "chunk-index":
        embedder, client = _dense_components(settings)
        collection = _chunk_collection(settings, args.strategy)
        created = create_dense_collection(
            client,
            collection,
            embedder.dimension,
            recreate=args.recreate,
        )
        if not created:
            print(
                f"Collection {collection!r} already exists; "
                "already indexed chunks will be skipped."
            )
        report = index_chunk_dataset(
            client=client,
            collection_name=collection,
            chunks_path=_chunks_dir(settings) / f"{args.strategy}.jsonl",
            embedder=embedder,
            embedding_batch_size=settings.dense_batch_size,
            upsert_batch_size=settings.dense_upsert_batch_size,
        )
        print(json.dumps(asdict(report), ensure_ascii=False, indent=2))
        return 0

    if args.command == "chunk-info":
        # Информация о коллекции читается напрямую из Qdrant без загрузки BGE.
        client = create_qdrant_client(settings.qdrant_url, settings.qdrant_timeout)
        collection = _chunk_collection(settings, args.strategy)
        info = get_dense_collection_info(client, collection)
        manifest_path = _chunks_dir(settings) / f"{args.strategy}_manifest.json"
        info["strategy"] = args.strategy
        info["manifest"] = (
            json.loads(manifest_path.read_text(encoding="utf-8"))
            if manifest_path.exists()
            else None
        )
        print(json.dumps(info, ensure_ascii=False, indent=2))
        return 0

    if args.command == "chunk-search":
        top_k = args.top_k or settings.dense_default_top_k
        embedder, client = _dense_components(settings)
        result = search_chunks(
            client=client,
            collection_name=_chunk_collection(settings, args.strategy),
            strategy=args.strategy,
            embedder=embedder,
            query_text=args.query,
            query_instruction=settings.dense_query_instruction,
            top_k=top_k,
            candidate_multiplier=settings.chunk_candidate_multiplier,
        )
        print(
            json.dumps(
                {
                    "query": result.query_text,
                    "strategy": result.strategy,
                    "candidate_chunks": result.candidate_chunks,
                    "latency_ms": result.latency_ms,
                    "documents": [asdict(hit) for hit in result.document_hits],
                },
                ensure_ascii=False,
                indent=2,
            )
        )
        return 0

    if args.command == "chunk-run":
        top_k = args.top_k or settings.chunk_run_top_k
        qrels_path = settings.data_dir / "raw" / "scifact" / "qrels" / f"{args.split}.tsv"
        queries = read_test_queries(
            settings.data_dir / "raw" / "scifact" / "queries.jsonl",
            qrels_path,
        )
        embedder, client = _dense_components(settings)
        results = run_chunk_queries(
            client=client,
            collection_name=_chunk_collection(settings, args.strategy),
            strategy=args.strategy,
            embedder=embedder,
            query_instruction=settings.dense_query_instruction,
            queries=queries,
            top_k=top_k,
            candidate_multiplier=settings.chunk_candidate_multiplier,
        )
        run_name = f"chunk_{args.strategy}_{args.split}_top{top_k}"
        paths = save_chunk_run(
            results,
            args.output_dir,
            run_name,
            metadata={
                "aggregation": "max_chunk_score",
                "candidate_multiplier": settings.chunk_candidate_multiplier,
                "chunk_size": settings.chunk_size,
                "collection_name": _chunk_collection(settings, args.strategy),
                "embedding_model": settings.dense_model_name,
                "split": args.split,
                "strategy": args.strategy,
                "top_k": top_k,
            },
        )
        print(
            json.dumps(
                {name: str(path) for name, path in paths.items()},
                ensure_ascii=False,
                indent=2,
            )
        )
        return 0

    if args.command == "chunk-compare":
        top_k = args.top_k or settings.chunk_run_top_k
        qrels_path = settings.data_dir / "raw" / "scifact" / "qrels" / f"{args.split}.tsv"
        run_dir = Path("reports/results/runs")
        ranking_paths = {
            "bm25_document": run_dir / f"bm25_{args.split}_top{top_k}.jsonl",
            "dense_document": run_dir / f"dense_bge_small_{args.split}_top{top_k}.jsonl",
            "hybrid_document": run_dir / f"hybrid_rrf_{args.split}_top{top_k}.jsonl",
            **{
                f"chunk_{strategy}": run_dir
                / f"chunk_{strategy}_{args.split}_top{top_k}.jsonl"
                for strategy in CHUNK_STRATEGIES
            },
        }
        missing = [str(path) for path in ranking_paths.values() if not path.exists()]
        if missing:
            raise FileNotFoundError(f"Missing ranking files: {missing}")
        metrics = {
            name: evaluate_run_file(path, qrels_path)
            for name, path in ranking_paths.items()
        }
        payload = {
            "chunk_parameters": {
                "chunk_size": settings.chunk_size,
                "fixed_recursive_overlap_tokens": settings.chunk_overlap_tokens,
                "sentence_overlap": settings.chunk_sentence_overlap,
            },
            "metrics": metrics,
            "note": "Configurations are fixed in advance; test metrics are not used for tuning.",
            "split": args.split,
        }
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        print(json.dumps({"output": str(args.output), **payload}, ensure_ascii=False, indent=2))
        return 0

    if args.command == "rag-prepare":
        # Эта команда полностью локальная: GigaChat и credentials не используются.
        experiment_queries = select_experiment_queries(
            load_experiment_queries(args.plan), args.query_id
        )
        embedder, client = _dense_components(settings)
        collection = _chunk_collection(settings, "recursive")
        prepared = []
        for experiment_query in experiment_queries:
            retrieval = search_chunks(
                client=client,
                collection_name=collection,
                strategy="recursive",
                embedder=embedder,
                query_text=experiment_query.query_text,
                query_instruction=settings.dense_query_instruction,
                top_k=settings.rag_context_documents,
                candidate_multiplier=settings.chunk_candidate_multiplier,
                query_id=experiment_query.query_id,
            )
            prepared.append(
                build_prepared_example(
                    experiment_query,
                    retrieval,
                    settings.rag_context_documents,
                    settings.rag_context_max_chars,
                )
            )
        save_prepared_examples(prepared, args.output)
        print(
            json.dumps(
                {
                    "output": str(args.output),
                    "examples": len(prepared),
                    "query_ids": [example.query_id for example in prepared],
                    "contexts_with_relevant_document": sum(
                        bool(example.retrieved_relevant_doc_ids) for example in prepared
                    ),
                    "generation_requests_sent": 0,
                },
                ensure_ascii=False,
                indent=2,
            )
        )
        return 0

    if args.command == "rag-generate":
        if not args.confirm_generation:
            raise ValueError(
                "Generation was not confirmed; add --confirm-generation only after approval"
            )
        prepared_by_id = {
            example.query_id: example for example in load_prepared_examples(args.input)
        }
        missing = [query_id for query_id in args.query_id if query_id not in prepared_by_id]
        if missing:
            raise ValueError(f"Prepared examples are missing query IDs: {missing}")
        selected = [prepared_by_id[query_id] for query_id in args.query_id]
        # Проверяем лимит и дубликаты до OAuth и до первого расходуемого запроса.
        require_safe_generation_batch(selected, settings.rag_generation_batch_limit)
        ensure_answers_do_not_exist(args.output, args.query_id, args.mode)
        answers = []
        with GigaChatGenerator(settings) as generator:
            for example in selected:
                # Сохраняем каждый успешный ответ немедленно: если следующий запрос
                # завершится ошибкой, уже потраченные токены и ответ не потеряются.
                one_answer = generate_examples(
                    generator,
                    [example],
                    args.mode,
                    settings.rag_generation_batch_limit,
                )
                append_generated_answers(one_answer, args.output)
                answers.extend(one_answer)
        print(
            json.dumps(
                {
                    "output": str(args.output),
                    "mode": args.mode,
                    "requests_sent": len(answers),
                    "query_ids": [answer.query_id for answer in answers],
                    "model": settings.gigachat_model,
                    "total_prompt_tokens": sum(answer.prompt_tokens for answer in answers),
                    "total_completion_tokens": sum(
                        answer.completion_tokens for answer in answers
                    ),
                },
                ensure_ascii=False,
                indent=2,
            )
        )
        return 0

    if args.command == "rag-evaluate":
        summary = evaluate_answer_file(args.input)
        save_evaluation(summary, args.output)
        print(
            json.dumps(
                {"output": str(args.output), **summary},
                ensure_ascii=False,
                indent=2,
            )
        )
        return 0

    if args.command == "langfuse-check":
        # auth_check не создаёт trace и не расходует generation tokens.
        result = check_langfuse_connection(settings)
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return 0 if result["ok"] else 1

    if args.command == "langfuse-backfill":
        # Импортируются только уже сохранённые prompts/answers; GigaChat не вызывается.
        client = create_langfuse_client(settings)
        if not client.auth_check():
            raise ConnectionError("Langfuse authentication failed")
        report = backfill_rag_traces(
            client,
            args.prepared,
            args.answers,
            args.manual,
            args.state,
        )
        print(json.dumps(asdict(report), ensure_ascii=False, indent=2))
        return 0

    if args.command == "final-report":
        paths = build_analysis_artifacts(
            retrieval_metrics_path=Path(
                "reports/results/metrics/chunking_test.json"
            ),
            prepared_path=Path("reports/results/rag/stage_5_prepared.jsonl"),
            answers_path=Path("reports/results/rag/stage_5_answers.jsonl"),
            rag_summary_path=Path("reports/results/rag/stage_5_summary.json"),
            manual_path=Path("data/experiments/stage_5_manual_evaluation.csv"),
            output_dir=args.analysis_dir,
        )
        build_final_report(args.analysis_dir, args.output)
        print(
            json.dumps(
                {
                    **{name: str(path) for name, path in paths.items()},
                    "final_report": str(args.output),
                    "generation_requests_sent": 0,
                },
                ensure_ascii=False,
                indent=2,
            )
        )
        return 0

    if args.command == "check-services":
        results = check_local_services(settings)
    else:
        results = [check_gigachat(settings)]

    # Печатаем каждый сервис отдельно: так видно, что именно не готово, даже если
    # второй сервис уже доступен.
    for result in results:
        marker = "OK" if result.ok else "FAIL"
        print(f"[{marker}] {result.name}: {result.detail}")
    return 0 if all(result.ok for result in results) else 1


if __name__ == "__main__":
    # Ветка нужна для запуска модуля через ``python -m``. При обычном импорте она
    # не выполняется, поэтому тесты могут безопасно импортировать ``main``.
    raise SystemExit(main())

