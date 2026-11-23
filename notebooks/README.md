# Notebooks

Notebooks are optional exploration layers. Reusable loading, indexing, retrieval, and evaluation
logic belongs in `src/rag_playground` so every reported experiment can also run from the CLI.

- `01_stage_1_demo.ipynb` — полная демонстрация первого этапа: настройки, SciFact,
  примеры данных, проверка целостности, health checks, Pytest, Ruff и Docker Compose.
- `02_bm25_demo.ipynb` — демонстрация документного индекса Elasticsearch, BM25-поиска,
  batch run по 300 test-запросам, сохранённых артефактов и unit/integration-тестов.
- `03_dense_hybrid_demo.ipynb` — анализ длины корпуса tokenizer'ом BGE, Qdrant dense retrieval,
  объединение BM25+dense через RRF и единое сравнение Recall, Precision, MRR и nDCG.
- `04_chunking_demo.ipynb` — три token-safe стратегии chunking, статистика подготовленных наборов,
  возобновляемая индексация, поиск по чанкам с агрегацией документов и сравнение с baseline.
- `05_rag_demo.ipynb` — подготовленные recursive-контексты, exact prompts, восемь сохранённых
  ответов GigaChat-2, token usage, проверка цитат и ручной анализ ошибок без новых API-вызовов.
- `06_final_analysis.ipynb` — Langfuse Cloud backfill, итоговые retrieval/RAG-графики,
  классификация ошибок и полный сравнительный отчёт без новых генераций.

