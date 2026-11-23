# RAG Playground with SciFact and GigaChat

Учебный исследовательский стенд для сравнения BM25, dense и hybrid retrieval, экспериментов
с chunking и генерации ответов по найденному контексту. Релевантность SciFact размечена на уровне
документов, поэтому базовый retrieval и chunking оцениваются раздельно.

## Статус

Реализованы все 6 этапов технического задания:

- воспроизводимый Python-проект и конфигурация через окружение;
- локальные Qdrant и Elasticsearch в Docker Compose;
- загрузка SciFact и неизменённого qrels в стабильную локальную схему;
- проверки локальных сервисов и доступных аккаунту моделей GigaChat;
- unit-тесты нормализации данных и конфигурации.
- документный индекс Elasticsearch для всех 5 183 документов SciFact;
- BM25 baseline по полю `content`, объединяющему заголовок и текст;
- одиночный и пакетный поиск с замером latency;
- сохранение ranking, сводки по запросам и манифеста запуска;
- unit- и integration-тесты BM25 на реальном Elasticsearch.
- анализ длины всех документов tokenizer'ом BGE Small;
- бесплатные локальные embeddings `BAAI/bge-small-en-v1.5` через FastEmbed/ONNX;
- возобновляемая индексация в Qdrant и dense retrieval;
- hybrid retrieval через Reciprocal Rank Fusion;
- единые Recall, Precision, MRR и nDCG для BM25, dense и hybrid.
- три стратегии chunking: fixed-size, recursive и sentence-based;
- строгий лимит 256 токенов BGE для полного входа `заголовок + чанк`;
- отдельные возобновляемые Qdrant-коллекции для каждой стратегии;
- поиск по чанкам с агрегацией уникальных документов по максимальному chunk score;
- пакетные прогоны, chunk evidence и единое сравнение с документными baseline.
- воспроизводимый набор из восьми SciFact-примеров для RAG;
- сбор ограниченного контекста из recursive-чанков и строгий prompt с цитатами;
- генерация через `GigaChat-2` с hard limit не более пяти запросов за запуск;
- сохранение exact prompt, ответа, источников, latency и token usage;
- автоматическая и ручная оценка groundedness, полноты и качества цитат.
- восемь исторических RAG traces в Langfuse Cloud;
- наблюдения retrieval, context building, generation и evaluation;
- шесть детерминированных scores для каждого ответа без LLM-as-a-judge;
- защита секретов, стабильные trace ID и идемпотентный backfill;
- итоговые таблицы, три графика, анализ ошибок и Markdown-отчёт.

Проект завершён. Возможное продолжение — hybrid document retrieval с выбором recursive-чанков.

## Быстрый старт

Требования: Python 3.11–3.13 и Docker с Compose.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[dev]'
cp .env.example .env
```

Запуск локальных поисковых сервисов:

```bash
docker compose up -d
rag-playground check-services
```

Загрузка итогового test split и qrels:

```bash
rag-playground prepare-data --split test
```

Команда создаёт:

- `data/raw/scifact/corpus.jsonl` с полями `doc_id`, `title`, `text`;
- `data/raw/scifact/queries.jsonl` с полями `query_id`, `text`;
- `data/raw/scifact/qrels/test.tsv` с полями `query_id`, `doc_id`, `score`;
- `data/raw/scifact/manifest.json` с источниками и количеством записей.

Для пилотных прогонов train split загружается той же командой с `--split train`. Корпус и полный
набор запросов общие; qrels сохраняются отдельно и не переписываются.

## BM25 в Elasticsearch

Индекс создаётся с одним shard, без replica и с `english` analyzer. Elasticsearch
использует стандартные параметры BM25: на этапе baseline они не подгоняются по test qrels.

```bash
rag-playground bm25-index
rag-playground bm25-info
rag-playground bm25-search "Vitamin D supplementation improves bone mineral density" --top-k 5
rag-playground bm25-run --split test --top-k 100
```

`bm25-index` без флага не удаляет существующий индекс. Для явной полной
пересборки нужен `bm25-index --recreate`; удаляется только индекс с точным именем из
`ELASTICSEARCH_INDEX`.

Пакетный запуск пишет в `reports/results/runs/`:

- `bm25_test_top100.jsonl` — по одной строке на найденный документ;
- `bm25_test_top100_queries.jsonl` — текст запроса, latency и число hits;
- `bm25_test_top100_manifest.json` — версия Elasticsearch, имя индекса и параметры запуска.

## GigaChat

Заполните `GIGACHAT_CREDENTIALS` и при необходимости `GIGACHAT_CA_BUNDLE_FILE` в `.env`, затем:

```bash
rag-playground check-gigachat
```

Проверка TLS включена по умолчанию. Команда аутентифицируется, запрашивает `GET /models` и печатает
только идентификаторы доступных моделей — ключи и токены в вывод не попадают. Используемая во всех
экспериментах embedding-модель фиксируется параметром `GIGACHAT_EMBEDDING_MODEL`.

GigaChat не используется для dense retrieval третьего этапа: embeddings строятся
бесплатно и локально моделью BGE Small. GigaChat оставлен для позднейшей генерации RAG-ответов.

## Dense и hybrid retrieval

Первый запуск скачивает ONNX-модель около 67 МБ. Затем она работает из локального
кэша `data/models/fastembed` без API-ключа. Перед индексацией проводится анализ длины:

```bash
rag-playground dense-analyze
rag-playground dense-index
rag-playground dense-info
rag-playground dense-search "Vitamin D improves bone health" --top-k 5
rag-playground dense-run --split test --top-k 100
rag-playground hybrid-run --split test --top-k 100
```

`dense-index` возобновляется: готовые `doc_id` пропускаются. Полная пересборка возможна
только с явным `--recreate`. Документы длиннее 512 токенов помечаются в payload полем
`truncated`; заголовок всегда стоит первым.

Контрольный прогон показал: 455 из 5 183 документов (8,779%) превышают лимит.
Проблема полностью решается chunking'ом на четвёртом этапе.

Итоговые метрики test split сохраняются в
`reports/results/metrics/document_retrieval_test.json`.

## Chunking и поиск по фрагментам

Все стратегии используют один и тот же заранее зафиксированный лимит 256 токенов. Для fixed и
recursive применяется перекрытие до 32 токенов, для sentence-based — до одного предложения.
Заголовок добавляется к каждому фрагменту и учитывается в лимите, поэтому BGE не обрезает чанки.

```bash
rag-playground chunk-prepare --strategy all

rag-playground chunk-index --strategy fixed
rag-playground chunk-index --strategy recursive
rag-playground chunk-index --strategy sentence

rag-playground chunk-info --strategy recursive
rag-playground chunk-search "Vitamin D improves bone health" --strategy recursive --top-k 5

rag-playground chunk-run --strategy fixed --split test --top-k 100
rag-playground chunk-run --strategy recursive --split test --top-k 100
rag-playground chunk-run --strategy sentence --split test --top-k 100
rag-playground chunk-compare --split test --top-k 100
```

`chunk-index` без `--recreate` пропускает уже сохранённые `chunk_id`. Полная пересборка удаляет
только коллекцию выбранной стратегии и требует явного флага `--recreate`.

Подготовленные наборы содержат 9 902 fixed, 9 880 recursive и 10 489 sentence-чанков. Во всех
трёх наборах нет пустых фрагментов и превышений лимита. Выдача сначала получает кандидатов-чанки,
а затем оставляет один лучший чанк для каждого документа; исходные chunk hits сохраняются отдельно
как evidence.

Контрольное сравнение на test split не использовалось для настройки параметров. Среди трёх
заранее зафиксированных вариантов recursive показал лучший nDCG@10: 0,7249. Документный hybrid
остался лучшим по Recall@10: 0,8612 против 0,8340 у recursive. Полный результат находится в
`reports/results/metrics/chunking_test.json`.

## RAG с GigaChat-2

Пятый этап разделён на локальную подготовку и явную генерацию. Сначала retrieval сохраняет точные
prompts без обращения к GigaChat:

```bash
rag-playground rag-prepare
```

Перед генерацией можно проверить `reports/results/rag/stage_5_prepared.jsonl`. API-команда требует
явного `--confirm-generation`, конкретных query ID и физически отклоняет batch больше пяти запросов:

```bash
rag-playground rag-generate \
  --query-id 1271 --query-id 1089 --query-id 386 --query-id 295 \
  --mode rag --confirm-generation

rag-playground rag-evaluate
```

В контрольном эксперименте восемь ответов были получены двумя отдельно подтверждёнными batches по
четыре запроса. Всего использовано 9 602 prompt-токена и 1 008 completion-токенов; средняя latency
составила 1 240 мс. Все восемь ответов содержат ссылки, ссылок на несуществующие номера нет. В семи
контекстах из восьми присутствовал документ из qrels. Baseline без контекста реализован как режим
`--mode baseline`, но не запускался, чтобы не расходовать дополнительные токены.

Credentials читаются только из `.env`, не включаются в prompts, JSONL, notebook или CLI-вывод.

## Langfuse Cloud и итоговый отчёт

Шестой этап использует Langfuse SDK 4 и актуальную OpenTelemetry-трассировку. Он импортирует только
уже сохранённые результаты пятого этапа и не вызывает GigaChat:

```bash
rag-playground langfuse-check
rag-playground langfuse-backfill
rag-playground final-report
```

Для каждого из восьми запросов создана trace с root chain и четырьмя дочерними observations:
retrieval, context building, generation и evaluation. К trace прикреплены scores correctness,
groundedness, completeness, citation quality, retrieval relevance и citation validity.

Импорт занял приблизительно 96 units, или 0,192% месячного лимита Hobby в 50 000 units. Повторный
запуск пропускает все восемь trace ID: дополнительные observations и units не создаются. История
доступна в Cloud 30 дней в рамках бесплатного плана.

Итоговые артефакты находятся в `reports/results/analysis/`, а полный вывод проекта — в
`reports/final_report.md`.

## Структура

```text
src/rag_playground/   переиспользуемый код и CLI
tests/                быстрые unit-тесты
data/raw/             исходные нормализованные данные, не коммитятся
data/processed/       чанки и промежуточные наборы, не коммитятся
reports/results/      таблицы метрик, не коммитятся
notebooks/            исследовательские ноутбуки без дублирования основной логики
compose.yaml          локальные Qdrant и Elasticsearch
```

## Демонстрационные ноутбуки

Ноутбук `notebooks/01_stage_1_demo.ipynb` пошагово показывает весь первый этап:

- безопасный вывод конфигурации;
- загрузку и нормализацию SciFact;
- примеры документа, запроса и qrel;
- проверку целостности ссылок;
- health checks Qdrant, Elasticsearch и GigaChat;
- результаты Pytest, Ruff и валидации Docker Compose.

После установки dev-зависимостей его можно открыть в JupyterLab:

```bash
jupyter lab notebooks/01_stage_1_demo.ipynb
```

В файле уже сохранены выводы контрольного запуска. При повторном запуске SciFact не
скачивается заново, если локальный манифест уже существует.

Ноутбук `notebooks/02_bm25_demo.ipynb` показывает второй этап: mapping и count индекса,
одиночный поиск, пакетный test run, артефакты и все проверки:

```bash
jupyter lab notebooks/02_bm25_demo.ipynb
```

Ноутбук `notebooks/03_dense_hybrid_demo.ipynb` показывает анализ длины SciFact, состояние
Qdrant, dense-поиск, RRF и сравнительные метрики трёх методов.

Ноутбук `notebooks/04_chunking_demo.ipynb` показывает подготовку трёх наборов чанков, проверку
лимитов, состояние Qdrant-коллекций, поиск с лучшим фрагментом и итоговое сравнение retrieval.

Ноутбук `notebooks/05_rag_demo.ipynb` показывает восемь сохранённых prompts и ответов `GigaChat-2`,
token usage, автоматическую проверку цитат и ручной разбор ошибок. Он не отправляет новые запросы.

Ноутбук `notebooks/06_final_analysis.ipynb` объединяет все этапы, показывает статус восьми Langfuse
traces, строит retrieval/RAG-графики и выводит итоговый отчёт без новых генераций.

## План реализации

1. Данные, конфигурация и локальные сервисы — готово.
2. Elasticsearch: индекс целых документов, BM25 и сохранение ranking — готово.
3. BGE Small embeddings, Qdrant dense retrieval, RRF и документные метрики — готово.
4. Fixed-size, recursive и sentence-based chunking с учётом токенов — готово.
5. Генерация и оценка RAG-ответов на 5–10 примерах — готово.
6. Langfuse, анализ ошибок и итоговый сравнительный отчёт — готово.

## Источники

- [SciFact corpus and queries](https://huggingface.co/datasets/BeIR/scifact)
- [SciFact qrels](https://huggingface.co/datasets/BeIR/scifact-qrels)
- [GigaChat Python SDK](https://developers.sber.ru/docs/ru/gigachain/tools/python/gigachat)
- [Qdrant local installation](https://qdrant.tech/documentation/operations/installation/)
- [Elasticsearch Docker setup](https://www.elastic.co/docs/deploy-manage/deploy/self-managed/install-elasticsearch-docker-basic)
- [Langfuse Python SDK](https://python.reference.langfuse.com/langfuse)
- [Langfuse Cloud pricing](https://langfuse.com/pricing)

