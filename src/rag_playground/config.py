"""Чтение и валидация настроек проекта."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


def _load_dotenv(path: Path = Path(".env")) -> None:
    """Загрузить простые пары ``КЛЮЧ=ЗНАЧЕНИЕ`` из ``.env``"""
    if not path.exists():
        return
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        # Пустые строки, комментарии и строки без разделителя не являются
        # настройками и поэтому пропускаются.
        if not line or line.startswith("#") or "=" not in line:
            continue
        # split с maxsplit=1 сохраняет знаки "=" внутри самого значения, например в
        # токене или URL. Внешние кавычки снимаются для удобства.
        key, value = line.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip().strip("\"").strip("'"))


def _as_bool(value: str) -> bool:
    """Преобразовать текстовое логическое значение в ``bool``.

    Неизвестное значение не приводится к ``False`` . Так опечатка в настройке
    TLS будет замечена до сетевого запроса.
    """
    normalized = value.strip().lower()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off"}:
        return False
    raise ValueError(f"Expected a boolean value, got {value!r}")


def _as_positive_int(value: str, setting_name: str) -> int:
    """Преобразовать строку в целое число строго больше нуля.

    Явная проверка не позволяет запустить индексацию с нулевым batch size или
    поиск с отрицательным top-K. Имя настройки включается в текст ошибки.
    """
    try:
        parsed = int(value)
    except ValueError as exc:
        raise ValueError(f"{setting_name} must be a positive integer, got {value!r}") from exc
    if parsed <= 0:
        raise ValueError(f"{setting_name} must be a positive integer, got {value!r}")
    return parsed


@dataclass(frozen=True, slots=True)
class Settings:
    """Все настройки, необходимые текущему этапу проекта.

    ``frozen=True`` защищает конфигурацию от случайного изменения в середине
    эксперимента. ``slots=True`` запрещает случайно добавлять новые атрибуты
    с опечатками в именах.
    """

    # Базовые параметры воспроизводимости.
    data_dir: Path
    random_seed: int

    # Адреса локальных поисковых движков.
    qdrant_url: str
    elasticsearch_url: str

    # Параметры документного BM25 baseline в Elasticsearch.
    elasticsearch_index: str
    elasticsearch_request_timeout: int
    bm25_batch_size: int
    bm25_default_top_k: int
    bm25_run_top_k: int

    # Параметры бесплатной локальной BGE Small и Qdrant dense retrieval.
    dense_model_name: str
    dense_cache_dir: Path
    dense_collection: str
    dense_device: str
    dense_max_length: int
    dense_batch_size: int
    dense_upsert_batch_size: int
    dense_default_top_k: int
    dense_run_top_k: int
    dense_query_instruction: str
    qdrant_timeout: int
    rrf_k: int

    # Параметры трёх стратегий chunking и документной агрегации результатов.
    chunk_size: int
    chunk_overlap_tokens: int
    chunk_sentence_overlap: int
    chunk_collection_prefix: str
    chunk_candidate_multiplier: int
    chunk_run_top_k: int

    # Параметры GigaChat.
    gigachat_credentials: str | None
    gigachat_scope: str
    gigachat_base_url: str
    gigachat_verify_ssl_certs: bool
    gigachat_ca_bundle_file: str | None
    gigachat_embedding_model: str
    gigachat_model: str

    # Параметры пятого этапа: контекст и воспроизводимая генерация ответов.
    rag_context_documents: int
    rag_context_max_chars: int
    rag_generation_max_tokens: int
    rag_generation_batch_limit: int

    # Langfuse Cloud: ключи используются только клиентом и не сериализуются.
    langfuse_public_key: str | None
    langfuse_secret_key: str | None
    langfuse_host: str
    langfuse_environment: str

    def __post_init__(self) -> None:
        """Проверить настройки, значения которых зависят друг от друга."""
        if self.chunk_overlap_tokens >= self.chunk_size:
            raise ValueError("CHUNK_OVERLAP_TOKENS must be smaller than CHUNK_SIZE")
        if self.chunk_size > self.dense_max_length:
            raise ValueError("CHUNK_SIZE must not exceed DENSE_MAX_LENGTH")
        if not self.dense_collection or not self.chunk_collection_prefix:
            raise ValueError("Qdrant collection names must not be empty")
        if self.gigachat_model != "GigaChat-2":
            raise ValueError("GIGACHAT_MODEL must be exactly 'GigaChat-2' for stage 5")
        if self.rag_generation_batch_limit > 5:
            raise ValueError("RAG_GENERATION_BATCH_LIMIT must not exceed 5")

    @classmethod
    def from_env(cls, dotenv_path: Path = Path(".env")) -> Settings:
        """Собрать иммутабельные настройки из ``.env`` и окружения.

        Значения по умолчанию подходят для локального Docker Compose. Слеш в конце URL
        удаляется, чтобы позже к адресу можно было безопасно добавлять API-пути.
        """
        _load_dotenv(dotenv_path)
        return cls(
            data_dir=Path(os.getenv("RAG_DATA_DIR", "data")),
            random_seed=int(os.getenv("RAG_RANDOM_SEED", "42")),
            qdrant_url=os.getenv("QDRANT_URL", "http://localhost:6333").rstrip("/"),
            elasticsearch_url=os.getenv("ELASTICSEARCH_URL", "http://localhost:9200").rstrip("/"),
            elasticsearch_index=os.getenv(
                "ELASTICSEARCH_INDEX", "scifact-documents-v1"
            ).strip(),
            elasticsearch_request_timeout=_as_positive_int(
                os.getenv("ELASTICSEARCH_REQUEST_TIMEOUT", "30"),
                "ELASTICSEARCH_REQUEST_TIMEOUT",
            ),
            bm25_batch_size=_as_positive_int(
                os.getenv("BM25_BATCH_SIZE", "500"), "BM25_BATCH_SIZE"
            ),
            bm25_default_top_k=_as_positive_int(
                os.getenv("BM25_DEFAULT_TOP_K", "10"), "BM25_DEFAULT_TOP_K"
            ),
            bm25_run_top_k=_as_positive_int(
                os.getenv("BM25_RUN_TOP_K", "100"), "BM25_RUN_TOP_K"
            ),
            dense_model_name=os.getenv(
                "DENSE_MODEL_NAME", "BAAI/bge-small-en-v1.5"
            ).strip(),
            dense_cache_dir=Path(
                os.getenv("DENSE_CACHE_DIR", "data/models/fastembed")
            ),
            dense_collection=os.getenv(
                "DENSE_COLLECTION", "scifact-documents-bge-small-en-v1-5"
            ).strip(),
            # CPU — воспроизводимый вариант, не требующий отдельной видеокарты.
            dense_device=os.getenv("DENSE_DEVICE", "cpu").strip(),
            dense_max_length=_as_positive_int(
                os.getenv("DENSE_MAX_LENGTH", "512"), "DENSE_MAX_LENGTH"
            ),
            dense_batch_size=_as_positive_int(
                os.getenv("DENSE_BATCH_SIZE", "32"), "DENSE_BATCH_SIZE"
            ),
            dense_upsert_batch_size=_as_positive_int(
                os.getenv("DENSE_UPSERT_BATCH_SIZE", "128"),
                "DENSE_UPSERT_BATCH_SIZE",
            ),
            dense_default_top_k=_as_positive_int(
                os.getenv("DENSE_DEFAULT_TOP_K", "10"), "DENSE_DEFAULT_TOP_K"
            ),
            dense_run_top_k=_as_positive_int(
                os.getenv("DENSE_RUN_TOP_K", "100"), "DENSE_RUN_TOP_K"
            ),
            dense_query_instruction=os.getenv(
                "DENSE_QUERY_INSTRUCTION",
                "Represent this sentence for searching relevant passages:",
            ).strip(),
            qdrant_timeout=_as_positive_int(
                os.getenv("QDRANT_TIMEOUT", "60"), "QDRANT_TIMEOUT"
            ),
            rrf_k=_as_positive_int(os.getenv("RRF_K", "60"), "RRF_K"),
            chunk_size=_as_positive_int(
                os.getenv("CHUNK_SIZE", "256"), "CHUNK_SIZE"
            ),
            chunk_overlap_tokens=_as_positive_int(
                os.getenv("CHUNK_OVERLAP_TOKENS", "32"),
                "CHUNK_OVERLAP_TOKENS",
            ),
            chunk_sentence_overlap=_as_positive_int(
                os.getenv("CHUNK_SENTENCE_OVERLAP", "1"),
                "CHUNK_SENTENCE_OVERLAP",
            ),
            chunk_collection_prefix=os.getenv(
                "CHUNK_COLLECTION_PREFIX", "scifact-chunks-bge-small-en-v1-5"
            ).strip(),
            chunk_candidate_multiplier=_as_positive_int(
                os.getenv("CHUNK_CANDIDATE_MULTIPLIER", "5"),
                "CHUNK_CANDIDATE_MULTIPLIER",
            ),
            chunk_run_top_k=_as_positive_int(
                os.getenv("CHUNK_RUN_TOP_K", "100"), "CHUNK_RUN_TOP_K"
            ),
            gigachat_credentials=os.getenv("GIGACHAT_CREDENTIALS") or None,
            gigachat_scope=os.getenv("GIGACHAT_SCOPE", "GIGACHAT_API_PERS"),
            gigachat_base_url=os.getenv("GIGACHAT_BASE_URL", "https://api.giga.chat/v1"),
            gigachat_verify_ssl_certs=_as_bool(
                os.getenv("GIGACHAT_VERIFY_SSL_CERTS", "true")
            ),
            gigachat_ca_bundle_file=os.getenv("GIGACHAT_CA_BUNDLE_FILE") or None,
            gigachat_embedding_model=os.getenv(
                "GIGACHAT_EMBEDDING_MODEL", "EmbeddingsGigaR"
            ),
            # Для эксперимента модель фиксируется заранее и не выбирается по ответам.
            gigachat_model=os.getenv("GIGACHAT_MODEL", "GigaChat-2").strip(),
            rag_context_documents=_as_positive_int(
                os.getenv("RAG_CONTEXT_DOCUMENTS", "4"),
                "RAG_CONTEXT_DOCUMENTS",
            ),
            rag_context_max_chars=_as_positive_int(
                os.getenv("RAG_CONTEXT_MAX_CHARS", "12000"),
                "RAG_CONTEXT_MAX_CHARS",
            ),
            rag_generation_max_tokens=_as_positive_int(
                os.getenv("RAG_GENERATION_MAX_TOKENS", "512"),
                "RAG_GENERATION_MAX_TOKENS",
            ),
            rag_generation_batch_limit=_as_positive_int(
                os.getenv("RAG_GENERATION_BATCH_LIMIT", "5"),
                "RAG_GENERATION_BATCH_LIMIT",
            ),
            langfuse_public_key=os.getenv("LANGFUSE_PUBLIC_KEY") or None,
            langfuse_secret_key=os.getenv("LANGFUSE_SECRET_KEY") or None,
            langfuse_host=os.getenv(
                "LANGFUSE_HOST", "https://cloud.langfuse.com"
            ).rstrip("/"),
            langfuse_environment=os.getenv(
                "LANGFUSE_ENVIRONMENT", "portfolio"
            ).strip(),
        )

