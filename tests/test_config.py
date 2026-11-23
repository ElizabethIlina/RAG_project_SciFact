"""Тесты приоритетов и валидации конфигурации."""

from pathlib import Path

import pytest

from rag_playground.config import Settings, _as_bool, _as_positive_int


def test_settings_load_dotenv_without_overwriting_environment(tmp_path, monkeypatch):
    """Переменная процесса должна иметь приоритет над тем же ключом из ``.env``.

    ``tmp_path`` изолирует тест от настоящего .env разработчика, а ``monkeypatch`` гарантирует,
    что после теста изменённое окружение будет восстановлено.
    """
    # Искусственный .env задаёт и URL, и seed. Затем URL также задаётся на уровне
    # процесса, чтобы проверить конфликт источников.
    dotenv = tmp_path / ".env"
    dotenv.write_text(
        "QDRANT_URL=http://from-file:6333\nRAG_RANDOM_SEED=7\n",
        encoding="utf-8",
    )
    monkeypatch.setenv("QDRANT_URL", "http://from-process:6333")

    settings = Settings.from_env(Path(dotenv))

    assert settings.qdrant_url == "http://from-process:6333"
    assert settings.random_seed == 7


@pytest.mark.parametrize("value", ["maybe", "enabled", "2"])
def test_invalid_boolean_is_rejected(value):
    """Неоднозначные логические значения нельзя молча трактовать как ``False``.

    Параметризация запускает одинаковую проверку для нескольких типичных ошибочных
    значений и показывает их как отдельные тест-кейсы.
    """
    with pytest.raises(ValueError):
        _as_bool(value)


@pytest.mark.parametrize("value", ["0", "-1", "not-a-number"])
def test_invalid_positive_integer_is_rejected(value):
    """Batch size, timeout и top-K должны быть целыми числами больше нуля."""
    with pytest.raises(ValueError, match="BM25_BATCH_SIZE"):
        _as_positive_int(value, "BM25_BATCH_SIZE")


def test_dense_defaults_use_local_bge_small(tmp_path, monkeypatch):
    """По умолчанию embeddings не должны зависеть от платного внешнего API."""
    # Пустой файл изолирует тест от пользовательского .env в корне проекта.
    dotenv = tmp_path / ".env"
    dotenv.write_text("", encoding="utf-8")
    monkeypatch.delenv("DENSE_MODEL_NAME", raising=False)
    monkeypatch.delenv("DENSE_COLLECTION", raising=False)

    settings = Settings.from_env(dotenv)

    assert settings.dense_model_name == "BAAI/bge-small-en-v1.5"
    assert settings.dense_cache_dir == Path("data/models/fastembed")
    assert settings.dense_max_length == 512
    assert settings.dense_device == "cpu"
    assert settings.chunk_size == 256
    assert settings.chunk_overlap_tokens == 32
    assert settings.gigachat_model == "GigaChat-2"
    assert settings.rag_generation_batch_limit == 5
    assert settings.langfuse_host == "https://cloud.langfuse.com"
    assert settings.langfuse_environment == "portfolio"


def test_chunk_overlap_must_be_smaller_than_chunk_size(tmp_path, monkeypatch):
    """Ошибочная комбинация параметров должна останавливаться до подготовки данных."""
    dotenv = tmp_path / ".env"
    dotenv.write_text("", encoding="utf-8")
    monkeypatch.setenv("CHUNK_SIZE", "32")
    monkeypatch.setenv("CHUNK_OVERLAP_TOKENS", "32")

    with pytest.raises(ValueError, match="CHUNK_OVERLAP_TOKENS"):
        Settings.from_env(dotenv)


def test_generation_batch_limit_cannot_exceed_five(tmp_path, monkeypatch):
    """Ограничение пользователя защищается и на уровне конфигурации."""
    dotenv = tmp_path / ".env"
    dotenv.write_text("", encoding="utf-8")
    monkeypatch.setenv("RAG_GENERATION_BATCH_LIMIT", "6")

    with pytest.raises(ValueError, match="must not exceed 5"):
        Settings.from_env(dotenv)

