"""Создание современного Langfuse Cloud клиента из безопасной конфигурации."""

from __future__ import annotations

from langfuse import Langfuse

from rag_playground.config import Settings
from rag_playground.observability.redaction import make_langfuse_mask


def create_langfuse_client(settings: Settings) -> Langfuse:
    """Создать клиент v4 и установить маскирование до экспортера OpenTelemetry."""
    if not settings.langfuse_public_key or not settings.langfuse_secret_key:
        raise ValueError("LANGFUSE_PUBLIC_KEY and LANGFUSE_SECRET_KEY are required")
    secrets = tuple(
        value
        for value in (
            settings.langfuse_public_key,
            settings.langfuse_secret_key,
            settings.gigachat_credentials,
        )
        if value
    )
    return Langfuse(
        public_key=settings.langfuse_public_key,
        secret_key=settings.langfuse_secret_key,
        host=settings.langfuse_host,
        environment=settings.langfuse_environment,
        release="rag-playground-stage-6",
        # В коротких CLI-командах отправляем накопленное только явным flush().
        flush_at=100,
        mask=make_langfuse_mask(secrets),
    )


def check_langfuse_connection(settings: Settings) -> dict[str, object]:
    """Проверить credentials без записи traces и без вывода самих ключей."""
    client = create_langfuse_client(settings)
    return {
        "ok": bool(client.auth_check()),
        "host": settings.langfuse_host,
        "environment": settings.langfuse_environment,
    }
