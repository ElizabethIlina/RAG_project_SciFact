"""Минимальный клиент GigaChat для одного контролируемого batch запуска."""

from __future__ import annotations

from collections.abc import Iterable
from time import perf_counter
from typing import Any

from gigachat import GigaChat
from gigachat.models import Chat, Messages, MessagesRole

from rag_playground.config import Settings
from rag_playground.generation.models import ProviderGeneration


class GigaChatGenerator:
    """Обернуть SDK и никогда не включать credentials в результат или repr."""

    def __init__(self, settings: Settings) -> None:
        if not settings.gigachat_credentials:
            raise ValueError("GIGACHAT_CREDENTIALS is not configured")
        self.model = settings.gigachat_model
        self.max_tokens = settings.rag_generation_max_tokens
        self._client = GigaChat(
            base_url=settings.gigachat_base_url,
            credentials=settings.gigachat_credentials,
            scope=settings.gigachat_scope,
            verify_ssl_certs=settings.gigachat_verify_ssl_certs,
            ca_bundle_file=settings.gigachat_ca_bundle_file,
            max_retries=2,
        )

    def __enter__(self) -> GigaChatGenerator:
        """Разрешить один OAuth-сеанс для не более чем пяти запросов."""
        self._client.__enter__()
        return self

    def __exit__(self, exc_type: Any, exc: Any, traceback: Any) -> None:
        """Гарантированно закрыть HTTP-клиент SDK."""
        self._client.__exit__(exc_type, exc, traceback)

    def generate(self, prompt: str) -> tuple[ProviderGeneration, float]:
        """Отправить ровно один USER prompt в зафиксированную GigaChat-2."""
        chat = Chat(
            model=self.model,
            messages=[Messages(role=MessagesRole.USER, content=prompt)],
            temperature=0.1,
            max_tokens=self.max_tokens,
        )
        started = perf_counter()
        response = self._client.chat(chat)
        latency_ms = round((perf_counter() - started) * 1000, 3)
        choice = response.choices[0]
        usage = response.usage
        return (
            ProviderGeneration(
                answer=choice.message.content.strip(),
                model=response.model,
                finish_reason=choice.finish_reason,
                prompt_tokens=usage.prompt_tokens,
                completion_tokens=usage.completion_tokens,
                total_tokens=usage.total_tokens,
            ),
            latency_ms,
        )


def require_safe_generation_batch(
    items: Iterable[object],
    configured_limit: int,
) -> list[object]:
    """Материализовать batch и физически запретить 0 или более 5 API-вызовов."""
    materialized = list(items)
    hard_limit = min(configured_limit, 5)
    if not 1 <= len(materialized) <= hard_limit:
        raise ValueError(
            f"Generation batch must contain 1..{hard_limit} requests; "
            f"received {len(materialized)}"
        )
    return materialized
