"""Маскирование секретов до передачи данных внешнему наблюдаемому сервису."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

_SENSITIVE_KEY_PARTS = (
    "authorization",
    "credential",
    "password",
    "secret",
    "token",
)


def redact_payload(data: Any, secrets: tuple[str, ...] = ()) -> Any:
    """Рекурсивно скрыть чувствительные поля и точные значения известных секретов."""
    if isinstance(data, Mapping):
        redacted: dict[str, Any] = {}
        for key, value in data.items():
            key_text = str(key)
            if any(part in key_text.lower() for part in _SENSITIVE_KEY_PARTS):
                redacted[key_text] = "[REDACTED]"
            else:
                redacted[key_text] = redact_payload(value, secrets)
        return redacted
    if isinstance(data, str):
        result = data
        for secret in secrets:
            if secret:
                result = result.replace(secret, "[REDACTED]")
        return result
    if isinstance(data, Sequence) and not isinstance(data, (bytes, bytearray)):
        return [redact_payload(value, secrets) for value in data]
    return data


def make_langfuse_mask(secrets: tuple[str, ...]):
    """Создать callback формата Langfuse SDK, не раскрывая secrets через repr."""

    def mask(*, data: Any, **_kwargs: Any) -> Any:
        return redact_payload(data, secrets)

    return mask
