"""Безопасные проверки доступности внешних сервисов.

"""

from __future__ import annotations

import json
from dataclasses import dataclass
from urllib.error import URLError
from urllib.request import Request, urlopen

from rag_playground.config import Settings


@dataclass(frozen=True, slots=True)
class CheckResult:
    """Единый формат результа проверки одного сервиса.

    ``name`` — читаемое имя, ``ok`` — итоговый статус, ``detail`` — краткое
    безопасное пояснение для пользователя.
    """

    name: str
    ok: bool
    detail: str


def check_http_service(name: str, url: str) -> CheckResult:
    """Проверить HTTP-сервис коротким GET-запросом.

    Тело ответа ограничено 512 байтами: для health check не нужно скачивать большой
    ответ. Если корень сервиса отдаёт JSON с версией, она добавляется в детали.
    Ответ без JSON тоже считается валидным, если HTTP-статус успешен.
    """
    # Явно сообщаем, что предпочитаем JSON, хотя для проверки не требуем его обязательно.
    request = Request(url, headers={"Accept": "application/json"})
    try:
        # Короткий timeout не даёт CLI зависнуть, если контейнер ещё запускается
        # или адрес указан неверно.
        with urlopen(request, timeout=5) as response:  # noqa: S310 - configured local URL
            payload = response.read(512).decode("utf-8", errors="replace")
            detail = f"HTTP {response.status}"
            if payload:
                try:
                    parsed = json.loads(payload)
                    version = parsed.get("version")
                    if isinstance(version, dict):
                        detail += f"; version={version.get('number', 'unknown')}"
                except json.JSONDecodeError:
                    # Qdrant может отвечать обычным текстом. Невозможность извлечь версию
                    # не делает сам сервис недоступным.
                    pass
            return CheckResult(name, 200 <= response.status < 400, detail)
    except (OSError, URLError) as exc:
        return CheckResult(name, False, f"unavailable: {exc}")


def check_local_services(settings: Settings) -> list[CheckResult]:
    """Проверить оба поисковых хранилища в одинаковом формате."""
    return [
        check_http_service("Qdrant", settings.qdrant_url),
        check_http_service("Elasticsearch", settings.elasticsearch_url),
    ]


def check_gigachat(settings: Settings) -> CheckResult:
    """Авторизоваться в GigaChat и показать доступные аккаунту модели.

    Запрос списка моделей одновременно проверяет credentials, scope, URL и TLS. При этом
    не тратятся токены на генерацию. В результат попадают только имена моделей,
    но не credentials и не access token.
    """
    # До импорта SDK даём понятную ошибку для самого частого случая — незаполненного .env.
    if not settings.gigachat_credentials:
        return CheckResult("GigaChat", False, "GIGACHAT_CREDENTIALS is not configured")
    try:
        from gigachat import GigaChat

        # Контекстный менеджер гарантирует закрытие HTTP-клиента. Проверка TLS берётся
        # из настроек и по умолчанию остаётся включённой.
        with GigaChat(
            credentials=settings.gigachat_credentials,
            scope=settings.gigachat_scope,
            base_url=settings.gigachat_base_url,
            verify_ssl_certs=settings.gigachat_verify_ssl_certs,
            ca_bundle_file=settings.gigachat_ca_bundle_file,
            max_retries=2,
        ) as client:
            response = client.get_models()
            model_ids = [model.id_ for model in response.data]
        detail = "available models: " + ", ".join(model_ids)
        return CheckResult("GigaChat", True, detail)
    # SDK может вернуть разные типы ошибок: OAuth, TLS, HTTP или timeout. Для health check
    # важно не уронить весь CLI, а вернуть единый CheckResult с типом и текстом ошибки.
    except Exception as exc:  # SDK публикует несколько семейств исключений.
        return CheckResult("GigaChat", False, f"{type(exc).__name__}: {exc}")

