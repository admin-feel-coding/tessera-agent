from __future__ import annotations

from app.config import settings

_PLACEHOLDER_KEYS = ("", "placeholder-replace-me")

IS_LANGFUSE_ENABLED = settings.langfuse_public_key not in _PLACEHOLDER_KEYS


class _NoopSpan:
    def end(self, **kwargs: object) -> None: ...


class _NoopTrace:
    id: str = ""

    def span(self, **kwargs: object) -> _NoopSpan:
        return _NoopSpan()

    def generation(self, **kwargs: object) -> _NoopSpan:
        return _NoopSpan()

    def update(self, **kwargs: object) -> None: ...


class _NoopLangfuse:
    def trace(self, **kwargs: object) -> _NoopTrace:
        return _NoopTrace()

    def flush(self) -> None: ...


def _build_langfuse() -> object:
    if not IS_LANGFUSE_ENABLED:
        # noop trace so call sites stay branch-free in dev
        return _NoopLangfuse()

    from langfuse import Langfuse  # type: ignore[import-untyped]

    return Langfuse(
        public_key=settings.langfuse_public_key,
        secret_key=settings.langfuse_secret_key,
        host=settings.langfuse_host,
    )


langfuse: _NoopLangfuse = _build_langfuse()  # type: ignore[assignment]


def start_trace(name: str, input: dict) -> _NoopTrace:  # noqa: A002
    return langfuse.trace(name=name, input=input, metadata={"service": "tessera-agent"})  # type: ignore[return-value]
