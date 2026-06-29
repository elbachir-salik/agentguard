"""OpenAI Python SDK integration for AgentGuard."""

from __future__ import annotations

import inspect
from contextlib import asynccontextmanager, contextmanager
from typing import Any, AsyncGenerator, Generator

from agentguard.extractors import OpenAIExtractor
from agentguard.guard import Guard
from agentguard.session import Session

try:
    import openai  # noqa: F401

    _HAS_OPENAI = True
except ImportError:
    _HAS_OPENAI = False


def is_available() -> bool:
    return _HAS_OPENAI


def _is_async_client(client: Any) -> bool:
    create = getattr(getattr(getattr(client, "chat", None), "completions", None), "create", None)
    return inspect.iscoroutinefunction(create)


class _GuardedCompletions:
    """Sync chat.completions with recording and circuit-breaker checks."""

    def __init__(self, inner: Any, session: Session) -> None:
        self._inner = inner
        self._session = session

    def create(self, *args: Any, **kwargs: Any) -> Any:
        if kwargs.get("stream"):
            return self._session.stream(self._inner.create, *args, **kwargs)
        return self._session.call(
            self._inner.create,
            *args,
            **kwargs,
            extractor=OpenAIExtractor(),
        )

    def __getattr__(self, name: str) -> Any:
        return getattr(self._inner, name)


class _GuardedAsyncCompletions:
    """Async chat.completions with recording and circuit-breaker checks."""

    def __init__(self, inner: Any, session: Session) -> None:
        self._inner = inner
        self._session = session

    async def create(self, *args: Any, **kwargs: Any) -> Any:
        if kwargs.get("stream"):
            return await self._session.astream(self._inner.create, *args, **kwargs)
        return await self._session.acall(
            self._inner.create,
            *args,
            **kwargs,
            extractor=OpenAIExtractor(),
        )

    def __getattr__(self, name: str) -> Any:
        return getattr(self._inner, name)


class _GuardedChat:
    def __init__(self, inner: Any, session: Session, *, async_client: bool) -> None:
        self._inner = inner
        self._session = session
        self._async_client = async_client
        self._completions: _GuardedCompletions | _GuardedAsyncCompletions | None = None

    @property
    def completions(self) -> _GuardedCompletions | _GuardedAsyncCompletions:
        if self._completions is None:
            if self._async_client:
                self._completions = _GuardedAsyncCompletions(self._inner.completions, self._session)
            else:
                self._completions = _GuardedCompletions(self._inner.completions, self._session)
        return self._completions

    def __getattr__(self, name: str) -> Any:
        return getattr(self._inner, name)


class GuardedOpenAIClient:
    """Proxy around an OpenAI (or compatible) client with guarded ``chat.completions``."""

    def __init__(self, client: Any, session: Session) -> None:
        self._client = client
        self._session = session
        self._async_client = _is_async_client(client)
        self._chat = _GuardedChat(client.chat, session, async_client=self._async_client)

    @property
    def chat(self) -> _GuardedChat:
        return self._chat

    def __getattr__(self, name: str) -> Any:
        return getattr(self._client, name)


def guarded_client(client: Any, session: Session) -> GuardedOpenAIClient:
    """Wrap an OpenAI client so ``client.chat.completions.create`` is guarded."""
    return GuardedOpenAIClient(client, session)


@contextmanager
def guard_openai(
    guard: Guard,
    client: Any,
    metadata: dict | None = None,
    parent_session_id: str | None = None,
) -> Generator[tuple[Session, GuardedOpenAIClient], None, None]:
    """Open a Guard session and return ``(session, guarded_client)``."""
    with guard.session(metadata=metadata, parent_session_id=parent_session_id) as session:
        yield session, guarded_client(client, session)


@asynccontextmanager
async def aguard_openai(
    guard: Guard,
    client: Any,
    metadata: dict | None = None,
    parent_session_id: str | None = None,
) -> AsyncGenerator[tuple[Session, GuardedOpenAIClient], None]:
    """Async context manager — same as ``guard_openai`` for async client code."""
    with guard.session(metadata=metadata, parent_session_id=parent_session_id) as session:
        yield session, guarded_client(client, session)
