"""FastAPI integration for AgentGuard."""

from __future__ import annotations

from collections.abc import AsyncGenerator, Callable
from typing import Any

from agentguard.exceptions import CircuitBreakerTripped
from agentguard.guard import Guard
from agentguard.session import Session

try:
    from fastapi import FastAPI, Request
    from fastapi.responses import JSONResponse

    _HAS_FASTAPI = True
except ImportError:
    _HAS_FASTAPI = False
    FastAPI = Any  # type: ignore[misc, assignment]
    Request = Any  # type: ignore[misc, assignment]


def is_available() -> bool:
    return _HAS_FASTAPI


def _require_fastapi() -> None:
    if not _HAS_FASTAPI:
        raise ImportError(
            "FastAPI is required for this integration. "
            "Install with: pip install agentguard[fastapi]"
        )


def metadata_from_request(request: Request) -> dict[str, str]:
    """Build default session metadata from an HTTP request."""
    meta: dict[str, str] = {
        "path": request.url.path,
        "method": request.method,
    }
    if request.client and request.client.host:
        meta["client_host"] = request.client.host
    return meta


def setup_agentguard(
    app: FastAPI,
    guard: Guard,
    *,
    register_handlers: bool = True,
) -> None:
    """Attach a Guard to the FastAPI app and optionally register trip handlers."""
    _require_fastapi()
    app.state.agentguard = guard
    if register_handlers:
        register_trip_handler(app)


def register_trip_handler(app: FastAPI) -> None:
    """Map ``CircuitBreakerTripped`` to HTTP 429 with breaker details."""
    _require_fastapi()

    @app.exception_handler(CircuitBreakerTripped)
    async def _handle_trip(_request: Request, exc: CircuitBreakerTripped) -> JSONResponse:
        event = exc.event
        return JSONResponse(
            status_code=429,
            content={
                "detail": str(exc),
                "rule": event.rule,
                "trigger": event.trigger,
                "turn": event.turn,
            },
        )


def create_session_dependency(
    guard: Guard | None = None,
    metadata_fn: Callable[[Request], dict[str, str]] | None = None,
) -> Callable[..., AsyncGenerator[Session, None]]:
    """Return a FastAPI dependency that opens a Guard session per request.

    Usage::

        SessionDep = create_session_dependency(guard)
        # or rely on app.state.agentguard from setup_agentguard(app, guard)

        @app.post("/chat")
        async def chat(session: Session = Depends(SessionDep)):
            ...
    """
    _require_fastapi()

    async def _dependency(request: Request) -> AsyncGenerator[Session, None]:
        g: Guard = guard or request.app.state.agentguard
        meta = (metadata_fn or metadata_from_request)(request)
        with g.session(metadata=meta) as session:
            yield session

    return _dependency
