"""The HTTP API: `uvicorn app.api.main:app`. Run behind HTTPS in production."""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware

from .. import config
from ..auth import tokens, users
from ..database import init_db
from . import errors, routes_auth, routes_chat, routes_documents, routes_me, routes_quizzes

log = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(_: FastAPI):
    tokens.secret()  # refuse to start in production without a real AUTH_SECRET
    init_db()
    removed = users.delete_expired_guests()
    if removed:
        log.info("Removed %d expired guest accounts", removed)
    yield


def create_app() -> FastAPI:
    app = FastAPI(
        title="QuizForge API",
        lifespan=lifespan,
        docs_url=None if config.APP_ENV == "production" else "/api/docs",
        redoc_url=None,
        openapi_url=None if config.APP_ENV == "production" else "/api/openapi.json",
    )
    app.add_middleware(
        CORSMiddleware,
        allow_origins=[config.APP_ORIGIN],
        allow_credentials=True,
        allow_methods=["GET", "POST", "DELETE"],
        allow_headers=["Authorization", "Content-Type", "X-Api-Key", "X-Provider", "X-Requested-With"],
    )

    @app.middleware("http")
    async def secure_headers(request: Request, call_next):
        response = await call_next(request)
        response.headers.setdefault("X-Content-Type-Options", "nosniff")
        response.headers.setdefault("X-Frame-Options", "DENY")
        response.headers.setdefault("Referrer-Policy", "no-referrer")
        if request.url.path.startswith("/api/auth") or request.url.path.startswith("/api/me"):
            response.headers["Cache-Control"] = "no-store"
        return response

    errors.install(app)
    for module in (routes_auth, routes_me, routes_documents, routes_quizzes, routes_chat):
        app.include_router(module.router, prefix="/api")

    @app.get("/api/health")
    def health():
        return {"status": "ok"}

    return app


app = create_app()
