"""FastAPI web application for the authenticated RAG workspace."""

from __future__ import annotations

import hmac
import logging
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from fastapi import Depends, FastAPI, Form, HTTPException, Request, status
from fastapi.responses import HTMLResponse, RedirectResponse
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel, Field
from starlette.staticfiles import StaticFiles
from starlette.middleware.sessions import SessionMiddleware

from rag.config.settings import settings
from rag.retrieval.rag_pipeline import RAGPipeline

load_dotenv()

TEMPLATES_DIR = Path(__file__).parent / "templates"


class ChatRequest(BaseModel):
    question: str = Field(min_length=1)
    k: int = Field(default=4, ge=1, le=20)


class ChatResponse(BaseModel):
    answer: str
    queries: list[str]
    retrieved_chunks: int


def create_app(
    pipeline: RAGPipeline | None = None,
    config: dict[str, Any] | None = None,
) -> FastAPI:
    """Create the authenticated FastAPI application."""
    app = FastAPI(title="Northstar RAG API", version="1.0.0")
    app_config: dict[str, Any] = {
        "secret_key": settings.app_secret_key,
        "app_username": settings.app_username,
        "app_password": settings.app_password,
    }
    app_config.update(config or {})
    app.state.config = app_config
    app.state.rag_pipeline = pipeline or RAGPipeline()
    app.state.templates = Jinja2Templates(directory=str(TEMPLATES_DIR))
    if hasattr(app.state.rag_pipeline, "sync_and_index"):
        try:
            app.state.rag_pipeline.sync_and_index()
        except Exception:
            logging.getLogger("rag.api.routes").exception(
                "Failed to index documents during app startup. The app will continue without an initial sync."
            )
    app.mount("/static", StaticFiles(directory=str(Path(__file__).parent / "static")), name="static")
    app.add_middleware(SessionMiddleware, secret_key=app_config["secret_key"])

    def is_authenticated(request: Request) -> bool:
        return bool(request.session.get("authenticated"))

    def require_auth(request: Request) -> None:
        if not is_authenticated(request):
            raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="authentication required")

    @app.get("/", response_class=HTMLResponse)
    def home(request: Request) -> Any:
        if is_authenticated(request):
            return RedirectResponse("/chat", status_code=status.HTTP_303_SEE_OTHER)
        return app.state.templates.TemplateResponse(request, "login.html")

    @app.post("/login", response_class=HTMLResponse)
    def login(
        request: Request,
        username: str = Form(...),
        password: str = Form(...),
    ) -> Any:
        valid_username = hmac.compare_digest(username, str(app_config["app_username"]))
        valid_password = hmac.compare_digest(password, str(app_config["app_password"]))
        if not (valid_username and valid_password):
            return app.state.templates.TemplateResponse(
                request,
                "login.html",
                {"error": "Invalid username or password."},
                status_code=status.HTTP_401_UNAUTHORIZED,
            )

        request.session.clear()
        request.session["authenticated"] = True
        request.session["username"] = username
        return RedirectResponse("/chat", status_code=status.HTTP_303_SEE_OTHER)

    @app.post("/logout")
    def logout(request: Request) -> RedirectResponse:
        request.session.clear()
        return RedirectResponse("/", status_code=status.HTTP_303_SEE_OTHER)

    @app.get("/chat", response_class=HTMLResponse)
    def chat(request: Request) -> Any:
        if not is_authenticated(request):
            return RedirectResponse("/", status_code=status.HTTP_303_SEE_OTHER)
        return app.state.templates.TemplateResponse(
            request,
            "chat.html",
            {"username": request.session.get("username", "User")},
        )

    @app.post("/api/chat", response_model=ChatResponse)
    def api_chat(payload: ChatRequest, _: None = Depends(require_auth)) -> ChatResponse:
        try:
            result = app.state.rag_pipeline.ask(payload.question.strip(), k=payload.k)
            return ChatResponse(
                answer=result.answer,
                queries=result.queries,
                retrieved_chunks=result.retrieved_chunks,
            )
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        except Exception as exc:
            raise HTTPException(status_code=500, detail="Unable to answer the question right now.") from exc

    @app.get("/health")
    def health_check() -> dict[str, str]:
        return {"status": "ok"}

    return app


app = create_app()
