"""Upload files or web pages. The reply is a stream of JSON lines so a long scan can show its progress."""

from __future__ import annotations

import io
import json
import logging
import queue
import threading
import uuid
from typing import Callable, Optional

from fastapi import APIRouter, Depends, File, Response, UploadFile
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from .. import config, documents
from ..llm import LLMError
from ..loaders import UploadError, check_size
from ..ocr import OCRLimitError, gemini_ocr, image_mime
from ..pdf_loader import EmptyPDFError
from ..qa_generator import ingest_document, ingest_url
from . import llm_factory
from .deps import ModelCredentials, Principal, current_principal, model_credentials
from .errors import ApiProblem, model_problem

log = logging.getLogger(__name__)
router = APIRouter(prefix="/documents", tags=["documents"])


class UrlBody(BaseModel):
    url: str = Field(max_length=2000)


def error_event(exc: Exception) -> dict:
    """A user-safe description of why reading failed. Unexpected errors are logged, not shown."""
    if isinstance(exc, documents.LimitReached):
        return {
            "event": "error",
            "code": "guest_limit",
            "message": f"Guests can add up to {config.GUEST_MAX_FILES} files. Create an account to add more.",
        }
    if isinstance(exc, OCRLimitError):
        return {"event": "error", "code": "ocr_limit", "message": str(exc)}
    if isinstance(exc, UploadError):
        code = "needs_gemini" if "needs the Gemini" in str(exc) else "unsupported"
        return {"event": "error", "code": code, "message": str(exc)}
    if isinstance(exc, EmptyPDFError):
        return {"event": "error", "code": "unreadable", "message": str(exc)}
    if isinstance(exc, LLMError):
        problem = model_problem(exc)
        return {"event": "error", "code": problem.code, "message": problem.message}
    log.exception("document ingestion failed")
    return {"event": "error", "code": "unknown", "message": "Something went wrong reading this file."}


def stream(work: Callable[[Callable[[dict], None]], dict]) -> StreamingResponse:
    """Run `work(emit)` on a thread and send what it emits as newline-delimited JSON, then a done or error line."""
    events: "queue.Queue[Optional[dict]]" = queue.Queue()

    def run() -> None:
        try:
            events.put({"event": "done", "document": work(events.put)})
        except Exception as exc:  # reported to the caller as an error line
            events.put(error_event(exc))
        events.put(None)

    threading.Thread(target=run, daemon=True).start()

    def lines():
        while (item := events.get()) is not None:
            yield json.dumps(item) + "\n"

    return StreamingResponse(lines(), media_type="application/x-ndjson", headers={"Cache-Control": "no-store"})


def _record(who: Principal, doc_id: str, **fields) -> dict:
    """Register a finished upload. For a guest the file limit is checked again here, atomically; an upload that
    loses that race has its chunks removed so nothing is left behind."""
    limit = config.GUEST_MAX_FILES if who.user.is_guest else None
    try:
        return documents.register(who.user.id, doc_id, limit=limit, **fields)
    except documents.LimitReached:
        from ..vector_store import delete_doc

        delete_doc(doc_id)
        raise


def _guest_check(who: Principal, size: int = 0) -> None:
    if not who.user.is_guest:
        return
    if documents.count(who.user.id) >= config.GUEST_MAX_FILES:
        raise ApiProblem(
            "guest_limit", f"Guests can add up to {config.GUEST_MAX_FILES} files. Create an account to add more."
        )
    if size > config.GUEST_MAX_UPLOAD_MB * 1024 * 1024:
        raise ApiProblem(
            "guest_limit",
            f"Guests can add files up to {config.GUEST_MAX_UPLOAD_MB:g} MB. Create an account for larger files.",
        )


def short_name(name: str, limit: int = 200) -> str:
    """Cut a long file name without losing its extension (the type of a text file is read from it)."""
    if len(name) <= limit:
        return name
    stem, dot, ext = name.rpartition(".")
    if dot and stem and 0 < len(ext) <= 10:
        return stem[: limit - len(ext) - 1] + "." + ext
    return name[:limit]


def _page_count(data: bytes) -> Optional[int]:
    if data[:5] != b"%PDF-":
        return None
    from pypdf import PdfReader

    try:
        return len(PdfReader(io.BytesIO(data)).pages)
    except Exception:
        return None


@router.post("")
def upload(
    file: UploadFile = File(...),
    who: Principal = Depends(current_principal),
    creds: ModelCredentials = Depends(model_credentials),
):
    data = file.file.read(int(config.MAX_UPLOAD_MB * 1024 * 1024) + 1)
    name = short_name(file.filename or "upload")
    _guest_check(who, len(data))
    try:
        check_size(len(data))
    except UploadError as exc:
        raise ApiProblem("invalid_request", str(exc)) from None

    def work(emit: Callable[[dict], None]) -> dict:
        doc_id = str(uuid.uuid4())
        vision = llm_factory.make_vision(creds) if config.OCR_MODE != "off" else None
        used = {"ocr": False}
        reader = None
        if vision is not None:
            inner = gemini_ocr(
                vision,
                on_progress=lambda d, t, tb, f: emit(
                    {"event": "progress", "page": d, "pages": t, "tables": tb, "figures": f}
                ),
            )

            def reader(raw: bytes):  # noqa: F811 - wraps the real reader to note that it ran
                used["ocr"] = True
                return inner(raw)

            reader.tally = inner.tally  # type: ignore[attr-defined]
        ingest_document(None, doc_id, filename=name, data=data, ocr=reader, owner_id=who.user.id)
        method = "image" if image_mime(data) else "ocr" if used["ocr"] else "text"
        tally = getattr(reader, "tally", {"tables": 0, "figures": 0})
        return _record(
            who,
            doc_id,
            name=name,
            pages=_page_count(data),
            method=method,
            tables=tally["tables"],
            figures=tally["figures"],
        )

    return stream(work)


@router.post("/url")
def add_url(body: UrlBody, who: Principal = Depends(current_principal)):
    _guest_check(who)

    def work(_emit: Callable[[dict], None]) -> dict:
        doc_id = str(uuid.uuid4())
        ingest_url(body.url.strip(), doc_id, owner_id=who.user.id)
        from urllib.parse import urlparse

        parsed = urlparse(body.url.strip())
        return _record(
            who,
            doc_id,
            name=(parsed.hostname or "web") + (parsed.path.rstrip("/") or ""),
            pages=None,
            method="text",
        )

    return stream(work)


@router.get("")
def list_mine(who: Principal = Depends(current_principal)):
    return documents.list_documents(who.user.id)


@router.delete("/{doc_id}", status_code=204)
def remove(doc_id: str, who: Principal = Depends(current_principal)):
    if not documents.delete(who.user.id, doc_id):
        raise ApiProblem("not_found", "That file was not found.")
    return Response(status_code=204)
