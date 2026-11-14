"""One error shape for the whole API: {"code": "...", "message": "..."}."""

from __future__ import annotations

from typing import Dict, Optional

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from starlette.exceptions import HTTPException as StarletteHTTPException

STATUS: Dict[str, int] = {
    "unauthorized": 401,
    "invalid_credentials": 401,
    "forbidden": 403,
    "csrf": 403,
    "guest_limit": 403,
    "not_found": 404,
    "email_taken": 409,
    "rate_limited": 429,
    "invalid_request": 400,
    "invalid_email": 400,
    "weak_password": 400,
    "wrong_password": 400,
    "not_guest": 400,
    "no_password": 400,
    "confirmation_mismatch": 400,
    "needs_gemini": 400,
    "invalid_key": 400,
    "model_error": 502,
    "not_implemented": 501,
}


class ApiProblem(Exception):
    def __init__(self, code: str, message: str, *, status: Optional[int] = None, headers: Optional[dict] = None):
        super().__init__(message)
        self.code, self.message = code, message
        self.status = status or STATUS.get(code, 400)
        self.headers = headers or {}


def model_problem(exc: Exception) -> ApiProblem:
    """A model call failed. Say why in words the person can act on; never repeat the provider's raw message,
    which can include request details."""
    text = str(exc)
    if "API_KEY_INVALID" in text or "API key not valid" in text or "PERMISSION_DENIED" in text:
        return ApiProblem("invalid_key", "Gemini did not accept that API key. Check it and try again.")
    if "RESOURCE_EXHAUSTED" in text or " 429" in text or "quota" in text.lower():
        return ApiProblem(
            "rate_limited",
            "Your Gemini quota is used up for now. Wait a while, or use another key.",
            headers={"Retry-After": "60"},
        )
    return ApiProblem("model_error", "The model could not be reached. Try again in a moment.")


def install(app: FastAPI) -> None:
    @app.exception_handler(ApiProblem)
    async def _problem(_: Request, exc: ApiProblem):
        return JSONResponse({"code": exc.code, "message": exc.message}, status_code=exc.status, headers=exc.headers)

    @app.exception_handler(RequestValidationError)
    async def _invalid(_: Request, exc: RequestValidationError):
        first = exc.errors()[0] if exc.errors() else {}
        field = ".".join(str(p) for p in first.get("loc", []) if p not in ("body", "query"))
        return JSONResponse(
            {
                "code": "invalid_request",
                "message": f"Check {field or 'the request'}: {first.get('msg', 'invalid value')}.",
            },
            status_code=400,
        )

    @app.exception_handler(StarletteHTTPException)
    async def _http(_: Request, exc: StarletteHTTPException):
        code = {404: "not_found", 405: "invalid_request"}.get(exc.status_code, "unknown")
        return JSONResponse({"code": code, "message": str(exc.detail)}, status_code=exc.status_code)
