"""Chat over your documents. The retrieval and answering logic is not in this copy of QuizForge yet."""

from __future__ import annotations

from typing import List

from fastapi import APIRouter, Depends
from pydantic import BaseModel, Field

from .. import documents
from .deps import Principal, current_principal
from .errors import ApiProblem

router = APIRouter(tags=["chat"])


class ChatBody(BaseModel):
    documentIds: List[str] = Field(min_length=1, max_length=20)
    message: str = Field(min_length=1, max_length=4000)


@router.post("/chat")
def chat(body: ChatBody, who: Principal = Depends(current_principal)):
    try:  # the ownership check is real, so the endpoint is already safe to build on
        documents.owned(who.user.id, body.documentIds)
    except documents.NotOwned:
        raise ApiProblem("not_found", "One of those files was not found.") from None
    raise ApiProblem("not_implemented", "Chat is not available in this version of the back end yet.")
