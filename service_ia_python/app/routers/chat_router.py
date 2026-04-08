"""
app/routers/chat_router.py

Routeur FastAPI pour le chat IA (RAG / LLM).

Endpoint :
  POST /api/ai/chat

Utilise : app.services.llm_service
"""

from __future__ import annotations

import logging

from fastapi import APIRouter, HTTPException, status

from app.models.schemas import ChatRequest, ChatResponse
from app.services import llm_service

logger = logging.getLogger(__name__)

router = APIRouter(
    prefix="/api/ai",
    tags=["Chat IA"],
)


@router.post(
    "/chat",
    response_model=ChatResponse,
    status_code=status.HTTP_200_OK,
    summary="Répondre à une question d'un étudiant via LLM / RAG",
    description=(
        "Reçoit une question en langage naturel posée par un étudiant et retourne "
        "une réponse générée par le service LLM (simulé en mode mock, "
        "remplaçable par un vrai modèle LLM + pipeline RAG)."
    ),
)
async def chat(payload: ChatRequest) -> ChatResponse:
    """
    Endpoint de chat IA.

    - **question** : La question posée par l'étudiant.
    - **student_id** : L'identifiant de l'étudiant (utilisé pour la personnalisation future).
    """
    try:
        resultat = llm_service.repondre_question(
            question=payload.question,
            student_id=payload.student_id,
        )
    except Exception as exc:
        logger.exception("Erreur dans le service LLM pour l'étudiant '%s'.", payload.student_id)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Erreur interne du service LLM. Veuillez réessayer.",
        ) from exc

    return ChatResponse(
        student_id=payload.student_id,
        question=payload.question,
        answer=resultat["answer"],
        source_documents=resultat["source_documents"],
    )
