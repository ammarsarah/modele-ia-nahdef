"""
app/routers/matchmaking_router.py

Routeur FastAPI pour le matchmaking d'étudiants similaires.

Endpoint :
  POST /api/ai/match-students

Utilise : app.services.ml_service
"""

from __future__ import annotations

import logging

from fastapi import APIRouter, HTTPException, status

from app.models.schemas import MatchmakingRequest, MatchmakingResponse
from app.services import ml_service

logger = logging.getLogger(__name__)

router = APIRouter(
    prefix="/api/ai",
    tags=["Matchmaking"],
)


@router.post(
    "/match-students",
    response_model=MatchmakingResponse,
    status_code=status.HTTP_200_OK,
    summary="Trouver des étudiants similaires (matchmaking)",
    description=(
        "Reçoit le vecteur de features d'un étudiant et retourne la liste des IDs "
        "des étudiants les plus similaires, ainsi que le cluster K-Means auquel "
        "il appartient (ou des données mock si les modèles .pkl ne sont pas disponibles)."
    ),
)
async def match_students(payload: MatchmakingRequest) -> MatchmakingResponse:
    """
    Endpoint de matchmaking.

    - **student_id** : L'identifiant de l'étudiant de référence.
    - **features** : Vecteur numérique décrivant l'étudiant.
    - **nombre_voisins** : Nombre d'étudiants similaires à retourner (défaut : 5).
    """
    try:
        resultat = ml_service.trouver_etudiants_similaires(
            features=payload.features,
            nombre_voisins=payload.nombre_voisins,
        )
    except Exception as exc:
        logger.exception(
            "Erreur dans le service ML (matchmaking) pour l'étudiant '%s'.", payload.student_id
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Erreur interne du service de matchmaking. Veuillez réessayer.",
        ) from exc

    return MatchmakingResponse(
        student_id=payload.student_id,
        etudiants_similaires=resultat["etudiants_similaires"],
        groupe_cluster=resultat["groupe_cluster"],
        modele_utilise=resultat["modele_utilise"],
    )
