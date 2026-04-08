"""
app/routers/prediction_router.py

Routeur FastAPI pour la prédiction de réussite académique.

Endpoint :
  POST /api/ai/predict-success

Utilise : app.services.ml_service
"""

from __future__ import annotations

import logging

from fastapi import APIRouter, HTTPException, status

from app.models.schemas import PredictionRequest, PredictionResponse
from app.services import ml_service

logger = logging.getLogger(__name__)

router = APIRouter(
    prefix="/api/ai",
    tags=["Prédiction de réussite"],
)


@router.post(
    "/predict-success",
    response_model=PredictionResponse,
    status_code=status.HTTP_200_OK,
    summary="Prédire la probabilité de réussite d'un étudiant",
    description=(
        "Reçoit les données comportementales et académiques d'un étudiant et retourne "
        "la probabilité de réussite prédite par un modèle Random Forest "
        "(ou une valeur mock si le modèle .pkl n'est pas encore disponible)."
    ),
)
async def predire_reussite(payload: PredictionRequest) -> PredictionResponse:
    """
    Endpoint de prédiction de réussite.

    Les features sont transmises au service ML dans l'ordre attendu par le modèle :
    [temps_connexion_hebdo, nombre_exercices_completes, moyenne_notes,
     taux_participation_forum, nombre_cours_suivis]
    """
    features = [
        payload.temps_connexion_hebdo,
        float(payload.nombre_exercices_completes),
        payload.moyenne_notes,
        payload.taux_participation_forum,
        float(payload.nombre_cours_suivis),
    ]

    try:
        resultat = ml_service.predire_reussite(features=features)
    except Exception as exc:
        logger.exception(
            "Erreur dans le service ML (prédiction) pour l'étudiant '%s'.", payload.student_id
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Erreur interne du service de prédiction. Veuillez réessayer.",
        ) from exc

    proba = resultat["probabilite_reussite"]

    return PredictionResponse(
        student_id=payload.student_id,
        probabilite_reussite=proba,
        probabilite_pourcentage=f"{proba * 100:.2f} %",
        classe_predite=resultat["classe_predite"],
        modele_utilise=resultat["modele_utilise"],
    )
