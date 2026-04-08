"""
app/routers/graph_router.py

Routeur FastAPI pour l'extraction de concepts et la génération du graphe cognitif.

Endpoint :
  POST /api/ai/extract-concepts

Utilise : app.services.nlp_service
"""

from __future__ import annotations

import logging

from fastapi import APIRouter, HTTPException, status

from app.models.schemas import GraphNode, GraphRelation, GraphRequest, GraphResponse
from app.services import nlp_service

logger = logging.getLogger(__name__)

router = APIRouter(
    prefix="/api/ai",
    tags=["Graphe Cognitif"],
)


@router.post(
    "/extract-concepts",
    response_model=GraphResponse,
    status_code=status.HTTP_200_OK,
    summary="Extraire les concepts clés d'un texte de cours",
    description=(
        "Analyse un texte de cours via NLP (spaCy NER ou règles simples) et retourne "
        "les concepts extraits sous forme de nœuds et de relations pour alimenter "
        "un graphe cognitif côté frontend Angular."
    ),
)
async def extraire_concepts(payload: GraphRequest) -> GraphResponse:
    """
    Endpoint d'extraction de concepts pour le graphe cognitif.

    - **texte_cours** : Le texte du cours à analyser.
    - **langue** : Code de langue ('fr' ou 'en', défaut 'fr').
    """
    try:
        resultat = nlp_service.extraire_concepts(
            texte=payload.texte_cours,
            langue=payload.langue,
        )
    except Exception as exc:
        logger.exception("Erreur dans le service NLP lors de l'extraction de concepts.")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Erreur interne du service NLP. Veuillez réessayer.",
        ) from exc

    noeuds = [GraphNode(**n) for n in resultat["noeuds"]]
    relations = [GraphRelation(**r) for r in resultat["relations"]]

    return GraphResponse(
        noeuds=noeuds,
        relations=relations,
        meta=resultat["meta"],
    )
