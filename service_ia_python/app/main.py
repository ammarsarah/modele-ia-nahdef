"""
app/main.py

Point d'entrée principal du microservice IA Python.

Démarrage :
    uvicorn app.main:app --reload

L'API sera disponible sur http://localhost:8000
Documentation interactive : http://localhost:8000/docs
"""

from __future__ import annotations

import logging

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.routers import chat_router, graph_router, matchmaking_router, prediction_router

# ---------------------------------------------------------------------------
# Configuration du logging
# ---------------------------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Instanciation de l'application FastAPI
# ---------------------------------------------------------------------------

app = FastAPI(
    title="Microservice IA – Plateforme Éducative Nahdef",
    description=(
        "API d'Intelligence Artificielle pour la plateforme éducative Nahdef.\n\n"
        "Fonctionnalités :\n"
        "- **Chat IA** : Réponses aux questions des étudiants via LLM / RAG\n"
        "- **Prédiction de réussite** : Modèle Random Forest\n"
        "- **Matchmaking** : Regroupement d'étudiants similaires (K-Means + KNN)\n"
        "- **Graphe cognitif** : Extraction de concepts NLP (spaCy)\n\n"
        "Ce service est conçu pour fonctionner avec un backend Spring Boot "
        "et un frontend Angular."
    ),
    version="1.0.0",
    contact={
        "name": "Équipe Nahdef",
    },
    license_info={
        "name": "MIT",
    },
)

# ---------------------------------------------------------------------------
# Configuration CORS
# IMPORTANT : Autorise les requêtes provenant du frontend Angular (port 4200)
# et du backend Spring Boot (port 8080).
# Adaptez `allow_origins` selon votre environnement de déploiement.
# ---------------------------------------------------------------------------

# TODO – En production, remplacez "*" par les domaines exacts autorisés,
#         ex: ["https://nahdef.example.com", "https://api.nahdef.example.com"]
ALLOWED_ORIGINS = [
    "http://localhost:4200",   # Angular (développement)
    "http://localhost:8080",   # Spring Boot (développement)
    "http://127.0.0.1:4200",
    "http://127.0.0.1:8080",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------------------------------------------------------
# Inclusion des routeurs
# ---------------------------------------------------------------------------

app.include_router(chat_router.router)
app.include_router(prediction_router.router)
app.include_router(matchmaking_router.router)
app.include_router(graph_router.router)

# ---------------------------------------------------------------------------
# Route de santé (health check)
# ---------------------------------------------------------------------------

@app.get("/health", tags=["Santé"], summary="Vérifier l'état du service")
async def health_check() -> dict[str, str]:
    """Endpoint de vérification de l'état du microservice."""
    return {"status": "ok", "service": "microservice-ia-nahdef"}


# ---------------------------------------------------------------------------
# Point d'entrée pour l'exécution directe
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import uvicorn

    logger.info("Démarrage du microservice IA Nahdef…")
    uvicorn.run("app.main:app", host="0.0.0.0", port=8000, reload=True)
