"""
app/services/ml_service.py

Service Machine Learning pour :
  1. La prédiction de réussite académique (Random Forest Classifier).
  2. Le matchmaking d'étudiants similaires (K-Means + KNN).

ÉTAT ACTUEL :
  - Si les fichiers .pkl ne sont pas présents dans `saved_models/`, le service utilise
    des valeurs factices (mock) afin que l'API reste fonctionnelle dès le démarrage.
  - Les modèles sont chargés une seule fois au démarrage du serveur (cache module).

INTÉGRATION FUTURE – Prédiction de réussite :
  Entraînez un RandomForestClassifier dans le notebook dédié, puis :
      import joblib
      joblib.dump(modele, "saved_models/prediction_reussite.pkl")
  Le service chargera automatiquement ce fichier au prochain démarrage.

INTÉGRATION FUTURE – Matchmaking :
  Entraînez un KMeans ET un NearestNeighbors dans le notebook dédié, puis :
      joblib.dump(kmeans, "saved_models/matchmaking_kmeans.pkl")
      joblib.dump(knn,    "saved_models/matchmaking_knn.pkl")
  Fournissez également la liste des student_ids dans le même ordre que les données
  d'entraînement via `saved_models/student_ids.pkl`.
"""

from __future__ import annotations

import logging
import os
import random
from pathlib import Path
from typing import Any

import joblib
import numpy as np

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Chemins vers les fichiers de modèles sauvegardés
# ---------------------------------------------------------------------------

_BASE_DIR = Path(__file__).resolve().parent.parent.parent
_SAVED_MODELS_DIR = _BASE_DIR / "saved_models"

_PATH_PREDICTION = _SAVED_MODELS_DIR / "prediction_reussite.pkl"
_PATH_KMEANS = _SAVED_MODELS_DIR / "matchmaking_kmeans.pkl"
_PATH_KNN = _SAVED_MODELS_DIR / "matchmaking_knn.pkl"
_PATH_STUDENT_IDS = _SAVED_MODELS_DIR / "student_ids.pkl"


# ---------------------------------------------------------------------------
# Chargement paresseux des modèles (une seule fois par démarrage)
# ---------------------------------------------------------------------------

def _charger_modele(chemin: Path, nom: str) -> Any | None:
    """
    Tente de charger un modèle joblib depuis le chemin indiqué.

    Retourne `None` si le fichier n'existe pas encore (mode mock activé).
    """
    if chemin.exists():
        logger.info("Chargement du modèle '%s' depuis '%s'.", nom, chemin)
        return joblib.load(chemin)
    logger.warning(
        "Modèle '%s' introuvable (%s). Mode mock activé pour ce service.", nom, chemin
    )
    return None


# Chargement au démarrage du module
_modele_prediction = _charger_modele(_PATH_PREDICTION, "prediction_reussite")
_modele_kmeans = _charger_modele(_PATH_KMEANS, "matchmaking_kmeans")
_modele_knn = _charger_modele(_PATH_KNN, "matchmaking_knn")
_student_ids: list[str] = (
    joblib.load(_PATH_STUDENT_IDS) if _PATH_STUDENT_IDS.exists() else []
)


# ---------------------------------------------------------------------------
# 1. Prédiction de réussite
# ---------------------------------------------------------------------------

def predire_reussite(features: list[float]) -> dict[str, Any]:
    """
    Prédit la probabilité de réussite académique d'un étudiant.

    Args:
        features: Liste de valeurs numériques dans l'ordre :
                  [temps_connexion_hebdo, nombre_exercices_completes,
                   moyenne_notes, taux_participation_forum, nombre_cours_suivis]

    Returns:
        Dictionnaire contenant :
          - `probabilite_reussite` (float entre 0 et 1)
          - `classe_predite` ("REUSSITE" ou "ECHEC")
          - `modele_utilise` (str)

    TODO – Intégration réelle :
        X = np.array(features).reshape(1, -1)
        proba = _modele_prediction.predict_proba(X)[0][1]
        classe = "REUSSITE" if proba >= 0.5 else "ECHEC"
    """
    if _modele_prediction is not None:
        X = np.array(features, dtype=float).reshape(1, -1)
        proba: float = float(_modele_prediction.predict_proba(X)[0][1])
        classe = "REUSSITE" if proba >= 0.5 else "ECHEC"
        modele_nom = "RandomForestClassifier (entraîné)"
    else:
        # --- Mode mock : valeur factice basée sur la moyenne des notes (index 2) ---
        moyenne_notes = features[2] if len(features) > 2 else 10.0
        proba = min(max(round(moyenne_notes / 20.0, 2), 0.0), 1.0)
        classe = "REUSSITE" if proba >= 0.5 else "ECHEC"
        modele_nom = "Mock (aucun modèle .pkl chargé)"
        logger.debug("Mode mock – probabilité calculée depuis moyenne_notes=%.2f", moyenne_notes)

    return {
        "probabilite_reussite": proba,
        "classe_predite": classe,
        "modele_utilise": modele_nom,
    }


# ---------------------------------------------------------------------------
# 2. Matchmaking d'étudiants similaires
# ---------------------------------------------------------------------------

def trouver_etudiants_similaires(
    features: list[float],
    nombre_voisins: int = 5,
) -> dict[str, Any]:
    """
    Trouve les étudiants les plus similaires à partir d'un vecteur de features.

    Args:
        features: Vecteur de caractéristiques de l'étudiant de référence.
        nombre_voisins: Nombre de voisins (étudiants similaires) à retourner.

    Returns:
        Dictionnaire contenant :
          - `etudiants_similaires` (liste d'IDs)
          - `groupe_cluster` (int, numéro du cluster K-Means)
          - `modele_utilise` (str)

    TODO – Intégration réelle :
        X = np.array(features).reshape(1, -1)
        cluster = int(_modele_kmeans.predict(X)[0])
        distances, indices = _modele_knn.kneighbors(X, n_neighbors=nombre_voisins)
        ids_similaires = [_student_ids[i] for i in indices[0]]
    """
    if _modele_kmeans is not None and _modele_knn is not None:
        X = np.array(features, dtype=float).reshape(1, -1)
        groupe_cluster: int = int(_modele_kmeans.predict(X)[0])
        distances, indices = _modele_knn.kneighbors(X, n_neighbors=nombre_voisins)
        if _student_ids:
            ids_similaires = [_student_ids[i] for i in indices[0] if i < len(_student_ids)]
        else:
            ids_similaires = [f"etudiant-{i}" for i in indices[0]]
        modele_nom = "KMeans + KNN (entraînés)"
    else:
        # --- Mode mock : génère des IDs fictifs ---
        ids_similaires = [f"etudiant-{random.randint(100, 999)}" for _ in range(nombre_voisins)]
        groupe_cluster = random.randint(0, 3)
        modele_nom = "Mock (aucun modèle .pkl chargé)"
        logger.debug("Mode mock – %d étudiants similaires générés aléatoirement.", nombre_voisins)

    return {
        "etudiants_similaires": ids_similaires,
        "groupe_cluster": groupe_cluster,
        "modele_utilise": modele_nom,
    }
