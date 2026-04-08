"""
app/models/schemas.py

Définition de tous les modèles Pydantic utilisés comme schémas de requêtes (Request)
et de réponses (Response) pour les 4 endpoints de l'API IA.
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# 1. Chat IA (RAG / LLM)
# ---------------------------------------------------------------------------

class ChatRequest(BaseModel):
    """Corps de la requête pour le chat IA."""

    question: str = Field(
        ...,
        min_length=1,
        max_length=2000,
        description="La question posée par l'étudiant.",
        examples=["Comment fonctionne la récursivité ?"],
    )
    student_id: str = Field(
        ...,
        description="Identifiant unique de l'étudiant (correspond à l'ID Spring Boot).",
        examples=["etudiant-42"],
    )


class ChatResponse(BaseModel):
    """Corps de la réponse du chat IA."""

    student_id: str = Field(..., description="Identifiant de l'étudiant.")
    question: str = Field(..., description="La question telle qu'elle a été reçue.")
    answer: str = Field(..., description="Réponse générée par le modèle LLM / RAG.")
    source_documents: list[str] = Field(
        default_factory=list,
        description="Liste des titres de documents sources utilisés pour générer la réponse.",
    )


# ---------------------------------------------------------------------------
# 2. Prédiction de réussite académique
# ---------------------------------------------------------------------------

class PredictionRequest(BaseModel):
    """Corps de la requête pour la prédiction de réussite d'un étudiant."""

    student_id: str = Field(..., description="Identifiant unique de l'étudiant.")
    temps_connexion_hebdo: float = Field(
        ...,
        ge=0.0,
        description="Temps de connexion moyen par semaine (en heures).",
        examples=[8.5],
    )
    nombre_exercices_completes: int = Field(
        ...,
        ge=0,
        description="Nombre total d'exercices complétés par l'étudiant.",
        examples=[45],
    )
    moyenne_notes: float = Field(
        ...,
        ge=0.0,
        le=20.0,
        description="Moyenne générale de l'étudiant sur 20.",
        examples=[14.5],
    )
    taux_participation_forum: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description="Taux de participation aux forums (0 = jamais, 1 = très actif).",
        examples=[0.6],
    )
    nombre_cours_suivis: int = Field(
        default=1,
        ge=1,
        description="Nombre de cours auxquels l'étudiant est inscrit.",
        examples=[3],
    )


class PredictionResponse(BaseModel):
    """Corps de la réponse de la prédiction de réussite."""

    student_id: str = Field(..., description="Identifiant de l'étudiant.")
    probabilite_reussite: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Probabilité de réussite prédite (entre 0 et 1, ex: 0.80 = 80 %).",
    )
    probabilite_pourcentage: str = Field(
        ...,
        description="Probabilité formatée en pourcentage lisible, ex: '80.00 %'.",
    )
    classe_predite: str = Field(
        ...,
        description="Classe prédite : 'REUSSITE' ou 'ECHEC'.",
    )
    modele_utilise: str = Field(
        ...,
        description="Nom du modèle utilisé pour la prédiction.",
    )


# ---------------------------------------------------------------------------
# 3. Matchmaking d'étudiants similaires
# ---------------------------------------------------------------------------

class MatchmakingRequest(BaseModel):
    """Corps de la requête pour trouver des étudiants similaires (matchmaking)."""

    student_id: str = Field(..., description="Identifiant de l'étudiant de référence.")
    features: list[float] = Field(
        ...,
        min_length=1,
        description=(
            "Vecteur de caractéristiques numériques de l'étudiant "
            "(ex: [temps_connexion, notes, exercices, participation])."
        ),
        examples=[[8.5, 14.5, 45, 0.6]],
    )
    nombre_voisins: int = Field(
        default=5,
        ge=1,
        le=50,
        description="Nombre d'étudiants similaires à retourner.",
    )


class MatchmakingResponse(BaseModel):
    """Corps de la réponse du matchmaking."""

    student_id: str = Field(..., description="Identifiant de l'étudiant de référence.")
    etudiants_similaires: list[str] = Field(
        ...,
        description="Liste des identifiants des étudiants les plus similaires.",
    )
    groupe_cluster: int = Field(
        ...,
        description="Numéro du cluster K-Means auquel appartient l'étudiant.",
    )
    modele_utilise: str = Field(
        ...,
        description="Nom du modèle utilisé pour le matchmaking.",
    )


# ---------------------------------------------------------------------------
# 4. Extraction de concepts (Graphe Cognitif)
# ---------------------------------------------------------------------------

class GraphRequest(BaseModel):
    """Corps de la requête pour l'extraction de concepts d'un texte de cours."""

    texte_cours: str = Field(
        ...,
        min_length=10,
        description="Texte du cours à analyser pour en extraire les concepts clés.",
        examples=["La récursivité est une technique où une fonction s'appelle elle-même."],
    )
    langue: str = Field(
        default="fr",
        description="Code de langue du texte ('fr' pour français, 'en' pour anglais).",
        examples=["fr"],
    )


class GraphNode(BaseModel):
    """Représente un nœud (concept) dans le graphe cognitif."""

    id: str = Field(..., description="Identifiant unique du nœud.")
    label: str = Field(..., description="Libellé du concept extrait.")
    type: str = Field(..., description="Type d'entité : 'CONCEPT', 'TECHNOLOGIE', 'PERSONNE', etc.")


class GraphRelation(BaseModel):
    """Représente une relation (arc) entre deux nœuds du graphe cognitif."""

    source: str = Field(..., description="ID du nœud source.")
    cible: str = Field(..., description="ID du nœud cible.")
    relation: str = Field(..., description="Type de relation, ex: 'EST_LIÉ_À', 'UTILISE'.")


class GraphResponse(BaseModel):
    """Corps de la réponse de l'extraction de concepts."""

    noeuds: list[GraphNode] = Field(..., description="Liste des nœuds (concepts) extraits.")
    relations: list[GraphRelation] = Field(..., description="Liste des relations entre les concepts.")
    meta: dict[str, Any] = Field(
        default_factory=dict,
        description="Métadonnées supplémentaires (ex: nombre de tokens traités).",
    )
