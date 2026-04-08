"""
app/services/nlp_service.py

Service NLP pour l'extraction de concepts et de relations à partir d'un texte de cours.
Les résultats alimentent le graphe cognitif affiché côté Angular.

ÉTAT ACTUEL :
  - Tentative de chargement d'un modèle spaCy réel (fr_core_news_sm ou en_core_web_sm).
  - Si aucun modèle spaCy n'est installé, un extracteur basé sur des règles simples
    (regex + mots-clés) prend le relais afin que l'API reste fonctionnelle.

INTÉGRATION FUTURE :
  1. Installez un modèle spaCy adapté à votre langue :
         python -m spacy download fr_core_news_sm   # Français
         python -m spacy download en_core_web_sm    # Anglais
  2. Pour des relations plus riches, envisagez un modèle de Relation Extraction
     fine-tuné (ex: via spaCy + Prodigy, ou un modèle HuggingFace).
  3. Vous pouvez également utiliser LangChain avec un LLM pour extraire
     des triplets (sujet, relation, objet) directement depuis le texte :
         "Extrait tous les concepts et leurs relations dans ce texte : {texte}"
"""

from __future__ import annotations

import logging
import re
import uuid
from typing import Any

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Chargement du modèle spaCy (avec fallback)
# ---------------------------------------------------------------------------

_nlp_model = None
_MODELE_SPACY_UTILISE = "aucun (mode règles simples)"

_MODELES_SPACY_A_ESSAYER = ["fr_core_news_sm", "fr_core_news_md", "en_core_web_sm"]

for _nom_modele in _MODELES_SPACY_A_ESSAYER:
    try:
        import spacy  # noqa: PLC0415
        _nlp_model = spacy.load(_nom_modele)
        _MODELE_SPACY_UTILISE = _nom_modele
        logger.info("Modèle spaCy '%s' chargé avec succès.", _nom_modele)
        break
    except Exception:  # noqa: BLE001
        logger.debug("Modèle spaCy '%s' non disponible.", _nom_modele)

if _nlp_model is None:
    logger.warning(
        "Aucun modèle spaCy trouvé. Utilisation de l'extracteur basé sur des règles simples. "
        "Installez un modèle : python -m spacy download fr_core_news_sm"
    )


# ---------------------------------------------------------------------------
# Extraction via spaCy (NER)
# ---------------------------------------------------------------------------

def _extraire_avec_spacy(texte: str) -> tuple[list[dict[str, str]], list[dict[str, str]]]:
    """
    Utilise spaCy pour extraire les entités nommées du texte.

    Retourne un tuple (noeuds, relations).

    TODO – Améliorations futures :
      - Utiliser un modèle de dependency parsing pour extraire des relations syntaxiques.
      - Fine-tuner le modèle NER avec vos propres données de cours.
    """
    assert _nlp_model is not None
    doc = _nlp_model(texte)

    noeuds: list[dict[str, str]] = []
    entite_ids: list[str] = []
    labels_entites_vus: set[str] = set()

    for ent in doc.ents:
        label_normalise = ent.text.strip()
        if label_normalise in labels_entites_vus:
            continue
        labels_entites_vus.add(label_normalise)
        noeud_id = f"node-{uuid.uuid4().hex[:8]}"
        entite_ids.append(noeud_id)
        noeuds.append({"id": noeud_id, "label": label_normalise, "type": ent.label_})

    # Génère des relations simples « EST_LIÉ_À » entre entités consécutives
    relations: list[dict[str, str]] = []
    for i in range(len(entite_ids) - 1):
        relations.append(
            {"source": entite_ids[i], "cible": entite_ids[i + 1], "relation": "EST_LIÉ_À"}
        )

    return noeuds, relations


# ---------------------------------------------------------------------------
# Extracteur de repli basé sur des règles simples (sans spaCy)
# ---------------------------------------------------------------------------

_MOTS_VIDES_FR = {
    "le", "la", "les", "un", "une", "des", "de", "du", "et", "est", "en",
    "il", "elle", "ils", "elles", "ce", "qui", "que", "à", "au", "aux",
    "par", "pour", "dans", "sur", "avec", "sans", "pas", "ne", "se", "ou",
    "où", "mais", "donc", "car", "ni", "si", "plus", "très", "tout",
}


def _extraire_avec_regles(texte: str) -> tuple[list[dict[str, str]], list[dict[str, str]]]:
    """
    Extracteur de repli : identifie les mots importants (noms, termes techniques)
    via des heuristiques simples (majuscules, longueur, mots non vides).

    Retourne un tuple (noeuds, relations).
    """
    # Tokenisation grossière : mots de 4+ lettres, non vides
    tokens_bruts = re.findall(r"\b[A-Za-zÀ-ÿ]{4,}\b", texte)
    termes_uniques: list[str] = []
    vus: set[str] = set()
    for token in tokens_bruts:
        token_lower = token.lower()
        if token_lower not in _MOTS_VIDES_FR and token_lower not in vus:
            vus.add(token_lower)
            termes_uniques.append(token)
        if len(termes_uniques) >= 20:
            break

    noeuds: list[dict[str, str]] = []
    noeud_ids: list[str] = []
    for terme in termes_uniques:
        noeud_id = f"node-{uuid.uuid4().hex[:8]}"
        noeud_ids.append(noeud_id)
        type_entite = "CONCEPT" if terme[0].isupper() else "TERME"
        noeuds.append({"id": noeud_id, "label": terme, "type": type_entite})

    relations: list[dict[str, str]] = []
    for i in range(len(noeud_ids) - 1):
        relations.append(
            {"source": noeud_ids[i], "cible": noeud_ids[i + 1], "relation": "EST_LIÉ_À"}
        )

    return noeuds, relations


# ---------------------------------------------------------------------------
# Fonction principale exposée aux routeurs
# ---------------------------------------------------------------------------

def extraire_concepts(texte: str, langue: str = "fr") -> dict[str, Any]:
    """
    Extrait les concepts clés et leurs relations depuis un texte de cours.

    Args:
        texte: Contenu textuel du cours à analyser.
        langue: Code de langue ('fr' ou 'en').

    Returns:
        Dictionnaire contenant `noeuds`, `relations` et `meta`.
    """
    logger.info(
        "Extraction de concepts (langue=%s, modèle=%s, longueur=%d chars).",
        langue,
        _MODELE_SPACY_UTILISE,
        len(texte),
    )

    if _nlp_model is not None:
        noeuds, relations = _extraire_avec_spacy(texte)
    else:
        noeuds, relations = _extraire_avec_regles(texte)

    return {
        "noeuds": noeuds,
        "relations": relations,
        "meta": {
            "modele_nlp": _MODELE_SPACY_UTILISE,
            "langue": langue,
            "nb_caracteres": len(texte),
            "nb_concepts_extraits": len(noeuds),
        },
    }
