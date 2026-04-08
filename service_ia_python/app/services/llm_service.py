"""
app/services/llm_service.py

Service LLM (Large Language Model) avec architecture RAG (Retrieval-Augmented Generation).

ÉTAT ACTUEL : Simulation (mock) — les réponses sont générées de façon statique.

INTÉGRATION FUTURE :
- Remplacez la fonction `_repondre_mock()` par un vrai pipeline LangChain avec un LLM
  (OpenAI GPT-4, Mistral, LLaMA, etc.).
- Connectez une base de données vectorielle (FAISS, ChromaDB, Pinecone) pour le RAG
  afin de rechercher les documents de cours pertinents avant de générer la réponse.
- Définissez OPENAI_API_KEY (ou MISTRAL_API_KEY) dans un fichier .env à la racine du projet.
"""

from __future__ import annotations

import logging
import os

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Base de connaissances fictive pour la simulation
# ---------------------------------------------------------------------------

_BASE_CONNAISSANCES: dict[str, dict[str, str]] = {
    "récursivité": {
        "answer": (
            "La récursivité est une technique de programmation où une fonction s'appelle "
            "elle-même jusqu'à atteindre un cas de base. "
            "Exemple classique : le calcul de la factorielle (n! = n × (n-1)!)."
        ),
        "source": "Chapitre 5 – Algorithmes et récursivité",
    },
    "algorithme": {
        "answer": (
            "Un algorithme est une suite d'instructions précises et finies permettant de "
            "résoudre un problème. Il se caractérise par son entrée, sa sortie, sa finitude "
            "et sa correction."
        ),
        "source": "Chapitre 1 – Introduction aux algorithmes",
    },
    "machine learning": {
        "answer": (
            "Le Machine Learning (apprentissage automatique) est un sous-domaine de "
            "l'intelligence artificielle dans lequel les modèles apprennent des patterns "
            "à partir de données sans être explicitement programmés."
        ),
        "source": "Module IA – Introduction au Machine Learning",
    },
}

_REPONSE_DEFAUT = (
    "Je n'ai pas trouvé de réponse précise dans ma base de connaissances pour votre question. "
    "Veuillez reformuler ou consulter vos supports de cours."
)


# ---------------------------------------------------------------------------
# Fonction principale exposée aux routeurs
# ---------------------------------------------------------------------------

def repondre_question(question: str, student_id: str) -> dict[str, object]:
    """
    Génère une réponse à la question d'un étudiant.

    SIMULATION : recherche par mot-clé dans une base fictive.

    TODO – Intégration réelle :
    1. Vectoriser `question` avec un modèle d'embeddings (ex: OpenAIEmbeddings).
    2. Requêter une base de données vectorielle (FAISS / ChromaDB) pour récupérer
       les k documents les plus pertinents.
    3. Construire un prompt LangChain (RetrievalQA) et appeler le LLM :
           from langchain.chains import RetrievalQA
           from langchain_openai import ChatOpenAI
           llm = ChatOpenAI(model="gpt-4o", api_key=os.getenv("OPENAI_API_KEY"))
           qa_chain = RetrievalQA.from_chain_type(llm=llm, retriever=vectorstore.as_retriever())
           result = qa_chain.invoke({"query": question})
    4. Retourner `result["result"]` et les documents sources.

    Args:
        question: La question posée par l'étudiant.
        student_id: L'identifiant de l'étudiant (pour la personnalisation future).

    Returns:
        Dictionnaire contenant `answer` et `source_documents`.
    """
    logger.info("Traitement de la question de l'étudiant '%s' : %s", student_id, question)

    question_lower = question.lower()
    source_documents: list[str] = []
    answer = _REPONSE_DEFAUT

    for mot_cle, contenu in _BASE_CONNAISSANCES.items():
        if mot_cle in question_lower:
            answer = contenu["answer"]
            source_documents.append(contenu["source"])
            break

    return {
        "answer": answer,
        "source_documents": source_documents,
    }
