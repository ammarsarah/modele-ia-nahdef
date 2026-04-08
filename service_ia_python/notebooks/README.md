# Notebooks d'Entraînement

Ce dossier est destiné aux **Jupyter Notebooks** utilisés pour l'exploration des données et
l'entraînement des modèles de Machine Learning.

## Contenu prévu

- `01_exploration_donnees.ipynb` : Analyse exploratoire des données étudiantes
- `02_entrainement_prediction_reussite.ipynb` : Entraînement du modèle Random Forest pour prédire
  la réussite académique
- `03_entrainement_matchmaking.ipynb` : Entraînement du modèle K-Means / KNN pour le regroupement
  d'étudiants similaires
- `04_test_nlp_spacy.ipynb` : Expérimentation avec spaCy pour l'extraction d'entités et de concepts

## Workflow

1. Entraîner le modèle dans un notebook.
2. Sauvegarder le modèle entraîné au format `.pkl` dans le dossier `../saved_models/` à l'aide de
   `joblib.dump(model, "../saved_models/nom_du_modele.pkl")`.
3. Le service `app/services/ml_service.py` chargera automatiquement le fichier `.pkl` au démarrage.
