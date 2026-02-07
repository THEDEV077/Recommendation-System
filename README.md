# Système de Recommandation de Films - Rapport Technique

Ce document sert de référence technique détaillée pour la génération de rapport sur le projet de système de recommandation de films.

## 1. Vue d'ensemble du projet

Ce projet implémente un système de recommandation complet pour le dataset **MovieLens 100k**. Il combine des approches classiques de filtrage collaboratif (SVD, KNN) avec des méthodes modernes d'apprentissage profond (**Neural Collaborative Filtering**, **AutoRec**).

L'application est déployée via **Streamlit** et offre une interface interactive pour :
*   Explorer les données (statistiques, distributions).
*   Recevoir des recommandations personnalisées.
*   Analyser le profil utilisateur.
*   Visualiser l'espace latent des algorithmes (PCA 2D/3D).

## 2. Architecture Technique

### Structure du Code
*   `app.py` : Point d'entrée de l'application Streamlit.
*   `benchmark.py` : Pipeline d'entraînement et d'évaluation complet. Gère le chargement des données, le split train/test, l'entraînement de tous les modèles, le calcul des métriques et la sauvegarde des artefacts.
*   `models/` : Contient les définitions des classes de modèles et les artefacts sauvegardés (`.pkl`, `.pth`, `.json`).
    *   `ncf.py` : Implémentation PyTorch du Neural Collaborative Filtering.
    *   `autorec.py` : Implémentation PyTorch de l'Autoencodeur.
*   `utils/` :
    *   `recommender.py` : Logique de chargement des modèles et de prédiction pour l'inférence temps réel.
    *   `ui.py` : Composants graphiques et styles CSS.
*   `pages/` : Pages individuelles de l'application Streamlit (Exploration, Recommandations, Profil, Analyse).

### Stack Technologique
*   **Langage** : Python 3.10+
*   **Interface Web** : Streamlit
*   **Data Science** : Pandas, NumPy, Scikit-learn, SciPy
*   **Deep Learning** : PyTorch (pour NCF et AutoRec)
*   **Visualisation** : Plotly Express (Graphiques interactifs 2D/3D)

## 3. Modèles Implémentés

### A. Modèles Classiques (Scikit-learn)

1.  **Baseline (Moyenne)** :
    *   *Principe* : Prédit la note moyenne globale pour tous les films. Sert de point de comparaison minimal.
    *   *Utilité* : Vérifier que les modèles complexes apprennent réellement quelque chose.

2.  **KNN (K-Nearest Neighbors - Item-Based)** :
    *   *Principe* : Filtrage collaboratif basé sur la similarité entre items.
    *   *Méthode* : Calcule la similarité Cosinus entre les vecteurs de notation des films. Pour un utilisateur $u$ et un film $i$, la prédiction est une moyenne pondérée des notes de $u$ sur les $k$ films les plus similaires à $i$.
    *   *Avantage* : Intuitif et efficace pour dénicher des items similaires.

3.  **SVD (Singular Value Decomposition)** :
    *   *Principe* : Factorisation matricielle. Décompose la matrice Utilisateur-Item $R$ en trois matrices $U \Sigma V^T$.
    *   *Implémentation* : Utilise `TruncatedSVD` de Scikit-learn.
    *   *Note Technique* : Cette implémentation considère les valeurs manquantes comme des 0, ce qui pénalise la performance RMSE sur ce dataset explicitement noté (1-5), mais permet de capturer la structure globale.

4.  **NMF (Non-negative Matrix Factorization)** :
    *   *Principe* : Similaire à SVD mais avec une contrainte de non-négativité sur les facteurs latents.
    *   *Intérêt* : Les facteurs sont plus interprétables (parties additives), souvent assimilables à des "genres" ou "thèmes" latents.

### B. Modèles Deep Learning (PyTorch) - *Nouvellement Implémentés*

5.  **NCF (Neural Collaborative Filtering)** :
    *   *Source* : He et al., WWW 2017.
    *   *Architecture* : Combine deux voies :
        *   **GMF (Generalized Matrix Factorization)** : Produit scalaire de vecteurs d'embedding (linéaire).
        *   **MLP (Multi-Layer Perceptron)** : Concaténation des embeddings + couches denses (non-linéaire) pour capturer des interactions complexes.
    *   *Combinaison* : Les sorties de GMF et MLP sont concaténées avant la couche finale de prédiction (NeuMF).
    *   *Entraînement* : Adam Optimizer, MSE Loss.
    *   *Performance* : **Meilleur modèle du benchmark (RMSE ~0.93)**.

6.  **AutoRec (Autoencoder Recommender)** :
    *   *Source* : Sedhain et al., WWW 2015.
    *   *Architecture* : Autoencodeur Item-based (I-AutoRec).
    *   *Entrée* : Vecteur des notes d'un item (dimension $N_{users}$), avec des 0 pour les inconnues.
    *   *Encoder* : Compresse l'entrée vers une dimension cachée (ex: 500 neurones) via une fonction Sigmoid.
    *   *Decoder* : Reconstruit le vecteur original.
    *   *Hack de l'Entraînement* : On ne rétropropage l'erreur *que* sur les notes observées (Masked MSE), ignorant les 0. Cela permet au modèle de compléter les "trous" intelligemment.
    *   *Performance* : Très compétitif (RMSE ~0.96), capture bien les relations non-linéaires entre utilisateurs.

## 4. Résultats du Benchmark

Les performances ont été mesurées sur un jeu de test (20% du dataset) :

| Modèle | RMSE (Erreur Quadratique Moyenne) | MAE (Erreur Absolue Moyenne) | Commentaires |
| :--- | :--- | :--- | :--- |
| **NCF (Neural CF)** | **0.9339** | **0.7353** | **Meilleur modèle.** Excellente précision. |
| **AutoRec** | 0.9658 | 0.7663 | Très bon, proche de NCF. |
| **KNN (Item-Based)** | 0.9692 | 0.7583 | Performance solide et robuste. |
| Baseline | 1.1239 | 0.9420 | Référence de base. |
| SVD (Sklearn) | 2.6404 | 2.3711 | Pénalisé par la gestion des 0 (sparsity). |
| NMF (Sklearn) | 2.6246 | 2.3564 | Pénalisé par la gestion des 0 (sparsity). |

**Analyse** :
Les modèles de **Deep Learning (NCF, AutoRec)** surpassent les approches classiques sur ce dataset. NCF offre la meilleure erreur de prédiction, confirmant la supériorité des réseaux de neurones pour capturer les interactions utilisateur-item subtiles.

## 5. Fonctionnalités de l'Application

1.  **Exploration** :
    *   Métriques clés (Nombre de films, utilisateurs, densité de la matrice).
    *   Distribution des notes.
    *   Nuages de mots des titres.
    *   Popularité par genre.

2.  **Recommandations** :
    *   Choix dynamique de l'algorithme (Mode Automatique vs Manuel).
    *   Simulation pour différents User IDs.
    *   Affichage des affiches de films (via API OMDb).

3.  **Analyse Algorithmique** :
    *   **Carte des films (SVD/PCA)** : Visualisation 2D/3D interactive où la distance spatiale représente la similarité sémantique.
    *   **Interprétation (NMF)** : Découverte des "concepts cachés" (ex: un facteur peut regrouper tous les films d'horreur).
    *   **Réseau de Similarité** : Graphe des connexions entre films.

## 6. Installation et Exécution

**Prérequis** :
```bash
pip install -r requirements.txt
```
*(Contient streamlit, pandas, numpy, scikit-learn, torch, plotly, requests)*

**Lancer l'entraînement (Benchmark)** :
```bash
python benchmark.py
```
*Génère les modèles dans le dossier `models/`.*

**Lancer l'application** :
```bash
python -m streamlit run app.py
```

## 7. Conclusion

Ce projet démontre comment moderniser un système de recommandation classique. En intégrant NCF et AutoRec, nous avons réduit l'erreur de prédiction (RMSE) de manière significative par rapport à la baseline, tout en offrant une interface utilisateur riche pour l'explicabilité et l'exploration des données.