# Guide: Utiliser recommender_analysis.py comme Notebook

Le fichier `recommender_analysis.py` est déjà formaté pour fonctionner comme un notebook Jupyter interactif!

## ✅ Option 1: Utiliser VS Code (RECOMMANDÉ)

VS Code peut exécuter les fichiers Python comme des notebooks:

1. **Ouvrir** `recommender_analysis.py` dans VS Code
2. VS Code détecte automatiquement les cellules (marquées par `# %%`)
3. **Cliquer** sur "Run Cell" au-dessus de chaque section
4. Les visualisations Plotly s'affichent directement dans VS Code!

### Avantages:
- ✅ Pas besoin d'installer Jupyter
- ✅ Exécution cellule par cellule
- ✅ Visualisations interactives
- ✅ Déjà installé sur votre machine

## Option 2: Exécution Complète

Exécuter tout le script d'un coup:

```bash
cd notebooks
python recommender_analysis.py
```

### Ce que ça fait:
1. Charge les données MovieLens
2. Crée des visualisations interactives Plotly
3. Entraîne 4 modèles (Baseline, SVD, NMF, KNN)
4. Génère des analyses avancées (3D, radar charts, heatmaps)
5. **Sauvegarde les modèles** dans `../models/`
6. **Sauvegarde les métriques** dans `../models/metrics.json`

## Option 3: Jupyter Notebook Classique

Si vous voulez vraiment un fichier `.ipynb`:

```bash
# Installer jupyter
pip install jupyter

# Lancer jupyter
jupyter notebook

# Dans le navigateur:
# 1. Créer un nouveau notebook
# 2. Copier-coller chaque section du fichier recommender_analysis.py
# 3. Les sections "# %% [markdown]" → cellules Markdown
# 4. Les sections "# %%" → cellules Code
```

## 📊 Contenu du Script

### Visualisations Incluses:
1. **Distribution des Notes** (Bar + Pie Chart)
2. **Films Populaires** (Bar Chart Horizontal)
3. **Analyse des Genres** (Bar Chart)
4. **RMSE vs Temps** (Scatter Plot)
5. **Comparaison Multi-Critères** (Radar Chart)
6. **Espace Latent 3D** (3D Scatter avec genres)
7. **Corrélation Genre-Facteur** (Heatmap)

### Modèles Entraînés:
- ✅ Baseline (Moyenne Globale)
- ✅ SVD (Matrix Factorization)
- ✅ NMF (Non-Negative Matrix Factorization)
- ✅ KNN (Item-Based CF)
- ❌ LightFM (incompatible - supprimé)

## 🎯 Recommandation

**Utilisez VS Code** - c'est la méthode la plus simple et la plus puissante!

Ouvrez simplement `recommender_analysis.py` dans VS Code et exécutez les cellules une par une.
