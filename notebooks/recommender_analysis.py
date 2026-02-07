"""
Système de Recommandation de Films - MovieLens 100k
====================================================

Ce script implémente un système de recommandation complet avec visualisations interactives.
Compatible avec l'application Streamlit du projet.

Modèles utilisés:
- Baseline (Moyenne Globale)
- KNN (Item-Based Collaborative Filtering)
- SVD (Singular Value Decomposition)
- NMF (Non-Negative Matrix Factorization)

Note: LightFM n'est pas inclus car incompatible avec cet environnement Windows.
"""

# %% [markdown]
# # 1. Imports et Configuration

# %%
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pickle
import os
import time
import json
from pathlib import Path

# Scikit-Learn
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.decomposition import TruncatedSVD, NMF
from sklearn.metrics.pairwise import cosine_similarity
from scipy.sparse import csr_matrix

# Configuration
import warnings
warnings.filterwarnings('ignore')

print("✅ Imports réussis!")

# %% [markdown]
# # 2. Chargement des Données

# %%
# Chemins
DATA_DIR = Path("../ml-100k")
MODELS_DIR = Path("../models")
MODELS_DIR.mkdir(exist_ok=True)

# Charger les ratings
ratings = pd.read_csv(
    DATA_DIR / "u.data",
    sep='\t',
    names=['user_id', 'item_id', 'rating', 'timestamp'],
    engine='python'
)

# Charger les films
movies_cols = ['movie_id', 'title', 'release_date', 'video_release_date', 'imdb_url']
genre_cols = ['unknown', 'Action', 'Adventure', 'Animation', 'Childrens', 'Comedy', 'Crime',
              'Documentary', 'Drama', 'Fantasy', 'Film-Noir', 'Horror', 'Musical',
              'Mystery', 'Romance', 'Sci-Fi', 'Thriller', 'War', 'Western']
movies_cols.extend(genre_cols)

movies = pd.read_csv(
    DATA_DIR / "u.item",
    sep='|',
    names=movies_cols,
    encoding='latin-1',
    engine='python',
    index_col=False
)

print(f"📊 Données chargées:")
print(f"  - Ratings: {len(ratings):,} lignes")
print(f"  - Films: {len(movies):,} lignes")
print(f"  - Utilisateurs: {ratings['user_id'].nunique():,}")
print(f"  - Note moyenne: {ratings['rating'].mean():.2f}")

# %% [markdown]
# # 3. Exploration des Données (EDA)

# %% [markdown]
# ## 3.1 Distribution des Notes

# %%
# Distribution des notes
rating_counts = ratings['rating'].value_counts().sort_index()

fig = make_subplots(
    rows=1, cols=2,
    subplot_titles=("Distribution des Notes", "Répartition en %"),
    specs=[[{"type": "bar"}, {"type": "pie"}]]
)

# Bar chart
fig.add_trace(
    go.Bar(x=rating_counts.index, y=rating_counts.values, 
           marker_color='#E50914', name='Fréquence'),
    row=1, col=1
)

# Pie chart
fig.add_trace(
    go.Pie(labels=rating_counts.index, values=rating_counts.values,
           marker_colors=px.colors.sequential.Reds_r),
    row=1, col=2
)

fig.update_layout(
    template="plotly_dark",
    height=400,
    showlegend=False,
    title_text="Analyse de la Distribution des Notes"
)

fig.show()

# %% [markdown]
# ## 3.2 Films les Plus Populaires

# %%
# Top 10 films les plus notés
movie_stats = ratings.groupby('item_id').agg({
    'rating': ['count', 'mean']
}).reset_index()
movie_stats.columns = ['movie_id', 'count', 'mean']

# movie_id is already int from the CSV, just merge
movie_stats = movie_stats.merge(movies[['movie_id', 'title']], on='movie_id')

top_movies = movie_stats.nlargest(10, 'count')

fig = px.bar(
    top_movies,
    x='count',
    y='title',
    orientation='h',
    title="Top 10 Films les Plus Notés",
    labels={'count': 'Nombre de votes', 'title': 'Film'},
    color='mean',
    color_continuous_scale='RdYlGn',
    hover_data=['mean']
)

fig.update_layout(
    template="plotly_dark",
    yaxis={'categoryorder': 'total ascending'},
    height=500
)

fig.show()

# %% [markdown]
# ## 3.3 Analyse des Genres

# %%
# Popularité des genres
genre_counts = movies[genre_cols].sum().sort_values(ascending=False)

fig = px.bar(
    x=genre_counts.index,
    y=genre_counts.values,
    title="Nombre de Films par Genre",
    labels={'x': 'Genre', 'y': 'Nombre de films'},
    color=genre_counts.values,
    color_continuous_scale='Reds'
)

fig.update_layout(
    template="plotly_dark",
    xaxis_tickangle=-45,
    showlegend=False,
    height=500
)

fig.show()

# %% [markdown]
# # 4. Préparation des Données

# %%
# Split train/test
train_data, test_data = train_test_split(ratings, test_size=0.2, random_state=42)

print(f"📦 Split des données:")
print(f"  - Train: {len(train_data):,} ratings")
print(f"  - Test: {len(test_data):,} ratings")

# Créer la matrice user-item (sparse)
def create_user_item_matrix(data):
    """Crée une matrice user-item sparse"""
    user_ids = data['user_id'].unique()
    item_ids = data['item_id'].unique()
    
    user_id_map = {uid: idx for idx, uid in enumerate(sorted(user_ids))}
    item_id_map = {iid: idx for idx, iid in enumerate(sorted(item_ids))}
    
    rows = data['user_id'].map(user_id_map)
    cols = data['item_id'].map(item_id_map)
    values = data['rating'].values
    
    matrix = csr_matrix(
        (values, (rows, cols)),
        shape=(len(user_ids), len(item_ids))
    )
    
    return matrix, user_id_map, item_id_map

train_matrix, user_map, item_map = create_user_item_matrix(train_data)
print(f"✅ Matrice créée: {train_matrix.shape} (sparsity: {1 - train_matrix.nnz / (train_matrix.shape[0] * train_matrix.shape[1]):.2%})")

# %% [markdown]
# # 5. Entraînement des Modèles

# %% [markdown]
# ## 5.1 Baseline (Moyenne Globale)

# %%
print("🔄 Entraînement du modèle Baseline...")
start_time = time.time()

global_mean = train_data['rating'].mean()

# Prédictions sur test
test_predictions_baseline = np.full(len(test_data), global_mean)
rmse_baseline = np.sqrt(mean_squared_error(test_data['rating'], test_predictions_baseline))
mae_baseline = mean_absolute_error(test_data['rating'], test_predictions_baseline)

baseline_time = time.time() - start_time

print(f"✅ Baseline terminé en {baseline_time:.2f}s")
print(f"   RMSE: {rmse_baseline:.4f}, MAE: {mae_baseline:.4f}")

# %% [markdown]
# ## 5.2 SVD (Matrix Factorization)

# %%
print("🔄 Entraînement du modèle SVD...")
start_time = time.time()

svd_model = TruncatedSVD(n_components=50, random_state=42)
user_factors = svd_model.fit_transform(train_matrix)
item_factors = svd_model.components_.T

# Prédictions
def predict_svd(user_id, item_id, user_factors, item_factors, user_map, item_map, global_mean):
    if user_id not in user_map or item_id not in item_map:
        return global_mean
    u_idx = user_map[user_id]
    i_idx = item_map[item_id]
    pred = np.dot(user_factors[u_idx], item_factors[i_idx])
    return np.clip(pred, 1, 5)

test_predictions_svd = [
    predict_svd(row['user_id'], row['item_id'], user_factors, item_factors, user_map, item_map, global_mean)
    for _, row in test_data.iterrows()
]

rmse_svd = np.sqrt(mean_squared_error(test_data['rating'], test_predictions_svd))
mae_svd = mean_absolute_error(test_data['rating'], test_predictions_svd)
svd_time = time.time() - start_time

print(f"✅ SVD terminé en {svd_time:.2f}s")
print(f"   RMSE: {rmse_svd:.4f}, MAE: {mae_svd:.4f}")

# %% [markdown]
# ## 5.3 NMF (Non-Negative Matrix Factorization)

# %%
print("🔄 Entraînement du modèle NMF...")
start_time = time.time()

nmf_model = NMF(n_components=50, init='random', random_state=42, max_iter=200)
user_factors_nmf = nmf_model.fit_transform(train_matrix)
item_factors_nmf = nmf_model.components_.T

# Prédictions
test_predictions_nmf = [
    predict_svd(row['user_id'], row['item_id'], user_factors_nmf, item_factors_nmf, user_map, item_map, global_mean)
    for _, row in test_data.iterrows()
]

rmse_nmf = np.sqrt(mean_squared_error(test_data['rating'], test_predictions_nmf))
mae_nmf = mean_absolute_error(test_data['rating'], test_predictions_nmf)
nmf_time = time.time() - start_time

print(f"✅ NMF terminé en {nmf_time:.2f}s")
print(f"   RMSE: {rmse_nmf:.4f}, MAE: {mae_nmf:.4f}")

# %% [markdown]
# ## 5.4 KNN (Item-Based Collaborative Filtering)

# %%
print("🔄 Calcul de la similarité cosinus (KNN)...")
start_time = time.time()

# Calculer la similarité item-item
item_similarity = cosine_similarity(train_matrix.T)

# Prédictions KNN
def predict_knn(user_id, item_id, train_data, item_similarity, user_map, item_map, global_mean, k=20):
    if user_id not in user_map or item_id not in item_map:
        return global_mean
    
    u_idx = user_map[user_id]
    i_idx = item_map[item_id]
    
    # Trouver les items similaires que l'utilisateur a notés
    user_ratings = train_matrix[u_idx].toarray().flatten()
    rated_items = np.where(user_ratings > 0)[0]
    
    if len(rated_items) == 0:
        return global_mean
    
    # Similarités avec l'item cible
    sims = item_similarity[i_idx, rated_items]
    
    # Top-k items similaires
    top_k_idx = np.argsort(sims)[-k:]
    top_k_sims = sims[top_k_idx]
    top_k_ratings = user_ratings[rated_items[top_k_idx]]
    
    if np.sum(np.abs(top_k_sims)) == 0:
        return global_mean
    
    pred = np.sum(top_k_sims * top_k_ratings) / np.sum(np.abs(top_k_sims))
    return np.clip(pred, 1, 5)

test_predictions_knn = [
    predict_knn(row['user_id'], row['item_id'], train_data, item_similarity, user_map, item_map, global_mean)
    for _, row in test_data.iterrows()
]

rmse_knn = np.sqrt(mean_squared_error(test_data['rating'], test_predictions_knn))
mae_knn = mean_absolute_error(test_data['rating'], test_predictions_knn)
knn_time = time.time() - start_time

print(f"✅ KNN terminé en {knn_time:.2f}s")
print(f"   RMSE: {rmse_knn:.4f}, MAE: {mae_knn:.4f}")

# %% [markdown]
# # 6. Comparaison des Modèles

# %%
# Créer un DataFrame de résultats
results = pd.DataFrame({
    'algorithm': ['Global Mean', 'SVD (Sklearn)', 'NMF (Sklearn)', 'KNN (Item-Based)'],
    'rmse': [rmse_baseline, rmse_svd, rmse_nmf, rmse_knn],
    'mae': [mae_baseline, mae_svd, mae_nmf, mae_knn],
    'training_time': [baseline_time, svd_time, nmf_time, knn_time]
})

print("\n📊 Résultats de Comparaison:")
print(results.to_string(index=False))

# %% [markdown]
# ## 6.1 Visualisation: RMSE vs Temps d'Entraînement

# %%
fig = px.scatter(
    results,
    x='training_time',
    y='rmse',
    color='algorithm',
    size='training_time',
    hover_data=['mae'],
    text='algorithm',
    title="Compromis Précision vs Vitesse",
    labels={'training_time': 'Temps (s)', 'rmse': 'Erreur RMSE (Plus bas = Meilleur)'}
)

fig.update_traces(textposition='top center', textfont_size=10)
fig.update_layout(
    template="plotly_dark",
    height=500,
    margin=dict(l=50, r=50, t=80, b=50)
)

fig.show()

# %% [markdown]
# ## 6.2 Radar Chart: Comparaison Multi-Critères

# %%
# Normaliser les métriques pour le radar chart
results_norm = results.copy()
results_norm['rmse_norm'] = 1 - (results_norm['rmse'] - results_norm['rmse'].min()) / (results_norm['rmse'].max() - results_norm['rmse'].min())
results_norm['mae_norm'] = 1 - (results_norm['mae'] - results_norm['mae'].min()) / (results_norm['mae'].max() - results_norm['mae'].min())
results_norm['speed_norm'] = 1 - (results_norm['training_time'] - results_norm['training_time'].min()) / (results_norm['training_time'].max() - results_norm['training_time'].min())

fig = go.Figure()

for idx, row in results_norm.iterrows():
    fig.add_trace(go.Scatterpolar(
        r=[row['rmse_norm'], row['mae_norm'], row['speed_norm']],
        theta=['Précision (RMSE)', 'Précision (MAE)', 'Vitesse'],
        fill='toself',
        name=row['algorithm']
    ))

fig.update_layout(
    polar=dict(radialaxis=dict(visible=True, range=[0, 1])),
    template="plotly_dark",
    title="Comparaison Multi-Critères (Normalisé)",
    height=500
)

fig.show()

# %% [markdown]
# # 7. Analyse Avancée: Espace Latent (SVD)

# %% [markdown]
# ## 7.1 Visualisation 3D de l'Espace Latent

# %%
from sklearn.decomposition import PCA

# Réduire à 3D pour visualisation
pca = PCA(n_components=3)
item_factors_3d = pca.fit_transform(item_factors)

# Ajouter les genres dominants
item_ids_sorted = sorted(item_map.keys())
genres_for_items = []
for iid in item_ids_sorted:
    movie_row = movies[movies['movie_id'] == iid]
    if len(movie_row) > 0:
        genre_values = movie_row[genre_cols].values[0]
        if genre_values.sum() > 0:
            dominant_genre = genre_cols[np.argmax(genre_values)]
        else:
            dominant_genre = 'Unknown'
    else:
        dominant_genre = 'Unknown'
    genres_for_items.append(dominant_genre)

# Créer le DataFrame pour Plotly
df_3d = pd.DataFrame({
    'x': item_factors_3d[:, 0],
    'y': item_factors_3d[:, 1],
    'z': item_factors_3d[:, 2],
    'genre': genres_for_items
})

fig = px.scatter_3d(
    df_3d,
    x='x', y='y', z='z',
    color='genre',
    title="Espace Latent SVD (3D) - Coloré par Genre",
    labels={'x': 'Composante 1', 'y': 'Composante 2', 'z': 'Composante 3'}
)

fig.update_layout(
    template="plotly_dark",
    height=700
)

fig.show()

# %% [markdown]
# ## 7.2 Heatmap: Corrélation Genre-Facteur

# %%
# Calculer la corrélation entre genres et facteurs latents
genre_factor_corr = []

for genre in genre_cols[:10]:  # Top 10 genres
    genre_movies = movies[movies[genre] == 1]['movie_id'].values
    genre_indices = [item_map[mid] for mid in genre_movies if mid in item_map]
    
    if len(genre_indices) > 0:
        genre_factors = item_factors[genre_indices, :10]  # Top 10 factors
        avg_factors = genre_factors.mean(axis=0)
        genre_factor_corr.append(avg_factors)
    else:
        genre_factor_corr.append(np.zeros(10))

genre_factor_corr = np.array(genre_factor_corr)

fig = px.imshow(
    genre_factor_corr,
    x=[f'Factor {i+1}' for i in range(10)],
    y=genre_cols[:10],
    color_continuous_scale='RdBu_r',
    title="Heatmap: Corrélation Genre-Facteur Latent (SVD)",
    labels={'x': 'Facteur Latent', 'y': 'Genre', 'color': 'Intensité'}
)

fig.update_layout(
    template="plotly_dark",
    height=500
)

fig.show()

# %% [markdown]
# # 8. Export des Modèles

# %%
print("💾 Sauvegarde des modèles...")

# Sauvegarder SVD
with open(MODELS_DIR / 'svd_model.pkl', 'wb') as f:
    pickle.dump({
        'model': svd_model,
        'user_factors': user_factors,
        'item_factors': item_factors,
        'user_map': user_map,
        'item_map': item_map,
        'global_mean': global_mean
    }, f)

# Sauvegarder NMF
with open(MODELS_DIR / 'nmf_model.pkl', 'wb') as f:
    pickle.dump({
        'model': nmf_model,
        'user_factors': user_factors_nmf,
        'item_factors': item_factors_nmf,
        'user_map': user_map,
        'item_map': item_map,
        'global_mean': global_mean
    }, f)

# Sauvegarder la matrice de similarité (KNN)
with open(MODELS_DIR / 'similarity_matrix.pkl', 'wb') as f:
    pickle.dump({
        'similarity_matrix': item_similarity,
        'item_map': item_map
    }, f)

# Sauvegarder les métriques
metrics = {
    'global_mean': {
        'rmse': float(rmse_baseline),
        'mae': float(mae_baseline),
        'training_time': float(baseline_time)
    },
    'svd': {
        'rmse': float(rmse_svd),
        'mae': float(mae_svd),
        'training_time': float(svd_time)
    },
    'nmf': {
        'rmse': float(rmse_nmf),
        'mae': float(mae_nmf),
        'training_time': float(nmf_time)
    },
    'knn': {
        'rmse': float(rmse_knn),
        'mae': float(mae_knn),
        'training_time': float(knn_time)
    }
}

with open(MODELS_DIR / 'metrics.json', 'w') as f:
    json.dump(metrics, f, indent=2)

print("✅ Modèles sauvegardés dans:", MODELS_DIR)
print("\n🎉 Analyse terminée!")
print("\n📌 Résumé:")
print(f"  - Meilleur modèle (RMSE): {results.loc[results['rmse'].idxmin(), 'algorithm']}")
print(f"  - Modèle le plus rapide: {results.loc[results['training_time'].idxmin(), 'algorithm']}")
