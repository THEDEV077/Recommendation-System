import streamlit as st
import pandas as pd
import numpy as np
import pickle
import os
import hashlib
import requests
import re
from concurrent.futures import ThreadPoolExecutor

@st.cache_data(ttl=86400) # Cache posters for 24h
def get_movie_poster(title, movie_id=None, api_key="c9d2f347"):
    """
    Récupère l'affiche d'un film. 
    Cherche d'abord en local (assets/posters/), puis via l'API OMDb.
    """
    # 1. Tenter le chargement local si movie_id est fourni
    if movie_id:
        local_path = os.path.join("assets", "posters", f"movie_{movie_id}.jpg")
        if os.path.exists(local_path):
            return local_path

    # 2. Sinon, tenter via API OMDb
    try:
        # Extraire l'année si présente (ex: "Toy Story (1995)")
        year_match = re.search(r'\((\d{4})\)', title)
        year = year_match.group(1) if year_match else None
        
        # Nettoyer le titre
        clean_title = re.sub(r'\s*\(\d{4}\)', '', title).strip()
        if ", The" in clean_title: clean_title = "The " + clean_title.replace(", The", "")
        if ", A" in clean_title: clean_title = "A " + clean_title.replace(", A", "")
        
        params = {
            't': clean_title,
            'apikey': api_key,
            'type': 'movie'
        }
        if year:
            params['y'] = year
            
        response = requests.get("https://www.omdbapi.com/", params=params, timeout=3)
        if response.status_code == 200:
            data = response.json()
            poster_url = data.get('Poster')
            if data.get('Response') == 'True' and poster_url and poster_url != 'N/A':
                # Optionnel: On pourrait sauvegarder ici pour le prochain coup
                # Mais on laisse le script de téléchargement massif gérer ça
                return poster_url
    except Exception:
        pass
        
    # Fallback sur un placeholder stylisé
    display_title = title.replace(" ", "+")
    return f"https://placehold.co/400x600/1a1a1a/e50914?text={display_title}"

def get_posters_batch(movies_list, api_key="c9d2f347"):
    """
    Récupère les affiches en parallèle pour une liste de films.
    Input: list of dicts/tuples [{'title': '...', 'movie_id': ...}, ...]
    Output: list of poster urls in same order
    """
    def fetch_one(movie):
        return get_movie_poster(movie['title'], movie_id=movie['movie_id'], api_key=api_key)
    
    with ThreadPoolExecutor(max_workers=10) as executor:
        results = list(executor.map(fetch_one, movies_list))
    return results

@st.cache_resource
def load_data():
    """Charge les données et les modèles en cache."""
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    models_dir = os.path.join(base_dir, 'models')
    
    data = {}
    
    # 1. Charger mappings
    try:
        with open(os.path.join(models_dir, 'data_mappings.pkl'), 'rb') as f:
            mappings = pickle.load(f)
            data['movies'] = mappings['movies']
            data['ratings'] = mappings['ratings']
    except FileNotFoundError:
        st.error("Fichiers de données manquants. Veuillez exécuter le script d'entraînement.")
        return None

    # 2. Déterminer le MEILLEUR modèle via les métriques
    data['model_type'] = 'similarity' # Par défaut
    try:
        metrics_path = os.path.join(models_dir, 'model_metrics.json')
        if os.path.exists(metrics_path):
            import json
            with open(metrics_path, 'r') as f:
                metrics = json.load(f)
            best_model_info = min(metrics, key=lambda x: x['rmse'])
            data['best_model_name'] = best_model_info['algorithm']
            data['best_rmse'] = best_model_info['rmse']
            
            # Map algorithm name to internal model_type
            if 'SVD' in data['best_model_name']:
                data['model_type'] = 'svd'
            elif 'NMF' in data['best_model_name']:
                data['model_type'] = 'nmf'
            elif 'KNN' in data['best_model_name'] or 'Similarity' in data['best_model_name']:
                data['model_type'] = 'knn'
            elif 'LightFM' in data['best_model_name']:
                data['model_type'] = 'lightfm'
    except Exception:
        pass

    # 3. Charger les fichiers de poids
    try:
        with open(os.path.join(models_dir, 'similarity_matrix.pkl'), 'rb') as f:
            data['similarity_matrix'] = pickle.load(f)
    except FileNotFoundError:
        data['similarity_matrix'] = None

    # Charger SVD
    try:
        with open(os.path.join(models_dir, 'svd_model.pkl'), 'rb') as f:
            data['svd_model'] = pickle.load(f)
    except (FileNotFoundError, ModuleNotFoundError, ImportError):
        data['svd_model'] = None
        
    # Charger NMF
    try:
        with open(os.path.join(models_dir, 'nmf_model.pkl'), 'rb') as f:
            data['nmf_model'] = pickle.load(f)
    except (FileNotFoundError, ModuleNotFoundError, ImportError):
        data['nmf_model'] = None
        
    # Charger LightFM
    try:
        with open(os.path.join(models_dir, 'lightfm_model.pkl'), 'rb') as f:
            data['lightfm_model'] = pickle.load(f)
    except (FileNotFoundError, ModuleNotFoundError, ImportError):
        data['lightfm_model'] = None

    # Charger NCF
    try:
        ncf_path = os.path.join(models_dir, 'ncf_model.pth')
        if os.path.exists(ncf_path):
            import torch
            from models.ncf import NCF
            n_users = data['ratings']['user_id'].max()
            n_items = data['ratings']['item_id'].max()
            
            # Using default architecture from benchmark.py/ncf.py
            model = NCF(n_users, n_items) 
            model.load_state_dict(torch.load(ncf_path, map_location=torch.device('cpu')))
            model.eval()
            data['ncf_model'] = model
    except Exception as e:
        print(f"Error loading NCF: {e}")
        data['ncf_model'] = None

    # Charger AutoRec
    try:
        autorec_path = os.path.join(models_dir, 'autorec_model.pth')
        if os.path.exists(autorec_path):
            import torch
            from models.autorec import AutoRec
            n_users = data['ratings']['user_id'].max()
            
            # AutoRec needs num_inputs = n_users
            model = AutoRec(num_inputs=n_users)
            model.load_state_dict(torch.load(autorec_path, map_location=torch.device('cpu')))
            model.eval()
            data['autorec_model'] = model
    except Exception as e:
        print(f"Error loading AutoRec: {e}")
        data['autorec_model'] = None
        
    return data

def get_movie_title(movie_id, movies_df):
    """Retourne le titre d'un film."""
    try:
        return movies_df[movies_df['movie_id'] == movie_id]['title'].values[0]
    except IndexError:
        return "Unknown Title"

def get_recommendations_user(user_id, data, n=10, force_model=None):
    """Génère des recommandations pour un utilisateur."""
    ratings = data['ratings']
    movies = data['movies']
    
    # Choisir le modèle (forcé ou automatique)
    model_type = force_model if force_model else data.get('model_type', 'similarity')
    
    # Films déjà vus
    user_ratings = ratings[ratings['user_id'] == user_id]
    seen_movies = set(user_ratings['item_id'].values)
    all_movies = movies['movie_id'].unique()
    unseen_movies = [m for m in all_movies if m not in seen_movies]
    
    recommendations = []
    
    # --- SVD ---
    if model_type == 'svd' and data.get('svd_model'):
        model_data = data['svd_model']
        if isinstance(model_data, dict) and 'user_features' in model_data:
            user_features = model_data['user_features']
            item_features = model_data['item_features']
            user_index = model_data['user_index']
            item_index = model_data['item_index']
            
            if user_id in user_index:
                u_idx = user_index.get_loc(user_id)
                # Calculer les scores pour TOUS les items d'un coup
                # item_features shape: (n_items, n_components), user_features[u_idx]: (n_components,)
                scores = np.dot(item_features, user_features[u_idx])
                
                # Mapper les scores aux movie_ids
                for i, movie_id in enumerate(item_index):
                    if movie_id in unseen_movies:
                        recommendations.append((movie_id, scores[i]))
            else:
                return get_popular_movies(data, n)
    
    # --- NMF ---
    elif model_type == 'nmf' and data.get('nmf_model'):
        model_data = data['nmf_model']
        if isinstance(model_data, dict) and 'user_features' in model_data:
            user_features = model_data['user_features']
            item_features = model_data['item_features']
            user_index = model_data['user_index']
            item_index = model_data['item_index']
            
            if user_id in user_index:
                u_idx = user_index.get_loc(user_id)
                scores = np.dot(item_features, user_features[u_idx])
                for i, movie_id in enumerate(item_index):
                    if movie_id in unseen_movies:
                        recommendations.append((movie_id, scores[i]))
            else:
                return get_popular_movies(data, n)

    # --- NCF (Deep Learning) ---
    elif model_type == 'ncf' and data.get('ncf_model'):
        import torch
        model = data['ncf_model']
        # Prepare inputs
        u_idx = user_id - 1 # 1-based to 0-based
        if u_idx < model.num_users:
             # We need to map unseen movies (movie_id) to internal indices (0-based)
             # Assumption: item_id in ratings corresponds to movie_id in movies 
             # and items are 1-indexed in file but 0-indexed in embedding
             
             # Filter unseen movies to those within range
             valid_unseen = [m for m in unseen_movies if (m-1) < model.num_items]
             
             if valid_unseen:
                 user_indices = torch.tensor([u_idx] * len(valid_unseen), dtype=torch.long)
                 item_indices = torch.tensor([m-1 for m in valid_unseen], dtype=torch.long)
                 
                 with torch.no_grad():
                     predictions = model(user_indices, item_indices)
                     
                 # Determine if predictions is scalar (0-d), 1-d or what
                 # model returns output.squeeze()
                 scores = predictions.numpy()
                 # If only 1 item, scores might be 0-d array
                 if scores.ndim == 0:
                     scores = [scores.item()]
                 
                 for i, movie_id in enumerate(valid_unseen):
                     recommendations.append((movie_id, float(scores[i])))


    # --- AutoRec (Deep Learning) ---
    elif model_type == 'autorec' and data.get('autorec_model'):
        import torch
        model = data['autorec_model']
        
        # We need to construct the input vector for ALL items
        # Input for item i is the vector of ratings from all users for item i
        # Shape: (n_items, n_users)
        
        n_users = model.encoder.in_features
        # We need the max item id to know size
        n_items = movies['movie_id'].max() # Approx
        
        # Build Sparse Matrix equivalent
        # For efficiency, we can just build standard numpy array if it fits in memory (it does for 100k)
        # But we only need predictions for `unseen_movies`? 
        # No, the model takes an item vector and reconstructs it. 
        # Each item is processed independently (or in batches).
        # So for a specific unseen movie m, we need its ratings from ALL users.
        
        # Optimization: Only build for Unseen Movies
        # We need ratings for these movies from data['ratings']
        # Filter ratings for unseen movies
        relevant_ratings = ratings[ratings['item_id'].isin(unseen_movies)]
        
        # Create vectors
        # mapping: movie_id -> index in unseens
        unseen_map = {mid: i for i, mid in enumerate(unseen_movies)}
        
        # Matrix (n_unseen, n_users)
        X = np.zeros((len(unseen_movies), n_users))
        
        # Fill matrix
        # user_ids are 1-based, need 0-based
        for _, row in relevant_ratings.iterrows():
            m_id = row['item_id']
            u_id = int(row['user_id']) - 1
            if u_id < n_users:
                 X[unseen_map[m_id], u_id] = row['rating']
                 
        # Predict
        X_tensor = torch.FloatTensor(X)
        with torch.no_grad():
            reconstructed = model(X_tensor) # (n_unseen, n_users)
            
        # We want the score for OUR user (user_id)
        target_u_idx = user_id - 1
        if target_u_idx < n_users:
            user_scores = reconstructed[:, target_u_idx].numpy()
            
            for i, movie_id in enumerate(unseen_movies):
                recommendations.append((movie_id, float(user_scores[i])))
        else:
             return get_popular_movies(data, n)

    # --- LIGHTFM ---
    elif model_type == 'lightfm' and data.get('lightfm_model'):
        lfm_data = data['lightfm_model']
        model = lfm_data['model']
        dataset = lfm_data['dataset']
        
        user_id_map, _, item_id_map, _ = dataset.mapping()
        
        if user_id in user_id_map:
            u_idx = user_id_map[user_id]
            # Predict for all unseen items
            # We need internal indices for unseen items
            item_indices = []
            valid_movie_ids = []
            
            for m_id in unseen_movies:
                if m_id in item_id_map:
                    item_indices.append(item_id_map[m_id])
                    valid_movie_ids.append(m_id)
            
            if item_indices:
                scores = model.predict(np.array([u_idx] * len(item_indices)), np.array(item_indices))
                for i, score in enumerate(scores):
                    recommendations.append((valid_movie_ids[i], score))
        else:
            return get_popular_movies(data, n)

    # --- KNN (Item-Based Similarity) ---
    elif (model_type == 'knn' or model_type == 'similarity') and data['similarity_matrix'] is not None:
        # Fallback: Basé sur les films les mieux notés par l'utilisateur
        liked_movies = user_ratings[user_ratings['rating'] >= 4]['item_id'].values
        
        sim_scores = {}
        if len(liked_movies) > 0:
            sim_matrix = data['similarity_matrix']
            # Optimization: check intersection only once
            valid_liked = [m for m in liked_movies if m in sim_matrix.columns]
            
            if valid_liked:
                # Pre-calculate mean of relevant columns for all rows
                # This can be heavy if matrix is huge, but fine for ML-100k
                # A better way is to iterate over unseen if smaller
                for movie_id in unseen_movies:
                    if movie_id in sim_matrix.index:
                        score = sim_matrix.loc[movie_id, valid_liked].mean()
                        if score > 0: # Only relevant scores
                            sim_scores[movie_id] = score
            
            recommendations = [(k, v) for k, v in sim_scores.items()]
        else:
             return get_popular_movies(data, n)
             
    else:
        return get_popular_movies(data, n)

    # Si aucune recommandation (cas rares), fallback populaire
    if not recommendations:
        return get_popular_movies(data, n)

    # Trier et formater
    recommendations.sort(key=lambda x: x[1], reverse=True)
    top_recs = recommendations[:n]
    
    results = []
    for movie_id, score in top_recs:
        results.append({
            'movie_id': movie_id,
            'title': get_movie_title(movie_id, movies),
            'score': round(score, 2)
        })
        
    return pd.DataFrame(results)

def get_popular_movies(data, n=10):
    """Retourne les films les plus populaires."""
    ratings = data['ratings']
    movies = data['movies']
    
    stats = ratings.groupby('item_id').agg({'rating': ['count', 'mean']})
    stats.columns = ['count', 'mean']
    # Filter minimum ratings
    stats = stats[stats['count'] > 50]
    stats = stats.sort_values(by='mean', ascending=False).head(n)
    
    results = []
    for movie_id in stats.index:
        results.append({
            'movie_id': movie_id,
            'title': get_movie_title(movie_id, movies),
            'score': round(stats.loc[movie_id, 'mean'], 2)
        })
    return pd.DataFrame(results)

def get_similar_movies(movie_id, data, n=5):
    """Trouve les films similaires."""
    sim_matrix = data.get('similarity_matrix')
    movies = data['movies']
    
    if sim_matrix is None:
        return pd.DataFrame()
        
    if movie_id not in sim_matrix.index:
        return pd.DataFrame()
        
    # Get similarities
    sim_scores = sim_matrix[movie_id].sort_values(ascending=False)
    
    # Exclude itself (first one is usually itself with 1.0)
    sim_scores = sim_scores.iloc[1:n+1]
    
    results = []
    for m_id, score in sim_scores.items():
        results.append({
            'movie_id': m_id,
            'title': get_movie_title(m_id, movies),
            'score': round(score, 2)
        })
        
    return pd.DataFrame(results)
