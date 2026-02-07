import streamlit as st
import pandas as pd
import numpy as np
import pickle
import os
import plotly.express as px
import plotly.graph_objects as go
from sklearn.decomposition import PCA
from utils.recommender import load_data
from utils.ui import load_custom_css, render_header

st.set_page_config(page_title="Analyse Algorithmique", page_icon="🧠", layout="wide")

load_custom_css()
render_header()

st.markdown("""
Cette page permet de plonger au cœur des algorithmes pour comprendre comment ils "voient" les films.
Nous visualisons ici les **espaces latents** créés par la factorisation matricielle.
""")

# Load Models
@st.cache_resource
def load_models():
    models_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'models')
    models = {}
    
    # SVD
    svd_path = os.path.join(models_dir, 'svd_model.pkl')
    if os.path.exists(svd_path):
        with open(svd_path, 'rb') as f:
            models['svd'] = pickle.load(f)
            
    # NMF
    nmf_path = os.path.join(models_dir, 'nmf_model.pkl')
    if os.path.exists(nmf_path):
        with open(nmf_path, 'rb') as f:
            models['nmf'] = pickle.load(f)
            
    return models

models = load_models()
data = load_data()

if models and data:
    movies = data['movies']
    
    tab1, tab2, tab3 = st.tabs(["🔮 Espace Latent (SVD)", "🧩 Concepts (NMF)", "🕸️ Réseau de Similarité"])
    
    with tab1:
        st.subheader("Visualisation de l'espace des films (SVD)")
        if 'svd' in models:
            svd_model = models['svd']
            item_features = svd_model['item_features'] # (n_items, n_components)
            item_index = svd_model['item_index']
            
            # Already (n_items, n_components) - Do not transpose
            item_vectors = item_features
            
            # Filter to match movies df
            movie_map = movies.set_index('movie_id')['title'].to_dict()
            genre_cols = ['Action', 'Comedy', 'Drama', 'Horror', 'Sci-Fi']
            genre_map = movies.set_index('movie_id')[genre_cols].idxmax(axis=1).to_dict()
            
            # Create DF for Plotting
            plot_data = []
            for idx, movie_id in enumerate(item_index):
                if movie_id in movie_map:
                    # Check dimensions to avoid index error
                    if idx < item_vectors.shape[0]:
                        plot_data.append({
                            'movie_id': movie_id,
                            'title': movie_map[movie_id],
                            'genre': genre_map.get(movie_id, 'Unknown'),
                            'vec': item_vectors[idx]
                        })
            
            if len(plot_data) > 0:
                df_plot = pd.DataFrame(plot_data)
                vectors = np.stack(df_plot['vec'].values)
                
                viz_type = st.radio("Type de visualisation", ["2D (PCA)", "3D (PCA)"], horizontal=True)
                
                if viz_type == "2D (PCA)":
                    # PCA to 2D
                    pca = PCA(n_components=2)
                    components = pca.fit_transform(vectors)
                    
                    df_plot['x'] = components[:, 0]
                    df_plot['y'] = components[:, 1]
                    
                    fig = px.scatter(
                        df_plot, x='x', y='y', 
                        hover_name='title', color='genre',
                        title="Carte des films (Réduction PCA 2D)",
                        template="plotly_dark",
                        height=600
                    )
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    # PCA to 3D
                    pca = PCA(n_components=3)
                    components = pca.fit_transform(vectors)
                    
                    df_plot['x'] = components[:, 0]
                    df_plot['y'] = components[:, 1]
                    df_plot['z'] = components[:, 2]
                    
                    fig = px.scatter_3d(
                        df_plot, x='x', y='y', z='z',
                        hover_name='title', color='genre',
                        title="Carte des films (Réduction PCA 3D - Rotation interactive)",
                        template="plotly_dark",
                        height=700
                    )
                    fig.update_traces(marker=dict(size=4))
                    st.plotly_chart(fig, use_container_width=True)
                
                st.info("💡 Les points proches représentent des films considérés comme similaires par l'algorithme.")
                
                # Genre-Factor Heatmap
                st.subheader("🔥 Heatmap Genre-Facteur")
                st.markdown("Cette heatmap montre quels facteurs latents sont associés à quels genres.")
                
                # Calculate genre-factor correlation
                genre_factor_matrix = []
                for genre in genre_cols:
                    genre_movies = movies[movies[genre] == 1]['movie_id'].values
                    # Need to verify indices are valid
                    genre_indices = [i for i, mid in enumerate(item_index) if mid in genre_movies and i < len(item_vectors)]
                    if len(genre_indices) > 0:
                        genre_vectors = item_vectors[genre_indices]
                        avg_factors = genre_vectors.mean(axis=0)
                        genre_factor_matrix.append(avg_factors)
                
                if len(genre_factor_matrix) > 0:
                    genre_factor_df = pd.DataFrame(
                        genre_factor_matrix,
                        index=genre_cols,
                        columns=[f"Facteur {i}" for i in range(item_features.shape[1])]
                    )
                    
                    fig_heatmap = px.imshow(
                        genre_factor_df,
                        labels=dict(x="Facteurs Latents", y="Genres", color="Importance"),
                        title="Corrélation Genre-Facteur",
                        template="plotly_dark",
                        color_continuous_scale="Viridis"
                    )
                    st.plotly_chart(fig_heatmap, use_container_width=True)
        else:
            st.warning("Modèle SVD non trouvé.")

    with tab2:
        st.subheader("Interprétation des Facteurs Latents (NMF)")
        if 'nmf' in models:
            nmf_model = models['nmf']
            # item_features in pickle is (n_items, n_components)
            # We want (n_components, n_items) to iterate over factors
            item_features = nmf_model['item_features'].T 
            item_index = nmf_model['item_index']
            
            st.markdown("NMF force les facteurs à être positifs, ce qui les rend souvent interprétables comme des 'genres cachés' ou des 'thèmes'.")
            
            num_factors = item_features.shape[0]
            selected_factor = st.slider("Sélectionner un facteur latent (Concept)", 0, num_factors-1, 0)
            
            # Get top movies for this factor
            factor_scores = item_features[selected_factor]
            top_indices = factor_scores.argsort()[::-1][:10]
            
            top_movies = []
            movie_map = movies.set_index('movie_id')['title'].to_dict()
            
            for idx in top_indices:
                movie_id = item_index[idx]
                if movie_id in movie_map:
                    top_movies.append({'title': movie_map[movie_id], 'score': factor_scores[idx]})
            
            if top_movies:
                df_top = pd.DataFrame(top_movies)
                fig_bar = px.bar(
                    df_top, x='score', y='title', orientation='h',
                    title=f"Films représentatifs du Facteur {selected_factor}",
                    template="plotly_dark"
                )
                fig_bar.update_layout(yaxis={'categoryorder':'total ascending'})
                st.plotly_chart(fig_bar, use_container_width=True)
        else:
            st.warning("Modèle NMF non trouvé.")
    
    with tab3:
        st.subheader("🕸️ Réseau de Similarité entre Films")
        st.markdown("Visualisation des connexions entre films similaires (basé sur la similarité cosinus).")
        
        if 'svd' in models:
            svd_model = models['svd']
            item_features = svd_model['item_features'] # No transpose
            item_index = svd_model['item_index']
            
            # Select a movie
            movie_titles = [movies[movies['movie_id'] == mid]['title'].values[0] 
                           for mid in item_index if mid in movies['movie_id'].values][:100]
            
            selected_movie = st.selectbox("Choisir un film", movie_titles)
            
            if selected_movie:
                # Find movie index
                movie_id = movies[movies['title'] == selected_movie]['movie_id'].values[0]
                movie_idx = list(item_index).index(movie_id)
                
                # Calculate similarities
                from sklearn.metrics.pairwise import cosine_similarity
                similarities = cosine_similarity([item_features[movie_idx]], item_features)[0]
                top_similar_indices = similarities.argsort()[::-1][1:11]  # Top 10 similar
                
                # Create network graph
                edge_x = []
                edge_y = []
                node_x = []
                node_y = []
                node_text = []
                
                # Use 2D PCA for positioning
                pca = PCA(n_components=2)
                positions = pca.fit_transform(item_features[[movie_idx] + list(top_similar_indices)])
                
                # Center node
                node_x.append(positions[0, 0])
                node_y.append(positions[0, 1])
                node_text.append(selected_movie)
                
                # Similar nodes and edges
                for i, idx in enumerate(top_similar_indices, 1):
                    node_x.append(positions[i, 0])
                    node_y.append(positions[i, 1])
                    similar_movie_id = item_index[idx]
                    similar_title = movies[movies['movie_id'] == similar_movie_id]['title'].values[0]
                    node_text.append(f"{similar_title} ({similarities[idx]:.2f})")
                    
                    # Edge from center to this node
                    edge_x.extend([positions[0, 0], positions[i, 0], None])
                    edge_y.extend([positions[0, 1], positions[i, 1], None])
                
                # Create figure
                fig = go.Figure()
                
                # Add edges
                fig.add_trace(go.Scatter(
                    x=edge_x, y=edge_y,
                    mode='lines',
                    line=dict(width=1, color='#888'),
                    hoverinfo='none',
                    showlegend=False
                ))
                
                # Add nodes
                fig.add_trace(go.Scatter(
                    x=node_x, y=node_y,
                    mode='markers+text',
                    marker=dict(size=[20] + [12]*10, color=['#E50914'] + ['#666']*10),
                    text=node_text,
                    textposition="top center",
                    hoverinfo='text',
                    showlegend=False
                ))
                
                fig.update_layout(
                    title=f"Films similaires à '{selected_movie}'",
                    template="plotly_dark",
                    height=600,
                    xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                    yaxis=dict(showgrid=False, zeroline=False, showticklabels=False)
                )
                
                st.plotly_chart(fig, use_container_width=True)
        else:
            st.warning("Modèle SVD non trouvé.")

else:
    st.error("Données ou modèles manquants. Veuillez lancer l'entraînement.")
