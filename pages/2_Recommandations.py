import streamlit as st
from utils.recommender import load_data, get_recommendations_user, get_movie_title, get_posters_batch
from utils.ui import load_custom_css, render_header

st.set_page_config(page_title="Recommandations", page_icon="🎯")

load_custom_css()
render_header()
# st.title("🎯 Vos Recommandations")

data = load_data()

if data:
    # Sidebar Configuration
    with st.sidebar:
        st.header("⚙️ Configuration")
        with st.expander("🛠️ API Affiches (OMDb)"):
            omdb_api_key = st.text_input("Clé API OMDb", value="c9d2f347", type="password")
            st.caption("Obtenez une clé gratuite sur [omdbapi.com](https://www.omdbapi.com/apikey.aspx) si les images ne chargent plus.")

    # Model Selection
    model_options = ['Automatique (Meilleur Performance)', 'NCF (Neural Collaborative Filtering)', 'AutoRec (Autoencoder)', 'KNN (Item-Based)', 'SVD (Matrix Factorization)', 'NMF (Non-Negative Matrix Factorization)', 'Baseline (Moyenne Globale)']
    selected_model_label = st.selectbox("🤖 Choisir l'algorithme de recommandation", model_options)
    
    
    force_model = None
    if "NCF" in selected_model_label:
        force_model = 'ncf'
    elif "AutoRec" in selected_model_label:
        force_model = 'autorec'
    elif "KNN" in selected_model_label:
        force_model = 'knn'
    elif "SVD" in selected_model_label:
        force_model = 'svd'
    elif "NMF" in selected_model_label:
        force_model = 'nmf'
    elif "Baseline" in selected_model_label:
        force_model = 'baseline'
        
    best_name = data.get('best_model_name', data.get('model_type', 'N/A').upper())
    
    if force_model:
        st.info(f"⚙️ Mode Manuel : **{selected_model_label}** activé.")
    else:
        st.info(f"🚀 Mode Automatique : **{best_name}** est le plus performant.")
    
    # Input User ID with a nice slider or number input
    col_input, col_empty = st.columns([1, 2])
    with col_input:
        user_id = st.number_input("👤 Votre User ID", min_value=1, max_value=943, value=196)
    
    if st.button("✨ Générer mon catalogue personnalisé"):
        with st.spinner("Recherche des meilleures pépites pour vous..."):
            recs = get_recommendations_user(user_id, data, force_model=force_model)
            
            if not recs.empty:
                st.subheader(f"🍿 Recommandé pour vous (User {user_id})")
                
                # Netflix Grid Style (2 rows of 5 movies)
                # Prepare data for batch fetching
                top_recs_list = []
                for i in range(min(10, len(recs))):
                     row = recs.iloc[i]
                     top_recs_list.append({'title': row['title'], 'movie_id': row['movie_id']})
                
                # Batch fetch posters
                poster_urls = get_posters_batch(top_recs_list, api_key=omdb_api_key)
                
                # First Row (0-4)
                cols1 = st.columns(5)
                for i in range(min(5, len(top_recs_list))):
                    row = recs.iloc[i]
                    with cols1[i]:
                        poster_url = poster_urls[i]
                        st.image(poster_url, use_container_width=True)
                        st.markdown(f"**{row['title']}**")
                        # Hint if it's a real poster or fallback
                        is_real = "omdbapi" in poster_url or "media-amazon" in poster_url or "m.media-amazon" in poster_url or "assets" in poster_url
                        status = "🎬" if is_real else "🖼️"
                        st.caption(f"{status} Score: {row['score']}/5")
                
                # Second Row (5-9)
                if len(top_recs_list) > 5:
                    cols2 = st.columns(5)
                    for i in range(5, min(10, len(top_recs_list))):
                        row = recs.iloc[i]
                        with cols2[i-5]:
                            poster_url = poster_urls[i]
                            st.image(poster_url, use_container_width=True)
                            st.markdown(f"**{row['title']}**")
                            is_real = "omdbapi" in poster_url or "media-amazon" in poster_url or "m.media-amazon" in poster_url or "assets" in poster_url
                            status = "🎬" if is_real else "🖼️"
                            st.caption(f"{status} Score: {row['score']}/5")
            else:
                st.warning("Aucune recommandation trouvée.")
                
    st.markdown("---")
    st.subheader("🛠️ Détails Techniques")
    if 'AutoRec' in best_name or (force_model == 'autorec'):
        st.write("Ce système utilise **AutoRec**. C'est un autoencodeur profond qui apprend à reconstruire les préférences des utilisateurs en compressant puis décompressant les données de notation.")
    elif 'NCF' in best_name or (force_model == 'ncf'):
        st.write("Ce système utilise **NCF (Neural Collaborative Filtering)**. Il combine la factorisation matricielle généralisée (GMF) et les réseaux de neurones profonds (MLP) pour capturer les interactions complexes entre utilisateurs et items.")
    elif 'KNN' in best_name:
        st.write("Ce système utilise l'algorithme **KNN (K-Nearest Neighbors)**. Il identifie les films similaires à ceux que vous avez aimés en analysant les habitudes de millions d'autres spectateurs.")
    elif 'SVD' in best_name:
        st.write("Ce système utilise la **SVD (Singular Value Decomposition)**. C'est une technique de factorisation matricielle ultra-puissante qui prédit vos goûts futurs à partir de vos notations passées.")
    elif 'NMF' in best_name:
        st.write("Ce système utilise la **NMF (Non-negative Matrix Factorization)**. C'est similaire à SVD, mais avec des contraintes de non-négativité, ce qui la rend souvent plus interprétable pour les données de systèmes de recommandation.")
    elif 'LightFM' in best_name:
        st.write("Ce système utilise **LightFM**, un modèle hybride puissant combinant le filtrage collaboratif et l'analyse de contenu.")
    else:
        st.write("Calcul basé sur la similarité cosinus et les tendances actuelles.")
