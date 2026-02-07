import streamlit as st
from utils.recommender import get_similar_movies, load_data, get_posters_batch
from utils.ui import load_custom_css, render_header

st.set_page_config(page_title="Films Similaires", page_icon="🎬", layout="wide")

load_custom_css()
render_header()

st.markdown("""
Trouvez des films similaires à vos favoris en utilisant la **similarité cosinus** basée sur les facteurs latents.
""")

data = load_data()

if data:
    movies = data['movies']
    
    # Movie Selection
    movie_titles = movies['title'].tolist()
    
    with st.container():
        selected_title = st.selectbox("Sélectionnez un film que vous avez aimé :", movie_titles)
        
        if st.button("Trouver des films similaires", type="primary"):
            with st.spinner("🔍 Recherche de films similaires..."):
                # Convert title to movie_id
                movie_id = movies[movies['title'] == selected_title]['movie_id'].values[0]
                similar_movies = get_similar_movies(movie_id, data, n=10)
                
                if similar_movies is not None and len(similar_movies) > 0:
                    # OMDb API Key
                    omdb_api_key = "c9d2f347"
                    with st.sidebar:
                        with st.expander("🛠️ API Affiches (OMDb)"):
                            omdb_api_key = st.text_input("Clé API OMDb", value="c9d2f347", type="password")

                    st.success(f"Si vous avez aimé **{selected_title}**, vous aimerez peut-être :")
                    
                    # Prepare list for batch fetching (top 5 only)
                    sim_movies_list = []
                    for idx, row in similar_movies.head(5).iterrows():
                        sim_movies_list.append({'title': row['title'], 'movie_id': row['movie_id']})
                            
                    # Fetch posters in batch
                    poster_urls = get_posters_batch(sim_movies_list, api_key=omdb_api_key)
                    
                    # Display in grid
                    cols = st.columns(5)
                    for i, (idx, row) in enumerate(similar_movies.head(5).iterrows()):
                        with cols[i]:
                            poster_url = poster_urls[i]
                            st.image(poster_url, use_container_width=True)
                            st.markdown(f"**{row['title']}**")
                            # Hint if it's a real poster or fallback
                            is_real = "omdbapi" in poster_url or "media-amazon" in poster_url or "m.media-amazon" in poster_url or "assets" in poster_url
                            status = "🎬" if is_real else "🖼️"
                            st.caption(f"{status} Similarité: {row['score']:.3f}")
                    
                    # Full list
                    st.subheader("📋 Liste complète")
                    st.dataframe(similar_movies[['title', 'score']], use_container_width=True)
                else:
                    st.error("Aucun film similaire trouvé.")
else:
    st.error("Données non disponibles.")
