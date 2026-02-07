import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from utils.recommender import load_data, get_posters_batch
from utils.ui import load_custom_css, render_header

st.set_page_config(page_title="Profil Utilisateur", page_icon="👤", layout="wide")

load_custom_css()
render_header()

st.markdown("""
Analysez les préférences et l'historique de visionnage d'un utilisateur spécifique.
""")

data = load_data()

if data:
    movies = data['movies']
    ratings = data['ratings']
    
    user_id = st.sidebar.number_input("Sélectionner un utilisateur", min_value=1, max_value=943, value=196)
    
    # Filter user ratings
    user_ratings = ratings[ratings['user_id'] == user_id]
    user_movies = user_ratings.merge(movies, left_on='item_id', right_on='movie_id')
    
    if not user_movies.empty:
        # Metrics
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("🎬 Films vus", len(user_movies))
        with col2:
            st.metric("⭐ Note moyenne donnée", f"{user_movies['rating'].mean():.2f}")
        with col3:
            # Top Genre
            genre_cols = ['Action', 'Adventure', 'Animation', 'Childrens', 'Comedy', 'Crime', 'Documentary', 'Drama', 'Fantasy', 'Film-Noir', 'Horror', 'Musical', 'Mystery', 'Romance', 'Sci-Fi', 'Thriller', 'War', 'Western']
            # Ensure we only use columns that exist
            valid_genre_cols = [c for c in genre_cols if c in user_movies.columns]
            
            if valid_genre_cols:
                user_genres = user_movies[valid_genre_cols].sum().sort_values(ascending=False)
                if not user_genres.empty:
                    st.metric("🎭 Genre favori", user_genres.index[0])
                else:
                    st.metric("🎭 Genre favori", "N/A")
            else:
                st.metric("🎭 Genre favori", "N/A")
                user_genres = pd.Series()
        with col4:
            # Favorite rating
            most_common_rating = user_movies['rating'].mode()[0]
            st.metric("💯 Note préférée", int(most_common_rating))
            
        st.markdown("---")
        
        tab1, tab2, tab3 = st.tabs(["📊 Analyse des Goûts", "🏆 Films Préférés", "📜 Historique Complet"])
        
        with tab1:
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("Distribution des genres appréciés")
                fig_genre = px.bar(
                    x=user_genres.index, 
                    y=user_genres.values,
                    labels={'x': 'Genre', 'y': 'Nombre de films vus'},
                    color=user_genres.values,
                    color_continuous_scale='Reds'
                )
                fig_genre.update_layout(xaxis_tickangle=-45, template="plotly_dark", showlegend=False)
                st.plotly_chart(fig_genre, use_container_width=True)
            
            with col2:
                st.subheader("Distribution des notes données")
                rating_dist = user_movies['rating'].value_counts().sort_index()
                fig_ratings = px.bar(
                    x=rating_dist.index,
                    y=rating_dist.values,
                    labels={'x': 'Note', 'y': 'Fréquence'},
                    color=rating_dist.values,
                    color_continuous_scale='Greens'
                )
                fig_ratings.update_layout(template="plotly_dark", showlegend=False)
                st.plotly_chart(fig_ratings, use_container_width=True)
            
            # Radar chart for genre preferences
            st.subheader("🕸️ Profil de goûts (Radar)")
            top_genres = user_genres.head(8)
            
            fig_radar = go.Figure()
            fig_radar.add_trace(go.Scatterpolar(
                r=top_genres.values,
                theta=top_genres.index,
                fill='toself',
                name='Utilisateur',
                line_color='#E50914'
            ))
            
            fig_radar.update_layout(
                polar=dict(
                    radialaxis=dict(visible=True, range=[0, top_genres.max()])
                ),
                template="plotly_dark",
                showlegend=False
            )
            st.plotly_chart(fig_radar, use_container_width=True)
        
        with tab2:
            st.subheader("🏆 Top 10 Films Préférés")
            
            top_rated = user_movies.sort_values('rating', ascending=False).head(10)
            
            # OMDb API Key
            omdb_api_key = "c9d2f347"
            with st.sidebar:
                with st.expander("🛠️ API Affiches (OMDb)"):
                    omdb_api_key = st.text_input("Clé API OMDb", value="c9d2f347", type="password", key="user_profile_omdb")
            
            # Fetch posters for top 10
            poster_list = [{'title': row['title'], 'movie_id': row['movie_id']} for _, row in top_rated.head(10).iterrows()]
            poster_urls = get_posters_batch(poster_list, api_key=omdb_api_key)
            
            # Display grid - First row (5 movies)
            st.markdown("**Top 5**")
            cols = st.columns(5)
            for i, (idx, row) in enumerate(top_rated.head(5).iterrows()):
                with cols[i]:
                    st.image(poster_urls[i], use_container_width=True)
                    st.markdown(f"**{row['title']}**")
                    st.caption(f"⭐ Note: {int(row['rating'])}/5")
            
            # Second row (next 5 movies)
            st.markdown("**Top 6-10**")
            cols2 = st.columns(5)
            for i, (idx, row) in enumerate(top_rated.iloc[5:10].iterrows()):
                with cols2[i]:
                    st.image(poster_urls[i+5], use_container_width=True)
                    st.markdown(f"**{row['title']}**")
                    st.caption(f"⭐ Note: {int(row['rating'])}/5")
            
            # Full list
            st.dataframe(
                top_rated[['title', 'rating']].reset_index(drop=True),
                use_container_width=True,
                column_config={
                    "title": "Film",
                    "rating": st.column_config.NumberColumn("Note", format="⭐ %d")
                }
            )
        
        with tab3:
            st.subheader("📜 Historique de Visionnage Complet")
            
            # Add timestamp info
            user_movies_sorted = user_movies.sort_values('timestamp', ascending=False)
            user_movies_sorted['date'] = pd.to_datetime(user_movies_sorted['timestamp'], unit='s').dt.strftime('%Y-%m-%d')
            
            st.dataframe(
                user_movies_sorted[['title', 'rating', 'date']].reset_index(drop=True),
                use_container_width=True,
                column_config={
                    "title": "Film",
                    "rating": st.column_config.NumberColumn("Note", format="⭐ %d"),
                    "date": "Date"
                },
                height=600
            )
            
            # Timeline visualization
            st.subheader("📅 Activité dans le temps")
            user_movies_sorted['year_month'] = pd.to_datetime(user_movies_sorted['timestamp'], unit='s').dt.to_period('M').astype(str)
            monthly_activity = user_movies_sorted.groupby('year_month').size().reset_index(name='count')
            
            fig_timeline = px.bar(
                monthly_activity,
                x='year_month',
                y='count',
                title="Nombre de films vus par mois",
                labels={'year_month': 'Mois', 'count': 'Films vus'},
                color='count',
                color_continuous_scale='Reds'
            )
            fig_timeline.update_layout(template="plotly_dark", xaxis_tickangle=-45, showlegend=False)
            st.plotly_chart(fig_timeline, use_container_width=True)
    else:
        st.warning(f"Aucune donnée disponible pour l'utilisateur {user_id}.")
else:
    st.error("Données non disponibles.")
