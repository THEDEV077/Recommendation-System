import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from utils.recommender import load_data
from utils.ui import load_custom_css, render_header

st.set_page_config(page_title="Exploration des Données", page_icon="📊", layout="wide")

load_custom_css()
render_header()

st.markdown("""
Explorez les données MovieLens 100k avec des visualisations interactives pour mieux comprendre les tendances et les préférences des utilisateurs.
""")

data = load_data()

if data:
    movies = data['movies']
    ratings = data['ratings']
    
    # Overview Metrics
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("🎬 Films", len(movies))
    with col2:
        st.metric("👥 Utilisateurs", ratings['user_id'].nunique())
    with col3:
        st.metric("⭐ Notes", len(ratings))
    with col4:
        st.metric("📊 Note Moyenne", f"{ratings['rating'].mean():.2f}")
    
    st.markdown("---")
    
    tab1, tab2, tab3, tab4 = st.tabs(["🔥 Films Populaires", "📈 Distribution des Notes", "🎭 Analyse des Genres", "📅 Tendances Temporelles"])
    
    with tab1:
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Top 10 Films les plus notés")
            
            movie_stats = ratings.groupby('item_id').agg({'rating': ['count', 'mean']})
            movie_stats.columns = ['count', 'mean']
            
            # Merge with titles
            top_movies = movie_stats.sort_values(by='count', ascending=False).head(10)
            top_movies = top_movies.join(movies.set_index('movie_id'))
            
            # Plotly Bar Chart
            fig_pop = px.bar(
                top_movies, 
                x='count', 
                y='title', 
                orientation='h',
                title="Films avec le plus grand nombre d'avis",
                labels={'count': 'Nombre de votes', 'title': 'Film'},
                color='mean',
                color_continuous_scale='RdYlGn',
                hover_data=['mean']
            )
            fig_pop.update_layout(
                yaxis={'categoryorder':'total ascending'}, 
                template="plotly_dark",
                height=500
            )
            st.plotly_chart(fig_pop, use_container_width=True)
        
        with col2:
            # Best Rated (with minimum votes)
            st.subheader("🏆 Meilleurs Films (min. 50 votes)")
            qualified = movie_stats[movie_stats['count'] >= 50].sort_values('mean', ascending=False).head(10)
            qualified = qualified.join(movies.set_index('movie_id'))
            
            fig_best = px.bar(
                qualified,
                x='mean',
                y='title',
                orientation='h',
                title="Films les mieux notés",
                labels={'mean': 'Note moyenne', 'title': 'Film'},
                color='mean',
                color_continuous_scale='Greens'
            )
            fig_best.update_layout(
                yaxis={'categoryorder':'total ascending'},
                template="plotly_dark",
                height=500
            )
            st.plotly_chart(fig_best, use_container_width=True)
        
    with tab2:
        st.subheader("Distribution des Notes")
        
        col1, col2 = st.columns(2)
        
        with col1:
            rating_counts = ratings['rating'].value_counts().sort_index()
            fig_dist = px.bar(
                x=rating_counts.index,
                y=rating_counts.values,
                labels={'x': 'Note', 'y': 'Fréquence'},
                title="Histogramme des notes",
                color=rating_counts.values,
                color_continuous_scale='Reds'
            )
            fig_dist.update_layout(template="plotly_dark", showlegend=False)
            st.plotly_chart(fig_dist, use_container_width=True)
        
        with col2:
            fig_pie = px.pie(
                values=rating_counts.values, 
                names=rating_counts.index,
                title="Répartition des notes",
                color_discrete_sequence=px.colors.sequential.Reds_r
            )
            fig_pie.update_layout(template="plotly_dark")
            st.plotly_chart(fig_pie, use_container_width=True)
        
        # Box plot by user
        st.subheader("📦 Distribution des notes par utilisateur (échantillon)")
        sample_users = ratings['user_id'].unique()[:20]
        sample_data = ratings[ratings['user_id'].isin(sample_users)]
        
        fig_box = px.box(
            sample_data,
            x='user_id',
            y='rating',
            title="Variabilité des notes par utilisateur",
            labels={'user_id': 'ID Utilisateur', 'rating': 'Note'}
        )
        fig_box.update_layout(template="plotly_dark", showlegend=False)
        st.plotly_chart(fig_box, use_container_width=True)
        
    with tab3:
        st.subheader("Analyse des Genres")
        
        genre_cols = ['Action', 'Adventure', 'Animation', 'Childrens', 'Comedy', 'Crime', 'Documentary', 'Drama', 'Fantasy', 'Film-Noir', 'Horror', 'Musical', 'Mystery', 'Romance', 'Sci-Fi', 'Thriller', 'War', 'Western']
        genre_counts = movies[genre_cols].sum().sort_values(ascending=False)
        
        col1, col2 = st.columns(2)
        
        with col1:
            fig_genre = px.bar(
                x=genre_counts.index,
                y=genre_counts.values,
                labels={'x': 'Genre', 'y': 'Nombre de films'},
                title="Nombre de films par genre",
                color=genre_counts.values,
                color_continuous_scale='Reds'
            )
            fig_genre.update_layout(xaxis_tickangle=-45, template="plotly_dark", showlegend=False)
            st.plotly_chart(fig_genre, use_container_width=True)
        
        with col2:
            # Average rating by genre
            genre_ratings = {}
            for genre in genre_cols:
                genre_movies = movies[movies[genre] == 1]['movie_id']
                genre_ratings[genre] = ratings[ratings['item_id'].isin(genre_movies)]['rating'].mean()
            
            genre_ratings_df = pd.DataFrame(list(genre_ratings.items()), columns=['Genre', 'Rating'])
            genre_ratings_df = genre_ratings_df.sort_values('Rating', ascending=False)
            
            fig_rating = px.bar(
                genre_ratings_df,
                x='Genre',
                y='Rating',
                title="Note moyenne par genre",
                color='Rating',
                color_continuous_scale='RdYlGn'
            )
            fig_rating.update_layout(xaxis_tickangle=-45, template="plotly_dark", showlegend=False)
            st.plotly_chart(fig_rating, use_container_width=True)
    
    with tab4:
        st.subheader("📅 Évolution des notes dans le temps")
        
        # Convert timestamp to datetime
        ratings_time = ratings.copy()
        ratings_time['date'] = pd.to_datetime(ratings_time['timestamp'], unit='s')
        ratings_time['year_month'] = ratings_time['date'].dt.to_period('M').astype(str)
        
        # Ratings per month
        monthly_counts = ratings_time.groupby('year_month').size().reset_index(name='count')
        
        fig_time = px.line(
            monthly_counts,
            x='year_month',
            y='count',
            title="Nombre de notes par mois",
            labels={'year_month': 'Mois', 'count': 'Nombre de notes'},
            markers=True
        )
        fig_time.update_layout(template="plotly_dark", xaxis_tickangle=-45)
        st.plotly_chart(fig_time, use_container_width=True)
        
        # Average rating over time
        monthly_avg = ratings_time.groupby('year_month')['rating'].mean().reset_index()
        
        fig_avg_time = px.line(
            monthly_avg,
            x='year_month',
            y='rating',
            title="Note moyenne par mois",
            labels={'year_month': 'Mois', 'rating': 'Note moyenne'},
            markers=True
        )
        fig_avg_time.update_layout(template="plotly_dark", xaxis_tickangle=-45)
        st.plotly_chart(fig_avg_time, use_container_width=True)

else:
    st.error("Données non disponibles.")
