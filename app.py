import streamlit as st
import pandas as pd
import os
from utils.ui import load_custom_css, render_header

st.set_page_config(
    page_title="MovieLens Recommender",
    page_icon="🎬",
    layout="wide"
)

# Inject Premium UI
load_custom_css()
render_header()

# st.title("🎬 Système de Recommandation de Films") # Replaced by render_header
st.markdown("### Bienvenue sur votre moteur de recommandation personnel")

st.markdown("""
Cette application utilise le dataset **MovieLens 100k** pour vous suggérer des films.

#### Fonctionnalités :
- **📊 Exploration** : Découvrez les données, les films les plus populaires et la distribution des notes.
- **🎯 Recommandations** : Obtenez des suggestions personnalisées pour un utilisateur donné ou basées sur vos goûts.
- **🔗 Films Similaires** : Trouvez des films proches de vos favoris grâce à l'analyse de similarité (Cosine Similarity).

#### Technologies utilisées :
- **Python** (Pandas, Numpy, Scikit-learn)
- **Streamlit** pour l'interface utilisateur
- **Filtrage Collaboratif** (Item-Based)

👈 **Utilisez le menu à gauche pour naviguer !**
""")

# Afficher quelques stats rapides si possible
base_dir = os.path.dirname(os.path.abspath(__file__))
try:
    from utils.recommender import load_data
    data = load_data()
    if data:
        col1, col2, col3 = st.columns(3)
        col1.metric("Films", len(data['movies']))
        col2.metric("Notes", len(data['ratings']))
        col3.metric("Utilisateurs", data['ratings']['user_id'].nunique())
except Exception as e:
    st.warning("Impossible de charger les statistiques rapides.")

st.image("https://images.unsplash.com/photo-1536440136628-849c177e76a1?ixlib=rb-1.2.1&auto=format&fit=crop&w=1350&q=80", caption="Movie Night")
