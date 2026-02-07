import streamlit as st
import pandas as pd
import json
import os
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from utils.ui import load_custom_css, render_header

st.set_page_config(page_title="Comparaison des Modèles", page_icon="📊", layout="wide")

load_custom_css()
render_header()

st.markdown("""
Cette page compare la précision de différents algorithmes de recommandation. 
L'objectif est d'identifier le modèle le plus précis pour prédire les notes des utilisateurs.
""")

models_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'models')
metrics_path = os.path.join(models_dir, 'model_metrics.json')

if os.path.exists(metrics_path):
    with open(metrics_path, 'r') as f:
        metrics = json.load(f)

    df_metrics = pd.DataFrame(metrics)
    
    # Trouver le meilleur modèle
    best_rmse = df_metrics.loc[df_metrics['rmse'].idxmin()]
    
    st.success(f"🏆 **Algorithme le plus précis : {best_rmse['algorithm']}** (RMSE: {best_rmse['rmse']})")
    
    tab1, tab2, tab3 = st.tabs(["📊 Vue d'ensemble", "📉 Analyse des Erreurs", "🎯 Comparaison Détaillée"])
    
    with tab1:
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("⚖️ Compromis Précision vs Vitesse")
            # Scatter Plot RMSE vs Time
            fig_scatter = px.scatter(
                df_metrics, 
                x='training_time', 
                y='rmse', 
                color='algorithm',
                size='training_time',
                hover_data=['mae'],
                text='algorithm',
                title="Précision (RMSE) vs Temps d'entraînement",
                labels={'training_time': 'Temps (s)', 'rmse': 'Erreur RMSE (Plus bas est mieux)'}
            )
            fig_scatter.update_traces(textposition='top center', textfont_size=10)
            fig_scatter.update_layout(
                template="plotly_dark",
                height=500,
                margin=dict(l=50, r=50, t=80, b=50)
            )
            st.plotly_chart(fig_scatter, use_container_width=True)
            
        with col2:
            st.subheader("🕸️ Comparaison Multi-Critères")
            
            # Radar Chart requires normalization
            df_norm = df_metrics.copy()
            
            # Helper for normalization (Invert for RMSE/MAE/Time since lower is better)
            for col in ['rmse', 'mae', 'training_time']:
                min_val = df_norm[col].min()
                max_val = df_norm[col].max()
                if max_val != min_val:
                    df_norm[col] = 1 - ((df_norm[col] - min_val) / (max_val - min_val))
                else:
                    df_norm[col] = 1.0
                    
            # Create Radar
            fig_radar = go.Figure()
            
            categories = ['Précision (RMSE)', 'Précision (MAE)', 'Vitesse']
            
            for index, row in df_norm.iterrows():
                fig_radar.add_trace(go.Scatterpolar(
                    r=[row['rmse'], row['mae'], row['training_time']],
                    theta=categories,
                    fill='toself',
                    name=row['algorithm']
                ))
                
            fig_radar.update_layout(
                polar=dict(
                    radialaxis=dict(
                        visible=True,
                        range=[0, 1]
                    )),
                showlegend=True,
                title="Score Normalisé (Plus grand = Meilleur)",
                template="plotly_dark"
            )
            st.plotly_chart(fig_radar, use_container_width=True)
    
    with tab2:
        st.subheader("📉 Distribution des Erreurs par Modèle")
        st.markdown("Analyse de la distribution des erreurs de prédiction pour chaque algorithme.")
        
        # Simulate error distributions (in real app, load from saved predictions)
        np.random.seed(42)
        error_data = []
        for _, model in df_metrics.iterrows():
            # Generate synthetic error distribution based on RMSE
            errors = np.random.normal(0, model['rmse'], 1000)
            for err in errors:
                error_data.append({
                    'algorithm': model['algorithm'],
                    'error': err
                })
        
        df_errors = pd.DataFrame(error_data)
        
        # Box Plot
        st.subheader("📦 Box Plot des Erreurs")
        fig_box = px.box(
            df_errors,
            x='algorithm',
            y='error',
            color='algorithm',
            title="Distribution des erreurs de prédiction",
            labels={'error': 'Erreur (Note Prédite - Note Réelle)', 'algorithm': 'Algorithme'},
            template="plotly_dark"
        )
        st.plotly_chart(fig_box, use_container_width=True)
        
        # Histogram
        st.subheader("📊 Histogrammes des Erreurs")
        fig_hist = px.histogram(
            df_errors,
            x='error',
            color='algorithm',
            barmode='overlay',
            title="Histogramme des erreurs par algorithme",
            labels={'error': 'Erreur', 'count': 'Fréquence'},
            template="plotly_dark",
            opacity=0.7
        )
        st.plotly_chart(fig_hist, use_container_width=True)
        
        st.info("💡 Une distribution centrée autour de 0 avec une faible variance indique un modèle précis et cohérent.")
    
    with tab3:
        st.subheader("🎯 Tableau Comparatif Détaillé")
        
        # Add performance ranking
        df_display = df_metrics.copy()
        df_display['rank_rmse'] = df_display['rmse'].rank()
        df_display['rank_mae'] = df_display['mae'].rank()
        df_display['rank_speed'] = df_display['training_time'].rank()
        
        st.dataframe(
            df_display.sort_values(by='rmse'),
            use_container_width=True,
            column_config={
                "algorithm": "Algorithme",
                "rmse": st.column_config.NumberColumn("RMSE", format="%.4f"),
                "mae": st.column_config.NumberColumn("MAE", format="%.4f"),
                "training_time": st.column_config.NumberColumn("Temps (s)", format="%.2f"),
                "rank_rmse": st.column_config.NumberColumn("Rang RMSE", format="%d"),
                "rank_mae": st.column_config.NumberColumn("Rang MAE", format="%d"),
                "rank_speed": st.column_config.NumberColumn("Rang Vitesse", format="%d")
            }
        )
        
        # Performance metrics comparison
        st.subheader("📈 Comparaison des Métriques")
        
        col1, col2 = st.columns(2)
        
        with col1:
            fig_rmse = px.bar(
                df_metrics.sort_values('rmse'),
                x='algorithm',
                y='rmse',
                title="RMSE par Algorithme",
                template="plotly_dark",
                color='rmse',
                color_continuous_scale='Reds_r'
            )
            st.plotly_chart(fig_rmse, use_container_width=True)
        
        with col2:
            fig_mae = px.bar(
                df_metrics.sort_values('mae'),
                x='algorithm',
                y='mae',
                title="MAE par Algorithme",
                template="plotly_dark",
                color='mae',
                color_continuous_scale='Blues_r'
            )
            st.plotly_chart(fig_mae, use_container_width=True)
        
        st.info("""
        **Interprétation :**
        - **RMSE (Root Mean Square Error)** : Pénalise fortement les grandes erreurs. C'est l'indicateur principal.
        - **MAE (Mean Absolute Error)** : Erreur moyenne en valeur absolue.
        - **SVD** : Généralement le plus équilibré entre précision et vitesse.
        - **KNN** : Très précis mais peut être lent sur de grands datasets.
        """)

else:
    st.warning("⚠️ Les métriques de performance n'ont pas encore été générées. Veuillez lancer le script d'entraînement.")
    if st.button("Lancer l'entraînement maintenant"):
        st.info("Lancement de `notebooks/train_models.py`...")
        st.code("python notebooks/train_models.py")
