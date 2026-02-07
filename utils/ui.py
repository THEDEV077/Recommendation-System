import streamlit as st

def load_custom_css():
    """Injects premium custom CSS for a Netflix-like experience."""
    st.markdown("""
        <style>
        /* Import Google Fonts */
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;800&display=swap');

        /* General Body */
        .stApp {
            background-color: #141414; /* Netflix Dark Background */
            font-family: 'Inter', sans-serif;
            color: #ffffff;
        }

        /* Titles and Headers */
        h1, h2, h3 {
            color: #E50914 !important; /* Netflix Red */
            font-weight: 800 !important;
            text-shadow: 2px 2px 4px rgba(0,0,0,0.5);
        }
        
        h4, h5, h6 {
            color: #d1d5db !important;
            font-weight: 600 !important;
        }

        /* Movie Cards */
        div[data-testid="stImage"] {
            transition: transform 0.3s ease, box-shadow 0.3s ease;
            border-radius: 8px;
            overflow: hidden;
            cursor: pointer;
        }
        
        div[data-testid="stImage"]:hover {
            transform: scale(1.05);
            box-shadow: 0 10px 20px rgba(0,0,0,0.5);
            z-index: 10;
        }

        div[data-testid="stImage"] img {
            border-radius: 8px;
        }

        /* Sidebar */
        section[data-testid="stSidebar"] {
            background-color: #000000;
            border-right: 1px solid #333;
        }

        /* Buttons (Premium Red) */
        div.stButton > button {
            background-color: #E50914;
            color: white;
            border: none;
            border-radius: 4px;
            padding: 0.5rem 1rem;
            font-weight: bold;
            transition: background-color 0.2s;
        }

        div.stButton > button:hover {
            background-color: #f40612;
            color: white;
        }

        /* Metrics */
        div[data-testid="stMetricValue"] {
            color: #E50914 !important;
            font-size: 2rem !important;
        }
        div[data-testid="stMetricLabel"] {
            color: #9ca3af !important;
        }

        /* Expanders */
        .streamlit-expanderHeader {
            background-color: #262626;
            color: white;
            border-radius: 4px;
        }
        
        /* Links */
        a {
            color: #E50914 !important;
            text-decoration: none;
        }
        a:hover {
            text-decoration: underline;
        }
        </style>
    """, unsafe_allow_html=True)

def render_header():
    """Renders a consistent stylish header."""
    st.markdown("""
        <div style="display: flex; align-items: center; justify-content: space-between; padding: 1rem 0; margin-bottom: 2rem; border-bottom: 1px solid #333;">
            <div style="display: flex; align-items: center; gap: 10px;">
                <span style="font-size: 2.5rem;">🎬</span>
                <span style="font-size: 2rem; font-weight: 800; color: #E50914; letter-spacing: 1px;">MOVIELENS</span>
            </div>
            <div style="color: #666; font-size: 0.9rem;">
                Powered by AI & Streamlit
            </div>
        </div>
    """, unsafe_allow_html=True)
