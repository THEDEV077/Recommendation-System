import pickle
import pandas as pd
import os

try:
    with open('models/data_mappings.pkl', 'rb') as f:
        print("Loading models/data_mappings.pkl...")
        mappings = pickle.load(f)
        movies = mappings['movies']
        print("\n--- Movies Columns in Pickle ---")
        print(movies.columns.tolist())
        print(f"Shape: {movies.shape}")
        
    print("\n--- Reading u.item directly ---")
    headers = [
        'movie_id', 'title', 'release_date', 'video_release_date', 'imdb_url',
        'unknown', 'Action', 'Adventure', 'Animation', 'Childrens', 'Comedy', 
        'Crime', 'Documentary', 'Drama', 'Fantasy', 'Film-Noir', 'Horror', 
        'Musical', 'Mystery', 'Romance', 'Sci-Fi', 'Thriller', 'War', 'Western'
    ]
    df = pd.read_csv('ml-100k/u.item', sep='|', names=headers, encoding='latin-1', header=None)
    print("\n--- Movies Columns form Direct Read ---")
    print(df.columns.tolist())
    print(f"Shape: {df.shape}")
    
except Exception as e:
    print(f"Error: {e}")
