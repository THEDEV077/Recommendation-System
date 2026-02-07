import pandas as pd
import numpy as np
import time
import sys
import os
import pickle
import json
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.decomposition import TruncatedSVD, NMF
from sklearn.metrics.pairwise import cosine_similarity
from scipy.sparse import csr_matrix
import warnings
import traceback

# Suppress warnings
warnings.filterwarnings('ignore')

# --- Configuration ---
DATA_PATH = 'ml-100k/u.data'
ITEM_PATH = 'ml-100k/u.item'
SEPARATOR = '\t'
COLUMNS = ['user_id', 'item_id', 'rating', 'timestamp']
TEST_SIZE = 0.2
RANDOM_STATE = 42
TOP_K = 10
RELEVANCE_THRESHOLD = 4.0
MODELS_DIR = 'models'

# Movie Columns for correct mapping
MOVIE_COLS = [
    'movie_id', 'title', 'release_date', 'video_release_date', 'imdb_url',
    'unknown', 'Action', 'Adventure', 'Animation', 'Childrens', 'Comedy', 
    'Crime', 'Documentary', 'Drama', 'Fantasy', 'Film-Noir', 'Horror', 
    'Musical', 'Mystery', 'Romance', 'Sci-Fi', 'Thriller', 'War', 'Western'
]

# --- 1. Data Loading & Preprocessing ---

def load_data():
    print("Loading data...")
    try:
        ratings = pd.read_csv(DATA_PATH, sep=SEPARATOR, names=COLUMNS)
        
        # FIX: Load ALL columns for movies to support App Exploration features (Genres)
        movies = pd.read_csv(ITEM_PATH, sep='|', encoding='latin-1', header=None, names=MOVIE_COLS)
        
        print(f"Loaded {len(ratings)} ratings and {len(movies)} movies.")
        return ratings, movies
    except Exception as e:
        print(f"Error loading data: {e}")
        traceback.print_exc()
        return None, None

def get_train_test_split(ratings):
    train_data, test_data = train_test_split(ratings, test_size=TEST_SIZE, random_state=RANDOM_STATE)
    return train_data, test_data

def create_csr_matrix(df, n_users, n_items):
    row = df['user_id'] - 1
    col = df['item_id'] - 1
    data = df['rating']
    return csr_matrix((data, (row, col)), shape=(n_users, n_items))

# --- 2. Evaluation Metrics ---

def calculate_metrics(y_true, y_pred):
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    return rmse, mae

def calculate_ranking_metrics(model_func, test_data, train_data, n_users, n_items, k=TOP_K, threshold=RELEVANCE_THRESHOLD):
    # Simplified ranking evaluation
    # Returns P@K, R@K
    print(f"Calculating Ranking Metrics (@{k})...")
    test_users = test_data['user_id'].unique()
    precisions = []
    recalls = []
    
    ground_truth = {}
    test_data_relevant = test_data[test_data['rating'] >= threshold]
    for uid, group in test_data_relevant.groupby('user_id'):
        ground_truth[uid] = set(group['item_id'].values)

    train_seen = {}
    for uid, group in train_data.groupby('user_id'):
        train_seen[uid] = set(group['item_id'].values)

    for uid in test_users:
        if uid not in ground_truth:
            continue
            
        relevant_items = ground_truth[uid]
        seen_items = train_seen.get(uid, set())
        
        try:
            scores = model_func(uid)
            scores = np.asarray(scores).flatten()
            
            if scores.size == 0: continue

            for item_id in seen_items:
                if item_id <= len(scores):
                    scores[item_id - 1] = -np.inf
            
            current_k = min(k, len(scores))
            if current_k == 0: continue
                
            top_k_indices = np.argpartition(scores, -current_k)[-current_k:]
            top_k_indices = top_k_indices[np.argsort(scores[top_k_indices])[::-1]]
            top_k_items = [idx + 1 for idx in top_k_indices]
            
            n_rel = len(relevant_items)
            n_rel_and_rec = len(set(top_k_items) & relevant_items)
            
            precisions.append(n_rel_and_rec / k)
            recalls.append(n_rel_and_rec / n_rel)
            
        except Exception:
            continue

    return np.mean(precisions) if precisions else 0.0, np.mean(recalls) if recalls else 0.0

# --- 3. Models ---

# Pytorch NCF & AutoRec
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from models.ncf import NCF
from models.autorec import AutoRec

# ... (MovieLensDataset remains same) ...

# --- AutoRec Functions ---
def train_autorec_model(train_df, n_users, n_items, epochs=50, batch_size=128, lr=0.001):
    print(f"Training AutoRec (Epochs={epochs}, LR={lr}, BS={batch_size})...")
    
    # AutoRec (Item-Based) takes input as user vectors for each item
    # Input dim = n_users
    # Output dim = n_users
    # Rows = Items
    
    # Create Item-User Matrix
    # We need a matrix R where R[i, u] is rating of user u for item i
    # Fill missing with 0
    R = np.zeros((n_items, n_users))
    
    # Adjust to 0-based
    users = train_df['user_id'].values - 1
    items = train_df['item_id'].values - 1
    ratings = train_df['rating'].values
    
    R[items, users] = ratings
    
    # Convert to Tensor
    R_tensor = torch.FloatTensor(R)
    
    dataset = torch.utils.data.TensorDataset(R_tensor)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    model = AutoRec(num_inputs=n_users, hidden_dim=500)
    criterion = nn.MSELoss() # We will mask zeros manually in loop or use a custom loss
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    model.train()
    for epoch in range(epochs):
        total_loss = 0
        for batch in dataloader:
            inputs = batch[0] # (batch_size, n_users)
            optimizer.zero_grad()
            outputs = model(inputs)
            
            # Masking: Only calculate loss for observed ratings
            mask = inputs > 0
            loss = criterion(outputs[mask], inputs[mask])
            
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            
        if (epoch+1) % 10 == 0:
            print(f"Epoch {epoch+1}/{epochs}, Loss: {total_loss/len(dataloader):.4f}")
            
    return model, R

def predict_autorec_user_scores(model, R, user_id, n_items):
    # AutoRec predicts R' (completed matrix)
    # We want predictions for a specific user across all items
    # R shape: (n_items, n_users)
    # Model forward pass on all items gives R_pred: (n_items, n_users)
    # But that's heavy.
    # Actually, model takes Item Vector (ratings from all users for that item) 
    # and outputs reconstructed Item Vector.
    # So for user u, we look at column u of the output.
    
    model.eval()
    u_idx = user_id - 1
    
    if u_idx >= R.shape[1]: return np.zeros(n_items)
    
    # We need to run forward pass on ALL items to get the column for user u
    # Input: R (all items)
    R_tensor = torch.FloatTensor(R) 
    # Process in batches if needed, but for 1682 items it fits in memory
    
    with torch.no_grad():
        reconstructed = model(R_tensor)
        
    # Get column for user
    user_scores = reconstructed[:, u_idx].numpy()
    return user_scores

# ... (train_ncf_model remains same) ...

class MovieLensDataset(Dataset):
    def __init__(self, user_ids, item_ids, ratings):
        self.users = torch.tensor(user_ids, dtype=torch.long)
        self.items = torch.tensor(item_ids, dtype=torch.long)
        self.ratings = torch.tensor(ratings, dtype=torch.float32)

    def __len__(self):
        return len(self.ratings)

    def __getitem__(self, idx):
        return self.users[idx], self.items[idx], self.ratings[idx]

def train_ncf_model(train_df, n_users, n_items, epochs=10, batch_size=256, lr=0.001):
    print(f"Training NCF (Epochs={epochs}, LR={lr}, BS={batch_size})...")
    
    # Adjust for 0-based indexing for embedding layers
    # user_id 1..943 -> 0..942
    train_users = train_df['user_id'].values - 1
    train_items = train_df['item_id'].values - 1
    train_ratings = train_df['rating'].values
    
    dataset = MovieLensDataset(train_users, train_items, train_ratings)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    model = NCF(n_users, n_items)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    model.train()
    for epoch in range(epochs):
        total_loss = 0
        for u, i, r in dataloader:
            optimizer.zero_grad()
            outputs = model(u, i)
            loss = criterion(outputs, r)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        
        if (epoch+1) % 2 == 0:
            print(f"Epoch {epoch+1}/{epochs}, Loss: {total_loss/len(dataloader):.4f}")
            
    return model

def evaluate_ncf(model, test_df):
    model.eval()
    test_users = torch.tensor(test_df['user_id'].values - 1, dtype=torch.long)
    test_items = torch.tensor(test_df['item_id'].values - 1, dtype=torch.long)
    
    with torch.no_grad():
        predictions = model(test_users, test_items)
        
    return predictions.numpy()

def predict_ncf_user_scores(model, user_id, n_items):
    model.eval()
    if user_id > 943: return np.zeros(n_items)
    
    u_idx = user_id - 1
    # Create all item indices
    item_indices = torch.arange(n_items, dtype=torch.long)
    user_indices = torch.full((n_items,), u_idx, dtype=torch.long)
    
    with torch.no_grad():
        scores = model(user_indices, item_indices)
        
    return scores.numpy()

class BaselineModel:
    def __init__(self, train_data):
        self.global_mean = train_data['rating'].mean()
        self.n_items = 1682
        
    def predict(self, user_ids, item_ids):
        return np.full(len(user_ids), self.global_mean)
        
    def predict_user_scores(self, user_id):
        return np.random.rand(self.n_items) 
        
class SVDModel:
    def __init__(self, n_components=20):
        self.model = TruncatedSVD(n_components=n_components, random_state=42)
        self.user_factors = None
        self.item_factors = None
        
    def fit(self, train_matrix):
        self.user_factors = self.model.fit_transform(train_matrix)
        self.item_factors = self.model.components_.T 
        
    def predict(self, user_ids, item_ids):
        preds = []
        for u, i in zip(user_ids, item_ids):
            u_idx = u - 1
            i_idx = i - 1
            if 0 <= u_idx < self.user_factors.shape[0] and 0 <= i_idx < self.item_factors.shape[0]:
                pred = np.dot(self.user_factors[u_idx], self.item_factors[i_idx])
            else:
                pred = 3.53
            preds.append(pred)
        return np.array(preds)

    def predict_user_scores(self, user_id):
        u_idx = user_id - 1
        if 0 <= u_idx < self.user_factors.shape[0]:
            return np.dot(self.user_factors[u_idx], self.item_factors.T)
        else:
            return np.zeros(self.item_factors.shape[0])

class NMFModel:
    def __init__(self, n_components=20):
        self.model = NMF(n_components=n_components, init='random', random_state=42)
        self.user_factors = None
        self.item_factors = None
        
    def fit(self, train_matrix):
        self.user_factors = self.model.fit_transform(train_matrix)
        self.item_factors = self.model.components_.T
        
    def predict(self, user_ids, item_ids):
        preds = []
        for u, i in zip(user_ids, item_ids):
            u_idx = u - 1
            i_idx = i - 1
            if 0 <= u_idx < self.user_factors.shape[0] and 0 <= i_idx < self.item_factors.shape[0]:
                pred = np.dot(self.user_factors[u_idx], self.item_factors[i_idx])
            else:
                pred = 3.53
            preds.append(pred)
        return np.array(preds)

    def predict_user_scores(self, user_id):
        u_idx = user_id - 1
        if 0 <= u_idx < self.user_factors.shape[0]:
            return np.dot(self.user_factors[u_idx], self.item_factors.T)
        else:
            return np.zeros(self.item_factors.shape[0])

class KNNModel:
    def __init__(self, k=20):
        self.k = k
        self.sim_matrix = None
        self.train_matrix = None
        
    def fit(self, train_matrix):
        self.train_matrix = train_matrix
        self.sim_matrix = cosine_similarity(train_matrix.T, dense_output=True)
        np.fill_diagonal(self.sim_matrix, 0)

    def predict(self, user_ids, item_ids):
        preds = []
        for u, i in zip(user_ids, item_ids):
            u_idx = u - 1
            i_idx = i - 1
            if 0 <= u_idx < self.train_matrix.shape[0] and 0 <= i_idx < self.train_matrix.shape[1]:
                user_ratings = self.train_matrix[u_idx].toarray().flatten()
                sims = self.sim_matrix[i_idx]
                relevant_indices = np.where(user_ratings > 0)[0]
                if len(relevant_indices) == 0:
                    preds.append(3.53)
                    continue
                relevant_sims = sims[relevant_indices]
                relevant_ratings = user_ratings[relevant_indices]
                if len(relevant_sims) > self.k:
                    top_k_idx = np.argsort(relevant_sims)[-self.k:]
                    relevant_sims = relevant_sims[top_k_idx]
                    relevant_ratings = relevant_ratings[top_k_idx]
                if np.sum(relevant_sims) == 0:
                    preds.append(3.53)
                else:
                    pred = np.dot(relevant_sims, relevant_ratings) / np.sum(relevant_sims)
                    preds.append(pred)
            else:
                preds.append(3.53)
        return np.array(preds)

    def predict_user_scores(self, user_id):
        u_idx = user_id - 1
        if 0 <= u_idx < self.train_matrix.shape[0]:
            user_ratings = self.train_matrix[u_idx] 
            scores = user_ratings.dot(self.sim_matrix)
            return scores
        else:
            return np.zeros(self.sim_matrix.shape[0])

# --- Saving Function ---

def save_artifacts(ratings, movies, results, svd, nmf, knn, ncf, autorec, n_users, n_items):
    print(f"\nSaving artifacts to {MODELS_DIR}...")
    if not os.path.exists(MODELS_DIR):
        os.makedirs(MODELS_DIR)

    # 1. Data Mappings
    mappings = {'movies': movies, 'ratings': ratings}
    with open(os.path.join(MODELS_DIR, 'data_mappings.pkl'), 'wb') as f:
        pickle.dump(mappings, f)

    # 2. Metrics
    metrics = []
    for m in results:
        metrics.append({
            'algorithm': m['Model'],
            'rmse': m['RMSE'],
            'mae': m['MAE'],
            'training_time': 0 
        })
    with open(os.path.join(MODELS_DIR, 'model_metrics.json'), 'w') as f:
        json.dump(metrics, f, indent=4)

    # 3. SVD Model
    if svd and svd.user_factors is not None:
        svd_data = {
            'user_features': svd.user_factors,
            'item_features': svd.item_factors,
            'user_index': pd.Index(range(1, n_users + 1), name='user_id'),
            'item_index': pd.Index(range(1, n_items + 1), name='item_id')
        }
        with open(os.path.join(MODELS_DIR, 'svd_model.pkl'), 'wb') as f:
            pickle.dump(svd_data, f)
            
    # 4. NMF Model
    if nmf and nmf.user_factors is not None:
        nmf_data = {
            'user_features': nmf.user_factors,
            'item_features': nmf.item_factors,
            'user_index': pd.Index(range(1, n_users + 1), name='user_id'),
            'item_index': pd.Index(range(1, n_items + 1), name='item_id')
        }
        with open(os.path.join(MODELS_DIR, 'nmf_model.pkl'), 'wb') as f:
            pickle.dump(nmf_data, f)

    # 5. Similarity Matrix
    if knn and knn.sim_matrix is not None:
        sim_df = pd.DataFrame(
            knn.sim_matrix,
            index=range(1, n_items + 1),
            columns=range(1, n_items + 1)
        )
        with open(os.path.join(MODELS_DIR, 'similarity_matrix.pkl'), 'wb') as f:
            pickle.dump(sim_df, f) 

    # 6. NCF Model (PyTorch)
    if ncf:
        torch.save(ncf.state_dict(), os.path.join(MODELS_DIR, 'ncf_model.pth'))

    # 7. AutoRec Model (PyTorch)
    if autorec:
        # Save model state and the R matrix (needed for input)
        # But R matrix is huge? No, 1682x943 floats is ~6MB.
        # Actually we just need to reconstruct R from ratings in the app.
        # But saving it might be faster.
        # Let's save model state only, and rebuild R in app using load_data.
        torch.save(autorec.state_dict(), os.path.join(MODELS_DIR, 'autorec_model.pth'))
            
    print("Artifacts saved successfully.")

# --- 4. Main Execution Pipeline ---

def run_benchmark():
    ratings, movies = load_data()
    if ratings is None: return

    n_users = ratings['user_id'].max()
    n_items = ratings['item_id'].max()
    print(f"Stats: {n_users} users, {n_items} items.")

    train_df, test_df = get_train_test_split(ratings)
    train_matrix = create_csr_matrix(train_df, n_users, n_items)
    
    y_true = test_df['rating'].values
    test_user_ids = test_df['user_id'].values
    test_item_ids = test_df['item_id'].values

    results = []
    
    # Store models for saving
    svd_model = None
    nmf_model = None
    knn_model = None
    ncf_model = None
    autorec_model = None

    # 1. Baseline
    print("Evaluating Baseline...")
    try:
        baseline = BaselineModel(train_df)
        preds = baseline.predict(test_user_ids, test_item_ids)
        rmse, mae = calculate_metrics(y_true, preds)
        results.append({'Model': 'Baseline', 'RMSE': rmse, 'MAE': mae, 'P@10': 0.0, 'R@10': 0.0})
    except Exception as e:
        print(f"Baseline failed: {e}")

    # 2. SVD
    print("Evaluating SVD...")
    try:
        svd_model = SVDModel(n_components=20)
        svd_model.fit(train_matrix)
        preds = svd_model.predict(test_user_ids, test_item_ids)
        rmse, mae = calculate_metrics(y_true, preds)
        pk, rk = calculate_ranking_metrics(svd_model.predict_user_scores, test_df, train_df, n_users, n_items)
        results.append({'Model': 'SVD', 'RMSE': rmse, 'MAE': mae, 'P@10': pk, 'R@10': rk})
    except Exception as e:
        print(f"SVD failed: {e}")
        traceback.print_exc()

    # 3. NMF
    print("Evaluating NMF...")
    try:
        nmf_model = NMFModel(n_components=20)
        nmf_model.fit(train_matrix)
        preds = nmf_model.predict(test_user_ids, test_item_ids)
        rmse, mae = calculate_metrics(y_true, preds)
        pk, rk = calculate_ranking_metrics(nmf_model.predict_user_scores, test_df, train_df, n_users, n_items)
        results.append({'Model': 'NMF', 'RMSE': rmse, 'MAE': mae, 'P@10': pk, 'R@10': rk})
    except Exception as e:
        print(f"NMF failed: {e}")
        traceback.print_exc()

    # 4. KNN
    print("Evaluating KNN...")
    try:
        knn_model = KNNModel(k=20)
        knn_model.fit(train_matrix)
        preds = knn_model.predict(test_user_ids, test_item_ids)
        rmse, mae = calculate_metrics(y_true, preds)
        pk, rk = calculate_ranking_metrics(knn_model.predict_user_scores, test_df, train_df, n_users, n_items)
        results.append({'Model': 'KNN', 'RMSE': rmse, 'MAE': mae, 'P@10': pk, 'R@10': rk})
    except Exception as e:
        print(f"KNN failed: {e}")
        traceback.print_exc()
        
    # 5. NCF (PyTorch)
    print("Evaluating NCF (Deep Learning)...")
    try:
        ncf_model = train_ncf_model(train_df, n_users, n_items, epochs=7) # 7 Epochs for quick benchmark
        preds = evaluate_ncf(ncf_model, test_df)
        rmse, mae = calculate_metrics(y_true, preds)
        
        # Helper lambda for NCF ranking
        ncf_predict_func = lambda uid: predict_ncf_user_scores(ncf_model, uid, n_items)
        
        pk, rk = calculate_ranking_metrics(ncf_predict_func, test_df, train_df, n_users, n_items)
        results.append({'Model': 'NCF', 'RMSE': rmse, 'MAE': mae, 'P@10': pk, 'R@10': rk})
    except Exception as e:
        print(f"NCF failed: {e}")
        traceback.print_exc()

    # 6. AutoRec
    print("Evaluating AutoRec (Deep Learning)...")
    try:
        autorec_model, R_matrix = train_autorec_model(train_df, n_users, n_items, epochs=50)
        
        # For prediction, we need to pass the model and the full R matrix (or handle it inside)
        # But predict_autorec_user_scores takes (model, R, user_id, n_items)
        # We need a way to get predictions for test set (u, i) pairs for RMSE
        # Evaluate function?
        
        # Let's define a quick evaluate for AutoRec
        def evaluate_autorec(model, R, test_df):
            model.eval()
            R_tensor = torch.FloatTensor(R)
            with torch.no_grad():
                reconstructed = model(R_tensor) # (n_items, n_users)
                
            reconstructed_np = reconstructed.numpy()
            
            preds = []
            test_users = test_df['user_id'].values - 1
            test_items = test_df['item_id'].values - 1
            
            for u, i in zip(test_users, test_items):
                if u < R.shape[1] and i < R.shape[0]:
                    preds.append(reconstructed_np[i, u])
                else:
                    preds.append(3.5) # Fallback
            return np.array(preds)

        preds = evaluate_autorec(autorec_model, R_matrix, test_df)
        rmse, mae = calculate_metrics(y_true, preds)
        
        # Ranking metrics
        autorec_predict_func = lambda uid: predict_autorec_user_scores(autorec_model, R_matrix, uid, n_items)
        pk, rk = calculate_ranking_metrics(autorec_predict_func, test_df, train_df, n_users, n_items)
        
        results.append({'Model': 'AutoRec', 'RMSE': rmse, 'MAE': mae, 'P@10': pk, 'R@10': rk})
    except Exception as e:
        print(f"AutoRec failed: {e}")
        traceback.print_exc()

    # Display Results
    print("\n=== BENCHMARK RESULTS ===")
    results_df = pd.DataFrame(results)
    print(results_df.to_string(index=False))
    
    # Save Artifacts
    save_artifacts(ratings, movies, results, svd_model, nmf_model, knn_model, ncf_model, autorec_model, n_users, n_items)

if __name__ == "__main__":
    run_benchmark()
