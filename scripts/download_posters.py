import pandas as pd
import requests
import os
import re
import time
import pickle
from concurrent.futures import ThreadPoolExecutor

# Configuration
POSTER_DIR = "assets/posters"
MAPPING_PATH = "models/data_mappings.pkl"
LOG_FILE = "download_imdb.log"

def log_debug(msg):
    with open(LOG_FILE, "a", encoding="utf-8") as f:
        f.write(f"[{time.strftime('%H:%M:%S')}] {msg}\n")
    print(msg, flush=True)

def get_clean_title(title):
    year_match = re.search(r'\((\d{4})\)', title)
    year = year_match.group(1) if year_match else None
    clean_title = re.sub(r'\s*\(\d{4}\)', '', title).strip()
    # Normalize special names (The Toy Story -> Toy Story)
    if ", The" in clean_title: clean_title = "The " + clean_title.replace(", The", "")
    if ", A" in clean_title: clean_title = "A " + clean_title.replace(", A", "")
    return clean_title, year

def download_poster(row):
    movie_id = row['movie_id']
    title = row['title']
    target_path = os.path.join(POSTER_DIR, f"movie_{movie_id}.jpg")
    
    if os.path.exists(target_path):
        return "Exists"
    
    clean_title, year = get_clean_title(title)
    # IMDb suggestion API slug
    slug = re.sub(r'[^a-z0-9]', '_', clean_title.lower()).replace('__', '_').strip('_')
    if not slug: return "Invalid Slug"
    
    first_char = slug[0]
    api_url = f"https://v2.sg.media-imdb.com/suggestion/{first_char}/{slug}.json"
    
    try:
        response = requests.get(api_url, timeout=10)
        if response.status_code == 200:
            data = response.json()
            results = data.get('d', [])
            if not results:
                log_debug(f"NOT FOUND: {title} (No results for slug '{slug}')")
                return "Not Found"
            
            # Find best match
            best_match = None
            for res in results:
                if res.get('qid') == 'movie':
                    # Check year if possible
                    if year and str(res.get('y')) == str(year):
                        best_match = res
                        break
                    if not best_match:
                        best_match = res
            
            if best_match and 'i' in best_match:
                poster_url = best_match['i']['imageUrl']
                log_debug(f"Found {title} -> {poster_url[:50]}...")
                
                img_res = requests.get(poster_url, timeout=15)
                if img_res.status_code == 200:
                    with open(target_path, 'wb') as f:
                        f.write(img_res.content)
                    return "Downloaded"
                else:
                    log_debug(f"FAIL: Image download error {img_res.status_code} for {title}")
            else:
                log_debug(f"NO IMAGE: {title} found but no image URL.")
        else:
            log_debug(f"API ERROR {response.status_code} for {title}")
    except Exception as e:
        log_debug(f"EXCEPTION {title}: {str(e)}")
    
    return "Failed"

def main():
    if not os.path.exists(POSTER_DIR):
        os.makedirs(POSTER_DIR)
    
    # Reset log
    with open(LOG_FILE, "w", encoding="utf-8") as f:
        f.write("--- IMDb Download Session Started ---\n")
        
    log_debug(f"Loading movies from {MAPPING_PATH}...")
    if not os.path.exists(MAPPING_PATH):
        print(f"Error: {MAPPING_PATH} not found.")
        return
        
    with open(MAPPING_PATH, 'rb') as f:
        mappings = pickle.load(f)
        df = mappings['movies']
        
    log_debug(f"Starting download for {len(df)} movies (Parallel processing)...")
    
    with ThreadPoolExecutor(max_workers=10) as executor:
        results = list(executor.map(download_poster, [row for _, row in df.iterrows()]))
    
    success = sum(1 for r in results if r == "Downloaded")
    exists = sum(1 for r in results if r == "Exists")
    failed = len(results) - success - exists
    
    log_debug("\n--- Summary ---")
    log_debug(f"Downloaded: {success}")
    log_debug(f"Existed: {exists}")
    log_debug(f"Failed: {failed}")

if __name__ == "__main__":
    main()
