import sqlite3
import json
import re
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
import time
import os

# --- Configuration ---
DATABASE_PATH = os.path.join('extracted_data', 'legal_library.db')
# CORRECTED: The table name is now 'judgments'
TABLE_NAME = 'judgments'
MODEL_NAME = 'all-MiniLM-L6-v2'
FAISS_INDEX_PATH = 'faiss_index.bin'
CHUNKS_MAP_PATH = 'chunks_map.json'

def intelligent_chunking(text):
    """Cleans and chunks the text from the database."""
    if not text:
        return []
    text = re.sub(r'\[Page \d+\]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    
    paragraphs = text.split('\n\n')
    
    final_chunks = [para.strip() for para in paragraphs if len(para.strip()) > 150]
            
    return final_chunks

def main():
    """Main function to connect to the DB, process documents, and build the index."""
    print("--- Starting FAISS Index Creation Process ---")
    
    if not os.path.exists(DATABASE_PATH):
        print(f"❌ Fatal Error: Database not found at '{DATABASE_PATH}'.")
        return

    print(f"Loading sentence transformer model: '{MODEL_NAME}'...")
    try:
        model = SentenceTransformer(MODEL_NAME)
    except Exception as e:
        print(f"❌ Fatal Error: Could not load the embedding model. Error: {e}")
        return

    print(f"Connecting to database and reading from table '{TABLE_NAME}'...")
    try:
        conn = sqlite3.connect(DATABASE_PATH)
        cursor = conn.cursor()
        cursor.execute(f"SELECT id, full_text FROM {TABLE_NAME}")
        all_cases = cursor.fetchall()
        conn.close()
        print(f"✅ Successfully fetched {len(all_cases)} documents.")
    except sqlite3.Error as e:
        print(f"❌ Fatal Error: Could not read from the database. Error: {e}")
        return

    print("\nProcessing all documents (chunking text)...")
    all_chunks_text = []
    chunk_map = {} 
    chunk_index_counter = 0

    for case_id, full_text in all_cases:
        chunks = intelligent_chunking(full_text)
        for chunk_text in chunks:
            all_chunks_text.append(chunk_text)
            chunk_map[chunk_index_counter] = {
                "case_id": case_id,
                "text": chunk_text
            }
            chunk_index_counter += 1
    
    if not all_chunks_text:
        print("❌ No valid chunks were created. Aborting.")
        return
        
    print(f"✅ Total chunks created: {len(all_chunks_text)}")

    print("\nGenerating embeddings... (This will take time)")
    start_time = time.time()
    embeddings = model.encode(all_chunks_text, show_progress_bar=True)
    end_time = time.time()
    print(f"✅ Embeddings generated in {end_time - start_time:.2f} seconds.")

    dimension = embeddings.shape[1]
    print(f"\nBuilding FAISS index (Dimension: {dimension})...")
    index = faiss.IndexFlatL2(dimension)
    index.add(np.array(embeddings, dtype='float32'))
    
    print(f"Saving FAISS index to '{FAISS_INDEX_PATH}'...")
    faiss.write_index(index, FAISS_INDEX_PATH)
    print(f"✅ FAISS index saved. Total vectors: {index.ntotal}")

    print(f"Saving chunk map to '{CHUNKS_MAP_PATH}'...")
    with open(CHUNKS_MAP_PATH, 'w', encoding='utf-8') as f:
        json.dump(chunk_map, f)
    print("✅ Chunk map saved.")
    
    print("\n--- Indexing Process Complete! ---")

if __name__ == '__main__':
    main()

