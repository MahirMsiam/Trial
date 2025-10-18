import sqlite3
import json
import re
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
import time
import os
import argparse

# --- Configuration ---
# Updated paths to match your directory structure
DATABASE_PATH = os.path.join('extracted_data', 'database.db')
TABLE_NAME = 'judgments'
MODEL_NAME = 'all-MiniLM-L6-v2'

# Output files will be in the root project folder
FAISS_INDEX_PATH = 'faiss_index.bin'
CHUNKS_MAP_PATH = 'chunks_map.json'

def intelligent_chunking(text):
    """Cleans and chunks the text from the database, preserving paragraph boundaries."""
    if not text:
        return []
    
    # Step 1: Remove page markers
    text = re.sub(r'\[Page \d+\]', '', text)
    
    # Step 2: Normalize CRLF to LF
    text = text.replace('\r\n', '\n').replace('\r', '\n')
    
    # Step 3: Collapse 3+ newlines to exactly two (preserve paragraph boundaries)
    text = re.sub(r'\n{3,}', '\n\n', text)
    
    # Step 4: Normalize spaces within lines only
    lines = text.split('\n')
    normalized_lines = [re.sub(r'[ \t]+', ' ', line).strip() for line in lines]
    
    # Step 5: Re-join and split by paragraph boundaries
    text = '\n'.join(normalized_lines)
    paragraphs = text.split('\n\n')
    paragraphs = [p.strip() for p in paragraphs if p.strip()]
    
    # Step 6: Build chunks of ~800-1000 chars by concatenating paragraphs
    final_chunks = []
    current_chunk = ""
    
    for para in paragraphs:
        # If adding this paragraph would exceed 1000 chars, save current chunk
        if current_chunk and len(current_chunk) + len(para) + 2 > 1000:
            if len(current_chunk) >= 150:
                final_chunks.append(current_chunk)
            current_chunk = para
        else:
            # Add paragraph to current chunk
            if current_chunk:
                current_chunk += "\n\n" + para
            else:
                current_chunk = para
    
    # Add the last chunk if it's substantial
    if current_chunk and len(current_chunk) >= 150:
        final_chunks.append(current_chunk)
    
    return final_chunks

def main():
    """Main function to connect to the DB, process documents, and build the index."""
    print("--- Starting FAISS Index Creation Process ---")
    
    if not os.path.exists(DATABASE_PATH):
        print(f"❌ Fatal Error: Database not found at '{DATABASE_PATH}'.")
        print("Please ensure your 'database.db' file is inside the 'extracted_data' folder.")
        print("Run 'batch_processor.py' first to create the database.")
        return

    # 1. Load Embedding Model
    print(f"Loading sentence transformer model: '{MODEL_NAME}'...")
    try:
        model = SentenceTransformer(MODEL_NAME)
    except Exception as e:
        print(f"❌ Fatal Error: Could not load the embedding model. Error: {e}")
        return

    # 2. Connect to SQLite Database and Fetch Data
    print(f"Connecting to database: '{DATABASE_PATH}'...")
    try:
        conn = sqlite3.connect(DATABASE_PATH)
        cursor = conn.cursor()
        
        # Sanity check: verify judgments table exists
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='judgments'")
        if cursor.fetchone() is None:
            print(f"❌ Fatal Error: Table 'judgments' does not exist in the database.")
            print("Please run 'batch_processor.py' first to create the database schema.")
            conn.close()
            return
        
        # Try/fallback pattern for full_case_id column
        with_full_case_id = True
        try:
            cursor.execute(f"SELECT id, full_text, case_number, case_type, judgment_date, petitioner_name, respondent_name, full_case_id FROM {TABLE_NAME}")
            all_cases = cursor.fetchall()
        except sqlite3.OperationalError:
            with_full_case_id = False
            cursor.execute(f"SELECT id, full_text, case_number, case_type, judgment_date, petitioner_name, respondent_name FROM {TABLE_NAME}")
            all_cases = cursor.fetchall()
            print(f"⚠️  Warning: Column 'full_case_id' not found. Will construct from case_type and case_number.")
        
        conn.close()
        print(f"✅ Successfully fetched {len(all_cases)} documents from the database.")
            
    except sqlite3.Error as e:
        print(f"❌ Fatal Error: Could not read from the database. Error: {e}")
        print("Ensure the database schema includes the required columns.")
        return

    # 3. Process all documents and create a map
    print("\nProcessing all documents (chunking text)...")
    all_chunks_text = []
    chunk_map = {} 
    chunk_index_counter = 0

    # Process each case, handling variable column count
    for row in all_cases:
        if with_full_case_id:
            case_id, full_text, case_number, case_type, judgment_date, petitioner_name, respondent_name, full_case_id = row
        else:
            case_id, full_text, case_number, case_type, judgment_date, petitioner_name, respondent_name = row
            full_case_id = f"{case_type or ''} No. {case_number or ''}".strip()
        
        chunks = intelligent_chunking(full_text)
        for chunk_text in chunks:
            all_chunks_text.append(chunk_text)
            chunk_map[chunk_index_counter] = {
                "case_id": case_id,
                "text": chunk_text,
                "case_number": case_number if case_number else "Unknown",
                "case_type": case_type if case_type else "Unknown",
                "judgment_date": judgment_date if judgment_date else "Unknown",
                "petitioner": petitioner_name if petitioner_name else "Unknown",
                "respondent": respondent_name if respondent_name else "Unknown",
                "full_case_id": full_case_id if full_case_id else "Unknown"
            }
            chunk_index_counter += 1
    
    if not all_chunks_text:
        print("❌ No valid chunks were created. Aborting.")
        return
        
    print(f"✅ Total chunks created: {len(all_chunks_text)}")

    # 4. Generate Embeddings in batches
    print("\nGenerating embeddings in batches... (This will take time)")
    start_time = time.time()
    
    embeddings_parts = []
    batch_size = 512
    total_batches = (len(all_chunks_text) + batch_size - 1) // batch_size
    
    for i in range(0, len(all_chunks_text), batch_size):
        batch = all_chunks_text[i:i+batch_size]
        batch_num = (i // batch_size) + 1
        print(f"  Processing batch {batch_num}/{total_batches} ({len(batch)} chunks)...")
        
        emb = model.encode(
            batch,
            batch_size=64,
            show_progress_bar=False,
            convert_to_numpy=True,
            normalize_embeddings=True,  # Normalize for cosine similarity
        ).astype('float32')
        embeddings_parts.append(emb)
    
    embeddings = np.vstack(embeddings_parts)
    end_time = time.time()
    print(f"✅ Embeddings generated in {end_time - start_time:.2f} seconds.")

    # 5. Build and Save FAISS Index (using Inner Product for cosine similarity)
    dimension = embeddings.shape[1]
    print(f"\nBuilding FAISS index (Dimension: {dimension})...")
    index = faiss.IndexFlatIP(dimension)  # Inner Product for normalized vectors = cosine similarity
    index.add(np.array(embeddings, dtype='float32'))
    
    print(f"Saving FAISS index to '{FAISS_INDEX_PATH}'...")
    print(f"✅ FAISS index saved. Total vectors: {index.ntotal}")

    # 6. Save the Chunk Map
    print(f"Saving chunk map to '{CHUNKS_MAP_PATH}'...")
    try:
        with open(CHUNKS_MAP_PATH, 'w', encoding='utf-8') as f:
            json.dump(chunk_map, f, ensure_ascii=False, indent=2)
    except Exception as e:
        print(f"❌ Fatal Error: Failed to write chunk map. Error: {e}")
        return
    
    # Validate chunk map was written correctly
    try:
        with open(CHUNKS_MAP_PATH, 'r', encoding='utf-8') as f:
            loaded_map = json.load(f)
        
        if len(loaded_map) == 0:
            print(f"❌ Fatal Error: Chunk map is empty after writing!")
            print("Please re-run the indexing process.")
            return
        
        print(f"✅ Chunk map saved with {len(chunk_map)} entries.")
    except Exception as e:
        print(f"❌ Fatal Error: Failed to validate chunk map. Error: {e}")
        return
    
    print("\n--- Indexing Process Complete! ---")
    print("\n💡 To invalidate caches after reindexing, run:")
    print("   python -c \"from rag_pipeline import RAGPipeline; RAGPipeline().clear_caches()\"")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Create FAISS index from legal judgments')
    parser.add_argument('--invalidate-cache', action='store_true', 
                       help='Invalidate all caches after successful indexing')
    args = parser.parse_args()
    
    main()
    
    # Optionally invalidate caches if flag is set
    if args.invalidate_cache:
        try:
            print("\nInvalidating caches...")
            from rag_pipeline import RAGPipeline
            pipeline = RAGPipeline()
            pipeline.clear_caches()
            print("✅ Caches invalidated successfully")
        except Exception as e:
            print(f"⚠️  Failed to invalidate caches: {e}")
            print("   You can manually run: python -c \"from rag_pipeline import RAGPipeline; RAGPipeline().clear_caches()\"")


