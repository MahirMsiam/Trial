from rag_retriever import HybridRetriever

print("Testing HybridRetriever initialization...")
try:
    retriever = HybridRetriever()
    print("✅ HybridRetriever initialized successfully")
    
    print("\nTesting _get_db_connection...")
    with retriever._get_db_connection() as (cursor, conn):
        cursor.execute("SELECT COUNT(*) FROM judgments")
        count = cursor.fetchone()[0]
        print(f"✅ Database connection works! Found {count} judgments")
    
    print("\n✅ All tests passed!")
except Exception as e:
    print(f"❌ Error: {e}")
    import traceback
    traceback.print_exc()
