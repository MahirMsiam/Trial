import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# --- Database and Index Paths ---
DATABASE_PATH = 'extracted_data/database.db'
FAISS_INDEX_PATH = 'faiss_index.bin'
CHUNKS_MAP_PATH = 'chunks_map.json'

# --- Embedding Model Configuration ---
EMBEDDING_MODEL = 'all-MiniLM-L6-v2'  # sentence-transformers model
EMBEDDING_DIMENSION = 384  # dimension for all-MiniLM-L6-v2

# --- LLM Configuration ---
LLM_PROVIDER = os.getenv('LLM_PROVIDER', 'openai')  # options: 'openai', 'anthropic', 'local'
LLM_MODEL = os.getenv('LLM_MODEL', 'gpt-4-turbo-preview')  # or 'claude-3-sonnet-20240229'
LLM_TEMPERATURE = float(os.getenv('LLM_TEMPERATURE', '0.3'))  # lower for factual legal responses
LLM_MAX_TOKENS = int(os.getenv('LLM_MAX_TOKENS', '2000'))
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
ANTHROPIC_API_KEY = os.getenv('ANTHROPIC_API_KEY')
LOCAL_LLM_ENDPOINT = os.getenv('LOCAL_LLM_ENDPOINT', 'http://localhost:8000')
LOCAL_LLM_MODEL = os.getenv('LOCAL_LLM_MODEL', 'llama-2-7b-chat')

# --- Retrieval Configuration ---
TOP_K_CHUNKS = int(os.getenv('TOP_K_CHUNKS', '5'))  # number of FAISS chunks to retrieve
TOP_K_SQL_RESULTS = int(os.getenv('TOP_K_SQL_RESULTS', '10'))  # number of SQL keyword matches
SIMILARITY_THRESHOLD = float(os.getenv('SIMILARITY_THRESHOLD', '0.6'))  # minimum cosine similarity (range: -1 to 1)
HYBRID_WEIGHT_SEMANTIC = float(os.getenv('HYBRID_WEIGHT_SEMANTIC', '0.6'))  # weight for semantic search
HYBRID_WEIGHT_KEYWORD = float(os.getenv('HYBRID_WEIGHT_KEYWORD', '0.4'))  # weight for keyword search

# --- Chat Configuration ---
MAX_HISTORY_TURNS = int(os.getenv('MAX_HISTORY_TURNS', '5'))  # number of conversation turns in context
MAX_CONTEXT_LENGTH = int(os.getenv('MAX_CONTEXT_LENGTH', '4000'))  # tokens for context window
SESSION_TIMEOUT = int(os.getenv('SESSION_TIMEOUT', '3600'))  # seconds before session expires

# --- Logging Configuration ---
LOG_LEVEL = os.getenv('LOG_LEVEL', 'INFO')
LOG_FILE = os.getenv('LOG_FILE', 'logs/rag_system.log')

# --- Import Crime Keywords ---
try:
    from crime_keywords import CRIME_KEYWORDS
except ImportError:
    CRIME_KEYWORDS = {}
    print("Warning: Could not import CRIME_KEYWORDS from crime_keywords.py")

# --- Prompt Templates ---
SYSTEM_PROMPT_TEMPLATE = """You are a legal research assistant specializing in Bangladesh Supreme Court judgments.
Your role is to provide accurate, well-cited answers based ONLY on the provided judgment database.

Key Guidelines:
- Base all responses on the provided context from judgments
- Always cite case numbers, dates, and parties when referencing judgments
- Distinguish between High Court Division and Appellate Division cases
- Use formal legal language appropriate for Bangladesh legal system
- Handle both English and Bengali legal terms correctly
- Acknowledge when information is not available in the database
- Do NOT provide legal advice beyond what is stated in judgments
- Do NOT speculate about hypothetical cases

Citation Format: [Case: {case_number}, Parties: {petitioner} vs {respondent}, Date: {judgment_date}]
"""

CITATION_FORMAT = "[Case: {case_number}, Parties: {petitioner} vs {respondent}, Date: {judgment_date}]"

# Validation function
def validate_config():
    """
    Validate that required configuration is present.
    
    Returns warnings for missing FAISS index (allows degraded mode)
    but errors for critical issues like missing LLM keys or database.
    """
    errors = []
    
    if LLM_PROVIDER == 'openai' and not OPENAI_API_KEY:
        errors.append("OPENAI_API_KEY is required when LLM_PROVIDER is 'openai'")
    
    if LLM_PROVIDER == 'anthropic' and not ANTHROPIC_API_KEY:
        errors.append("ANTHROPIC_API_KEY is required when LLM_PROVIDER is 'anthropic'")
    
    if not os.path.exists(DATABASE_PATH):
        errors.append(f"Database not found at {DATABASE_PATH}")
    
    # Soft warnings for FAISS (allows degraded mode with keyword-only search)
    if not os.path.exists(FAISS_INDEX_PATH):
        errors.append(f"FAISS index not found at {FAISS_INDEX_PATH} (degraded mode available)")
    
    if not os.path.exists(CHUNKS_MAP_PATH):
        errors.append(f"Chunks map not found at {CHUNKS_MAP_PATH} (degraded mode available)")
    
    return errors

if __name__ == '__main__':
    errors = validate_config()
    if errors:
        print("❌ Configuration Errors:")
        for error in errors:
            print(f"  - {error}")
    else:
        print("✅ Configuration validated successfully")
