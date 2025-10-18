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
LLM_PROVIDER = os.getenv('LLM_PROVIDER', 'openai')  # options: 'openai', 'anthropic', 'gemini', 'local', 'fake'
USE_FAKE_LLM_FOR_TESTS = os.getenv('USE_FAKE_LLM_FOR_TESTS', 'false').lower() == 'true'
LLM_MODEL = os.getenv('LLM_MODEL', 'gpt-4-turbo-preview')  # or 'claude-3-sonnet-20240229'
LLM_TEMPERATURE = float(os.getenv('LLM_TEMPERATURE', '0.3'))  # lower for factual legal responses
LLM_MAX_TOKENS = int(os.getenv('LLM_MAX_TOKENS', '2000'))
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
ANTHROPIC_API_KEY = os.getenv('ANTHROPIC_API_KEY')
GEMINI_API_KEY = os.getenv('GEMINI_API_KEY')
GEMINI_MODEL = os.getenv('GEMINI_MODEL', 'gemini-1.5-pro')
LOCAL_LLM_ENDPOINT = os.getenv('LOCAL_LLM_ENDPOINT', 'http://localhost:8000')
LOCAL_LLM_MODEL = os.getenv('LOCAL_LLM_MODEL', 'llama-2-7b-chat')

# --- Retrieval Configuration ---
TOP_K_CHUNKS = int(os.getenv('TOP_K_CHUNKS', '5'))  # number of FAISS chunks to retrieve
TOP_K_SQL_RESULTS = int(os.getenv('TOP_K_SQL_RESULTS', '10'))  # number of SQL keyword matches
SIMILARITY_THRESHOLD = float(os.getenv('SIMILARITY_THRESHOLD', '0.6'))  # minimum cosine similarity (range: -1 to 1)
HYBRID_WEIGHT_SEMANTIC = float(os.getenv('HYBRID_WEIGHT_SEMANTIC', '0.6'))  # weight for semantic search
HYBRID_WEIGHT_KEYWORD = float(os.getenv('HYBRID_WEIGHT_KEYWORD', '0.4'))  # weight for keyword search
CRIME_WEIGHT_HYBRID = float(os.getenv('CRIME_WEIGHT_HYBRID', '0.2'))  # weight for crime category results in RRF

# --- Chat Configuration ---
MAX_HISTORY_TURNS = int(os.getenv('MAX_HISTORY_TURNS', '5'))  # number of conversation turns in context
MAX_CONTEXT_LENGTH = int(os.getenv('MAX_CONTEXT_LENGTH', '4000'))  # tokens for context window
SESSION_TIMEOUT = int(os.getenv('SESSION_TIMEOUT', '3600'))  # seconds before session expires

# --- Logging Configuration ---
LOG_LEVEL = os.getenv('LOG_LEVEL', 'INFO')
LOG_FILE = os.getenv('LOG_FILE', 'logs/rag_system.log')

# --- API Configuration ---
API_HOST = os.getenv('API_HOST', '0.0.0.0')
API_PORT = int(os.getenv('API_PORT', '8000'))
API_WORKERS = int(os.getenv('API_WORKERS', '4'))
CORS_ORIGINS = os.getenv('CORS_ORIGINS', '*').split(',')  # Split comma-separated list

# --- Caching Configuration ---
# Cache version - increment when config changes affect cached results (model, thresholds, etc.)
CACHE_VERSION = os.getenv('CACHE_VERSION', 'v1')
CACHE_ENABLED = os.getenv('CACHE_ENABLED', 'false').lower() == 'true'
CACHE_BACKEND = os.getenv('CACHE_BACKEND', 'memory')  # 'memory' or 'redis'
CACHE_TTL_QUERY = int(os.getenv('CACHE_TTL_QUERY', '3600'))  # 1 hour
CACHE_TTL_LLM = int(os.getenv('CACHE_TTL_LLM', '7200'))  # 2 hours
CACHE_TTL_EMBEDDING = int(os.getenv('CACHE_TTL_EMBEDDING', '86400'))  # 24 hours
REDIS_URL = os.getenv('REDIS_URL', 'redis://localhost:6379/0')
# Cache size limits (in-memory only; Redis bounded by server maxmemory)
CACHE_MAX_ITEMS_QUERY = int(os.getenv('CACHE_MAX_ITEMS_QUERY', '5000'))
CACHE_MAX_ITEMS_LLM = int(os.getenv('CACHE_MAX_ITEMS_LLM', '2000'))
CACHE_MAX_ITEMS_EMBEDDING = int(os.getenv('CACHE_MAX_ITEMS_EMBEDDING', '5000'))

# --- Advanced Ranking Configuration ---
USE_BM25 = os.getenv('USE_BM25', 'false').lower() == 'true'
USE_RRF = os.getenv('USE_RRF', 'false').lower() == 'true'
BM25_K1 = float(os.getenv('BM25_K1', '1.5'))  # Term frequency saturation
BM25_B = float(os.getenv('BM25_B', '0.75'))  # Length normalization
BM25_CORPUS_LIMIT = int(os.getenv('BM25_CORPUS_LIMIT', '1000'))  # Max documents for BM25 corpus
RRF_K = int(os.getenv('RRF_K', '60'))  # RRF constant

# --- Performance Monitoring ---
ENABLE_METRICS = os.getenv('ENABLE_METRICS', 'false').lower() == 'true'
METRICS_PORT = int(os.getenv('METRICS_PORT', '9090'))

# --- Database Optimization ---
USE_CONNECTION_POOL = os.getenv('USE_CONNECTION_POOL', 'false').lower() == 'true'

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
    
    Returns tuple of (errors, warnings) where:
    - errors: Critical issues that prevent operation
    - warnings: Non-critical issues or informational messages
    """
    errors = []
    warnings = []
    
    # Critical: LLM API keys
    if LLM_PROVIDER == 'openai' and not OPENAI_API_KEY:
        errors.append("OPENAI_API_KEY is required when LLM_PROVIDER is 'openai'")
    
    if LLM_PROVIDER == 'anthropic' and not ANTHROPIC_API_KEY:
        errors.append("ANTHROPIC_API_KEY is required when LLM_PROVIDER is 'anthropic'")
    
    if LLM_PROVIDER == 'gemini' and not GEMINI_API_KEY:
        errors.append("GEMINI_API_KEY is required when LLM_PROVIDER is 'gemini'")
    
    # Critical: Database
    if not os.path.exists(DATABASE_PATH):
        errors.append(f"Database not found at {DATABASE_PATH}")
    
    # Critical: Cache backend validation
    if CACHE_BACKEND not in ['memory', 'redis']:
        errors.append(f"CACHE_BACKEND must be 'memory' or 'redis', got '{CACHE_BACKEND}'")
    
    if CACHE_BACKEND == 'redis' and not REDIS_URL:
        errors.append("REDIS_URL required when CACHE_BACKEND is 'redis'")
    
    # Warning: Advanced ranking (experimental features)
    if USE_BM25:
        warnings.append("BM25 ranking enabled (experimental feature)")
    
    if USE_RRF:
        warnings.append("RRF fusion enabled (experimental feature)")
    
    # Warning: FAISS (allows degraded mode with keyword-only search)
    if not os.path.exists(FAISS_INDEX_PATH):
        warnings.append(f"FAISS index not found at {FAISS_INDEX_PATH} - semantic search disabled (degraded mode)")
    
    if not os.path.exists(CHUNKS_MAP_PATH):
        warnings.append(f"Chunks map not found at {CHUNKS_MAP_PATH} - semantic search disabled (degraded mode)")
    
    # Warning: API configuration validation
    if API_PORT < 1 or API_PORT > 65535:
        warnings.append(f"API_PORT must be between 1 and 65535, got {API_PORT}")
    
    if API_WORKERS < 1 or API_WORKERS > 16:
        warnings.append(f"API_WORKERS should be between 1 and 16, got {API_WORKERS}")
    
    return errors, warnings

if __name__ == '__main__':
    errors, warnings = validate_config()
    
    if errors:
        print("❌ Configuration Errors (Critical):")
        for error in errors:
            print(f"  - {error}")
    
    if warnings:
        print("\n⚠️  Configuration Warnings:")
        for warning in warnings:
            print(f"  - {warning}")
    
    if not errors and not warnings:
        print("✅ Configuration validated successfully")
    elif not errors:
        print("\n✅ Configuration OK (with warnings)")
    else:
        print("\n❌ Configuration has critical errors")
