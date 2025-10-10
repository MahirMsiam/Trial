# Bangladesh Supreme Court RAG System

A Retrieval-Augmented Generation (RAG) system for intelligent semantic search and question answering across Bangladesh Supreme Court judgments.

## Overview

This RAG system enables natural language queries over a database of legal judgments using:
- **Hybrid Retrieval**: Combines FAISS semantic search with SQL keyword search
- **Crime Categorization**: Specialized search for crime-related cases
- **LLM Generation**: OpenAI, Anthropic, or local LLM support for generating cited responses
- **Conversation Management**: Multi-turn dialogue with context preservation
- **Source Citations**: All responses include references to source judgments

## Architecture

```
User Query → Query Processing → Hybrid Retrieval (FAISS + SQL + Crime Keywords)
                                        ↓
                              Context Assembly + History
                                        ↓
                              LLM Generation (OpenAI/Anthropic/Local)
                                        ↓
                              Response with Citations → User
```

## Prerequisites

- **Python 3.8+**
- **Completed PDF Processing**: Database must exist from `batch_processor.py`
- **FAISS Index**: Created using `create_index.py`
- **API Key**: OpenAI or Anthropic (or local LLM setup)

## Installation

### Step 1: Install Dependencies

```bash
pip install -r requirements_rag.txt
```

### Step 2: Configure Environment

```bash
copy .env.example .env
```

Edit `.env` and add your API keys:

```env
OPENAI_API_KEY=sk-your-actual-key-here
# OR
ANTHROPIC_API_KEY=sk-ant-your-actual-key-here
```

### Step 3: Verify Database and Index

Ensure your database exists:
```bash
dir extracted_data\database.db
```

If database doesn't exist, run the batch processor first:
```bash
python batch_processor.py
```

### Step 4: Create/Update FAISS Index

**IMPORTANT**: The `create_index.py` file has been updated to use the correct database path (`database.db` instead of `legal_library.db`) and now includes rich metadata in chunks.

Run the indexing:
```bash
python create_index.py
```

This will:
- Read all judgments from `extracted_data/database.db`
- Generate embeddings for text chunks
- Create `faiss_index.bin` and `chunks_map.json`
- Include metadata: case numbers, parties, dates, etc.

Verify the index was created:
```bash
dir faiss_index.bin
dir chunks_map.json
```

## Configuration

Edit `config.py` or set environment variables in `.env`:

### LLM Provider
```env
LLM_PROVIDER=openai          # Options: openai, anthropic, local
LLM_MODEL=gpt-4-turbo-preview
LLM_TEMPERATURE=0.3          # Lower for factual responses
LLM_MAX_TOKENS=2000
```

### Retrieval Parameters
```env
TOP_K_CHUNKS=5               # Number of semantic search results
TOP_K_SQL_RESULTS=10         # Number of keyword search results
SIMILARITY_THRESHOLD=0.7     # Minimum similarity score
HYBRID_WEIGHT_SEMANTIC=0.6   # Weight for semantic search
HYBRID_WEIGHT_KEYWORD=0.4    # Weight for keyword search
```

### Conversation Settings
```env
MAX_HISTORY_TURNS=5          # Conversation turns to keep in context
MAX_CONTEXT_LENGTH=4000      # Token limit for context window
SESSION_TIMEOUT=3600         # Session expiry in seconds
```

## Usage

### Interactive CLI Chat

Start the chat interface:
```bash
python chat_interface.py
```

Available commands:
- `/help` - Show help and examples
- `/new` - Start new conversation session
- `/history` - View conversation history
- `/search [crime_type]` - Search crime categories
- `/case [case_id]` - Summarize specific case
- `/quit` or `/exit` - Exit chat

### Programmatic Usage

```python
from rag_pipeline import RAGPipeline

# Initialize pipeline
pipeline = RAGPipeline()

# Simple query
result = pipeline.process_query("What are the requirements for writ petitions?")
print(result['response'])

# Query with session (for multi-turn conversation)
session_id = pipeline.conversation_manager.create_session()
result1 = pipeline.process_query("What is Article 102?", session_id=session_id)
result2 = pipeline.process_query("Can you give examples?", session_id=session_id)

# Crime category search
crime_results = pipeline.search_crime_cases("Show me murder cases")
print(f"Found {crime_results['count']} murder cases")

# Case summarization
summary = pipeline.summarize_case(case_id=12345)
print(summary['summary'])

# Compare cases
comparison = pipeline.compare_cases(case_ids=[123, 456, 789])
print(comparison['comparison'])
```

## Query Examples

### Factual Queries
```
"What are the requirements for filing a writ petition under Article 102?"
"What is the difference between High Court and Appellate Division?"
"Which cases discussed fundamental rights violations?"
```

### Crime Category Searches
```
"Show me all murder cases"
"Find corruption cases from 2023"
"List dowry-related judgments"
```

### Specific Case Queries
```
"Summarize Writ Petition No. 1234 of 2023"
"What was the outcome in [Party Name] vs State?"
"Show details of case number 5678"
```

### Legal Research
```
"Which cases cited Section 302 of the Penal Code?"
"Find judgments about contempt of court"
"What are common outcomes in defamation cases?"
```

### Comparative Analysis
```
"Compare murder cases from 2022 and 2023"
"What are similarities between these corruption cases?"
"How has interpretation of Article 102 evolved?"
```

## File Structure

### New RAG System Files
```
config.py                  - Central configuration
.env.example               - Environment template
rag_retriever.py           - Hybrid retrieval (FAISS + SQL)
llm_client.py              - LLM client abstraction
prompt_templates.py        - Legal domain prompts
conversation_manager.py    - Session and history management
rag_pipeline.py            - Main RAG orchestrator
chat_interface.py          - CLI interface
requirements_rag.txt       - RAG dependencies
README_RAG.md              - This file
tests/test_rag_pipeline.py - Unit tests
```

### Integration with Existing Files
- **create_index.py** (MODIFIED): Now uses correct database path and includes metadata
- **batch_processor.py**: Creates the database
- **search_database.py**: Keyword search (used by hybrid retriever)
- **crime_keywords.py**: Crime categorization (used by retriever)

## Troubleshooting

### "Database not found"
**Problem**: `extracted_data/database.db` doesn't exist

**Solution**: Run the batch processor first:
```bash
python batch_processor.py
```

### "FAISS index not found"
**Problem**: `faiss_index.bin` or `chunks_map.json` missing

**Solution**: Run the indexing script:
```bash
python create_index.py
```

### "Empty chunks_map" or "no such table: cases"
**Problem**: Database path mismatch or wrong table name

**Solution**: This has been fixed in the updated `create_index.py`. Re-run:
```bash
python create_index.py
```

The script now correctly uses:
- Database: `extracted_data/database.db`
- Table: `judgments`

### "API key error"
**Problem**: LLM API key not configured

**Solution**: 
1. Copy `.env.example` to `.env`
2. Add your actual API key
3. Ensure no extra spaces or quotes

### "No relevant results found"
**Problem**: Query doesn't match any judgments

**Solutions**:
- Try different keywords or legal terminology
- Use broader search terms
- Check if database contains relevant cases
- Try crime category search: `/search [crime_type]`

### "LLM timeout" or slow responses
**Problem**: Large context or slow API

**Solutions**:
- Reduce `TOP_K_CHUNKS` in config (try 3 instead of 5)
- Reduce `MAX_CONTEXT_LENGTH`
- Use faster model (gpt-3.5-turbo instead of gpt-4)

## Performance Tuning

### Retrieval Quality
Adjust in `config.py` or `.env`:

```env
# For more precise results
SIMILARITY_THRESHOLD=0.8
HYBRID_WEIGHT_SEMANTIC=0.7

# For broader coverage
SIMILARITY_THRESHOLD=0.6
TOP_K_CHUNKS=7
```

### Response Generation
```env
# For more creative responses
LLM_TEMPERATURE=0.5

# For strictly factual responses
LLM_TEMPERATURE=0.1

# For longer responses
LLM_MAX_TOKENS=3000
```

### Conversation Memory
```env
# Keep more history
MAX_HISTORY_TURNS=10

# Reduce memory usage
MAX_HISTORY_TURNS=3
```

## Limitations and Future Enhancements

### Current Limitations
- **In-memory sessions**: Not persistent across restarts
- **No multi-user support**: Single-threaded CLI only
- **English-optimized**: Better for English queries than Bengali
- **Basic ranking**: Simple hybrid scoring

### Planned Enhancements
- **Web API**: FastAPI-based REST API
- **Persistent sessions**: Redis or database-backed sessions
- **Advanced ranking**: Learning-to-rank algorithms
- **Fine-tuned embeddings**: Legal domain-specific models
- **Bengali NLP**: Improved Bengali query processing
- **Caching**: Query result caching
- **Analytics**: Query logging and performance metrics

## Integration with Existing Tools

### How RAG Uses Existing Code

1. **Database**: Uses `database.db` created by `batch_processor.py`
2. **Keyword Search**: Leverages SQL queries from `search_database.py`
3. **Crime Keywords**: Uses `CRIME_KEYWORDS` dictionary from `crime_keywords.py`
4. **Indexing**: Builds on `create_index.py` with enhanced metadata

### Complementary Tools

- Use `search_database.py` for direct SQL queries
- Use `search_crimes.py` for crime-specific searches
- Use `quality_analyzer.py` for data quality checks
- Use `data_extraction.py` for metadata inspection

## API Reference

### RAGPipeline Class

#### `process_query(query, session_id=None, filters=None)`
Process a user query and generate response.

**Parameters**:
- `query` (str): User's question
- `session_id` (str, optional): Conversation session ID
- `filters` (dict, optional): Search filters (case_type, year, etc.)

**Returns**: Dict with keys:
- `response` (str): Generated answer with citations
- `sources` (list): Retrieved context chunks
- `query_type` (str): Detected query type
- `session_id` (str): Session identifier

#### `search_crime_cases(query, limit=20)`
Search for cases by crime category.

#### `compare_cases(case_ids)`
Compare multiple cases by ID.

#### `summarize_case(case_id)`
Generate structured summary of a judgment.

### HybridRetriever Class

#### `hybrid_retrieve(query, top_k=5, filters=None)`
Perform hybrid semantic + keyword search.

#### `retrieve_semantic(query, top_k=5)`
Semantic search using FAISS.

#### `retrieve_keyword(query, filters=None)`
SQL keyword search.

#### `retrieve_by_crime_category(query)`
Crime-specific retrieval.

### ConversationManager Class

#### `create_session(metadata=None)`
Create new conversation session.

#### `add_message(session_id, role, content)`
Add message to session history.

#### `get_context_for_prompt(session_id, max_turns=None)`
Get conversation context for LLM.

## Testing

Run unit tests:
```bash
pytest tests/test_rag_pipeline.py -v
```

Test individual components:
```bash
# Test retriever
python rag_retriever.py

# Test LLM client
python llm_client.py

# Test prompt building
python prompt_templates.py

# Test conversation manager
python conversation_manager.py
```

## Contributing

When adding features:
1. Follow existing code structure
2. Add logging for debugging
3. Update configuration in `config.py`
4. Add tests in `tests/`
5. Update this README

## License

[Your License Here]

## Support

For issues or questions:
1. Check this README's troubleshooting section
2. Verify configuration with `python config.py`
3. Check logs in `logs/rag_system.log`
4. Review error messages for specific guidance

---

**Version**: 1.0
**Last Updated**: 2025-10-11
**Status**: Production Ready ✅
