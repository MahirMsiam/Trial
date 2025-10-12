# Bangladesh Supreme Court RAG API

REST API for searching and querying 8000+ legal judgments from the Bangladesh Supreme Court using hybrid search (FAISS + SQL) and Retrieval-Augmented Generation (RAG).

## Architecture Overview

```
┌─────────────┐
│   Client    │
│  (Web/App)  │
└──────┬──────┘
       │ HTTP/REST
       ▼
┌─────────────┐
│  FastAPI    │
│  Backend    │
└──────┬──────┘
       │
       ▼
┌─────────────┐
│ RAG Pipeline│
└──────┬──────┘
       │
   ┌───┴────────────┬──────────────┐
   ▼                ▼              ▼
┌────────┐    ┌──────────┐   ┌─────────┐
│ FAISS  │    │  SQLite  │   │   LLM   │
│ Index  │    │ Database │   │(OpenAI/ │
│        │    │          │   │Anthropic│
└────────┘    └──────────┘   │/Gemini) │
                              └─────────┘
```

## Prerequisites

- **Python 3.8+** installed on your system
- **Completed RAG system setup** from `README_RAG.md`:
  - SQLite database populated with judgments
  - FAISS index created (optional but recommended)
  - LLM API keys configured in `.env` file (OpenAI, Anthropic, or Google Gemini)
- **All dependencies** from `requirements_api.txt`

## Installation

### Step 1: Install Dependencies

```bash
pip install -r requirements_api.txt
```

This will install FastAPI, Uvicorn, Pydantic, and all RAG system dependencies.

### Step 2: Verify Configuration

```bash
python config.py
```

Ensure no critical errors are reported. The system can run in degraded mode without FAISS index (keyword search only).

### Step 3: Verify Database and Index

- Database should exist at path specified in `.env` (`DATABASE_PATH`)
- FAISS index should exist at paths specified in `.env` (`FAISS_INDEX_PATH`, `CHUNKS_MAP_PATH`)
- If FAISS index is missing, run `python create_index.py` to create it

## Running the API

### Development Mode (with auto-reload)

```bash
uvicorn api.main:app --reload --host 0.0.0.0 --port 8000
```

The `--reload` flag enables automatic restart on code changes.

### Production Mode (multiple workers)

```bash
uvicorn api.main:app --host 0.0.0.0 --port 8000 --workers 4
```

Use multiple workers for better performance under load.

### Custom Port

```bash
uvicorn api.main:app --host 0.0.0.0 --port 5000
```

### Access Documentation

Once the API is running:

- **Swagger UI (Interactive)**: http://localhost:8000/api/docs
- **ReDoc (Documentation)**: http://localhost:8000/api/redoc
- **API Root**: http://localhost:8000/
- **Health Check**: http://localhost:8000/api/health

## API Endpoints

### Search Endpoints

#### 1. Keyword Search

**POST** `/api/search/keyword`

SQL-based keyword search through case metadata.

**Request Body:**
```json
{
  "query": "murder section 302",
  "filters": {
    "case_type": "Criminal Appeal",
    "year": 2020,
    "petitioner": "John Doe"
  }
}
```

**Response:**
```json
{
  "results": [
    {
      "id": 123,
      "case_number": "1234",
      "case_year": "2020",
      "case_type": "Criminal Appeal",
      "full_case_id": "Criminal Appeal No. 1234 of 2020",
      "petitioner_name": "John Doe",
      "respondent_name": "State",
      "judgment_date": "2020-06-15",
      "judgment_outcome": "Appeal Allowed",
      "court_name": "Appellate Division"
    }
  ],
  "count": 1,
  "query": "murder section 302"
}
```

**cURL Example:**
```bash
curl -X POST "http://localhost:8000/api/search/keyword" \
  -H "Content-Type: application/json" \
  -d '{"query": "murder", "filters": {"case_type": "Criminal Appeal"}}'
```

---

#### 2. Semantic Search

**POST** `/api/search/semantic`

FAISS-based semantic search using sentence embeddings.

**Request Body:**
```json
{
  "query": "What are the legal precedents for wrongful termination?",
  "top_k": 5
}
```

**Response:**
```json
[
  {
    "chunk_text": "The court held that termination without proper notice...",
    "similarity": 0.87,
    "case_id": 123,
    "case_number": "1234",
    "case_type": "Civil Appeal",
    "full_case_id": "Civil Appeal No. 1234 of 2020",
    "source": "case"
  }
]
```

**cURL Example:**
```bash
curl -X POST "http://localhost:8000/api/search/semantic" \
  -H "Content-Type: application/json" \
  -d '{"query": "wrongful termination precedents", "top_k": 5}'
```

---

#### 3. Hybrid Search

**POST** `/api/search/hybrid`

Combined semantic + keyword search for comprehensive results.

**Request Body:**
```json
{
  "query": "breach of contract damages",
  "top_k": 10,
  "filters": {
    "case_type": "Civil Appeal"
  }
}
```

**Response:** Returns case-level results with nested chunks from both retrieval methods.

**cURL Example:**
```bash
curl -X POST "http://localhost:8000/api/search/hybrid" \
  -H "Content-Type: application/json" \
  -d '{"query": "breach of contract", "top_k": 10}'
```

---

#### 4. Crime Category Search

**POST** `/api/search/crime`

Search for cases by crime category (murder, rape, theft, etc.).

**Request Body:**
```json
{
  "query": "murder cases",
  "limit": 20
}
```

**Response:**
```json
{
  "response": "Found 20 murder cases...",
  "crime_type": "murder",
  "count": 20,
  "cases": [
    {
      "chunk_text": "The accused was convicted under Section 302 IPC for murder...",
      "similarity": 0.81,
      "case_id": 123,
      "case_number": "1234",
      "case_type": "Criminal Appeal",
      "judgment_date": "2020-06-15",
      "petitioner": "State",
      "respondent": "Rahman",
      "full_case_id": "Criminal Appeal No. 1234 of 2020",
      "source": "keyword"
    }
  ],
  "summary": "These cases involve criminal charges under Section 302..."
}
```

**Note:** The `cases` field returns chunk-level previews (ChunkResponse format) with relevant text excerpts and case metadata. Each chunk represents a relevant portion of a judgment that matches the crime category.

**cURL Example:**
```bash
curl -X POST "http://localhost:8000/api/search/crime" \
  -H "Content-Type: application/json" \
  -d '{"query": "murder", "limit": 20}'
```

---

### Chat Endpoints

#### 5. RAG Chat Query

**POST** `/api/chat`

Ask questions about legal judgments using RAG (retrieval + LLM generation).

**Request Body:**
```json
{
  "query": "What are the requirements for proving self-defense in murder cases?",
  "session_id": "abc123",
  "filters": {
    "case_type": "Criminal Appeal"
  }
}
```

**Response:**
```json
{
  "response": "Based on the retrieved judgments, the requirements for proving self-defense include: 1) Imminent threat...",
  "sources": [
    {
      "chunk_text": "The court established that self-defense requires...",
      "case_id": 123,
      "full_case_id": "Criminal Appeal No. 1234 of 2020"
    }
  ],
  "query_type": "legal_question",
  "session_id": "abc123"
}
```

**cURL Example:**
```bash
curl -X POST "http://localhost:8000/api/chat" \
  -H "Content-Type: application/json" \
  -d '{"query": "What are self-defense requirements?", "session_id": "my-session"}'
```

---

#### 6. Streaming Chat Query

**POST** `/api/chat/stream`

Streaming RAG responses using Server-Sent Events (SSE).

**Request Body:**
```json
{
  "query": "Explain the concept of res judicata",
  "session_id": "abc123"
}
```

**Response:** Server-Sent Events stream with incremental tokens.

**JavaScript Fetch Example (POST with SSE):**
```javascript
const response = await fetch('http://localhost:8000/api/chat/stream', {
  method: 'POST',
  headers: { 'Content-Type': 'application/json' },
  body: JSON.stringify({
    query: 'Explain the concept of res judicata',
    session_id: 'abc123'
  })
});

const reader = response.body.getReader();
const decoder = new TextDecoder();

while (true) {
  const { done, value } = await reader.read();
  if (done) break;
  
  const chunk = decoder.decode(value);
  const lines = chunk.split('\n');
  
  for (const line of lines) {
    if (line.startsWith('data: ')) {
      const data = JSON.parse(line.slice(6));
      if (data.type === 'token') {
        console.log(data.token);
      } else if (data.type === 'complete') {
        console.log('Final response:', data.response);
      }
    }
  }
}
```

**Note:** This endpoint uses POST (not GET), so standard `EventSource` won't work. Use the fetch API with streaming as shown above.

**cURL Example:**
```bash
curl -X POST "http://localhost:8000/api/chat/stream" \
  -H "Content-Type: application/json" \
  -d '{"query": "Explain res judicata", "stream": true}' \
  -N
```

---

#### 7. Summarize Case

**POST** `/api/case/{case_id}/summary`

Generate AI summary of a specific case.

**Path Parameter:** `case_id` (integer)

**Response:**
```json
{
  "summary": "This criminal appeal involves charges under Section 302...",
  "case_data": {
    "id": 123,
    "case_number": "1234",
    "full_case_id": "Criminal Appeal No. 1234 of 2020"
  }
}
```

**cURL Example:**
```bash
curl -X POST "http://localhost:8000/api/case/123/summary" \
  -H "Content-Type: application/json"
```

---

#### 8. Compare Cases

**POST** `/api/cases/compare`

Compare multiple cases using LLM analysis.

**Request Body:**
```json
{
  "case_ids": [123, 456, 789]
}
```

**Response:**
```json
{
  "comparison": "Comparing these three cases reveals similar patterns in applying Section 302...",
  "cases": [...]
}
```

**cURL Example:**
```bash
curl -X POST "http://localhost:8000/api/cases/compare" \
  -H "Content-Type: application/json" \
  -d '{"case_ids": [123, 456, 789]}'
```

---

### Session Management Endpoints

#### 9. Create Session

**POST** `/api/session`

Create a new conversation session for tracking chat history.

**Request Body:**
```json
{
  "metadata": {
    "user_id": "user123",
    "purpose": "legal research"
  }
}
```

**Response:**
```json
{
  "session_id": "abc123",
  "created_at": "2025-10-12T10:00:00Z",
  "last_active": "2025-10-12T10:00:00Z",
  "message_count": 0
}
```

**cURL Example:**
```bash
curl -X POST "http://localhost:8000/api/session" \
  -H "Content-Type: application/json" \
  -d '{}'
```

---

#### 10. Get Session Info

**GET** `/api/session/{session_id}`

Retrieve information about a specific session.

**Response:**
```json
{
  "session_id": "abc123",
  "created_at": "2025-10-12T10:00:00Z",
  "last_active": "2025-10-12T10:15:00Z",
  "message_count": 5
}
```

**cURL Example:**
```bash
curl -X GET "http://localhost:8000/api/session/abc123"
```

---

#### 11. Get Session History

**GET** `/api/session/{session_id}/history?max_turns=10`

Retrieve conversation history for a session.

**Query Parameters:**
- `max_turns` (optional): Maximum number of conversation turns to return

**Response:**
```json
[
  {
    "role": "user",
    "content": "What is Section 302?",
    "timestamp": "2025-10-12T10:00:00Z"
  },
  {
    "role": "assistant",
    "content": "Section 302 of the Penal Code pertains to murder...",
    "timestamp": "2025-10-12T10:00:05Z"
  }
]
```

**cURL Example:**
```bash
curl -X GET "http://localhost:8000/api/session/abc123/history?max_turns=10"
```

---

#### 12. Delete Session

**DELETE** `/api/session/{session_id}`

Clear/delete a conversation session.

**Response:** 204 No Content

**cURL Example:**
```bash
curl -X DELETE "http://localhost:8000/api/session/abc123"
```

---

### Utility Endpoints

#### 13. Health Check

**GET** `/api/health`

Check system health and component availability.

**Response:**
```json
{
  "status": "healthy",
  "database_connected": true,
  "faiss_index_loaded": true,
  "llm_provider": "openai"
}
```

**Status Values:**
- `healthy`: All systems operational
- `degraded`: Database available but FAISS index missing (keyword search only)
- `unhealthy`: Critical components unavailable

**cURL Example:**
```bash
curl -X GET "http://localhost:8000/api/health"
```

---

#### 14. Database Statistics

**GET** `/api/stats`

Get database statistics and counts.

**Response:**
```json
{
  "total_judgments": 8234,
  "case_types": 15,
  "total_advocates": 1523,
  "total_laws_cited": 456,
  "case_type_breakdown": [
    {"case_type": "Criminal Appeal", "count": 2341},
    {"case_type": "Civil Appeal", "count": 1876}
  ]
}
```

**cURL Example:**
```bash
curl -X GET "http://localhost:8000/api/stats"
```

---

#### 15. API Version

**GET** `/api/version`

Get API version information.

**Response:**
```json
{
  "version": "1.0.0",
  "api_name": "Bangladesh Supreme Court RAG API",
  "description": "REST API for legal judgment search and RAG-based question answering"
}
```

**cURL Example:**
```bash
curl -X GET "http://localhost:8000/api/version"
```

---

## Example Usage

### Python Client Example

```python
import requests

# Base URL
BASE_URL = "http://localhost:8000"

# Create a session
response = requests.post(f"{BASE_URL}/api/session")
session_id = response.json()["session_id"]

# Ask a question
response = requests.post(
    f"{BASE_URL}/api/chat",
    json={
        "query": "What are the legal requirements for proving murder?",
        "session_id": session_id
    }
)
result = response.json()
print(f"Answer: {result['response']}")
print(f"Sources: {len(result['sources'])} cases referenced")

# Search for specific cases
response = requests.post(
    f"{BASE_URL}/api/search/hybrid",
    json={
        "query": "self-defense murder cases",
        "top_k": 5
    }
)
cases = response.json()
print(f"Found {len(cases)} relevant cases")
```

### JavaScript/TypeScript Fetch Example

```javascript
// Create a session
const sessionResponse = await fetch('http://localhost:8000/api/session', {
  method: 'POST',
  headers: { 'Content-Type': 'application/json' },
  body: JSON.stringify({})
});
const { session_id } = await sessionResponse.json();

// Ask a question
const chatResponse = await fetch('http://localhost:8000/api/chat', {
  method: 'POST',
  headers: { 'Content-Type': 'application/json' },
  body: JSON.stringify({
    query: 'What is the burden of proof in criminal cases?',
    session_id: session_id
  })
});
const result = await chatResponse.json();
console.log('Answer:', result.response);
console.log('Sources:', result.sources.length);
```

## Authentication

**Current Status:** No authentication implemented (development mode).

**Production Recommendation:** Add API key authentication or JWT tokens before deploying to production.

**Future Enhancement:** Uncomment and configure API key validation in `.env`:
```bash
API_KEY_ENABLED=true
API_KEY_HEADER=X-API-Key
```

## Error Handling

All error responses follow a consistent format:

```json
{
  "error": "Error message",
  "detail": "Detailed error information",
  "status_code": 500,
  "path": "/api/chat"
}
```

### HTTP Status Codes

- **200 OK**: Successful request
- **201 Created**: Resource created (sessions)
- **204 No Content**: Successful deletion
- **400 Bad Request**: Invalid request parameters
- **404 Not Found**: Resource not found (session, case)
- **422 Validation Error**: Request body validation failed
- **500 Internal Server Error**: Server-side error
- **503 Service Unavailable**: System component unavailable (database, FAISS, LLM)

## Rate Limiting

**Current Status:** No rate limiting implemented.

**Production Recommendation:** Implement rate limiting to prevent abuse:

```bash
# Example with slowapi
pip install slowapi

# Configure in .env
RATE_LIMIT_ENABLED=true
RATE_LIMIT_REQUESTS=100
RATE_LIMIT_WINDOW=60  # seconds
```

## CORS Configuration

### Current Configuration

```python
# In api/main.py
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # PERMISSIVE - allows all origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
```

### Production Configuration

**Recommendation:** Restrict CORS to specific frontend domains:

```python
# In api/main.py - modify for production
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "https://yourdomain.com",
        "https://app.yourdomain.com"
    ],
    allow_credentials=True,
    allow_methods=["GET", "POST", "DELETE"],
    allow_headers=["Content-Type", "Authorization"],
)
```

**Or use environment variable:**

```bash
# In .env
CORS_ORIGINS=https://yourdomain.com,https://app.yourdomain.com
```

## Performance Considerations

### Singleton RAG Pipeline

- **RAG pipeline is loaded once** at startup and reused for all requests
- **Models cached in memory**: Sentence transformer, FAISS index, LLM client
- **Benefit**: Eliminates ~5-10 second model loading time per request

### Database Connections

- **Per-request connections**: New database connection for each search request
- **Automatic cleanup**: Connections closed after request completes
- **Thread-safe**: Multiple requests can execute concurrently

### Streaming Responses

- **Use streaming** for long-running LLM queries to reduce perceived latency
- **Server-Sent Events**: Tokens sent as generated, not all at once
- **Benefit**: Better user experience for lengthy answers

### Production Recommendations

1. **Multiple Workers**: Run with 4+ workers for concurrent requests
   ```bash
   uvicorn api.main:app --workers 4
   ```

2. **Redis Session Storage**: Use Redis for session persistence instead of in-memory
   ```python
   # Future enhancement
   REDIS_URL=redis://localhost:6379/0
   ```

3. **Response Caching**: Cache frequently requested case summaries and searches

4. **Load Balancing**: Use nginx or similar for horizontal scaling

## Deployment

### Development Deployment

```bash
# With auto-reload for development
uvicorn api.main:app --reload --host 0.0.0.0 --port 8000
```

### Production Deployment

```bash
# Multiple workers for production
uvicorn api.main:app --host 0.0.0.0 --port 8000 --workers 4 --log-level info
```

### Nginx Reverse Proxy Configuration

```nginx
server {
    listen 80;
    server_name api.yourdomain.com;

    location / {
        proxy_pass http://127.0.0.1:8000;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
        
        # For streaming responses
        proxy_buffering off;
        proxy_cache off;
    }
}
```

### Environment Variables

Configure in `.env` file (copy from `.env.example`):

```bash
# Database
DATABASE_PATH=path/to/database.db

# FAISS Index
FAISS_INDEX_PATH=path/to/faiss_index.bin
CHUNKS_MAP_PATH=path/to/chunks_map.pkl

# LLM Provider
LLM_PROVIDER=openai  # or anthropic, gemini, local
OPENAI_API_KEY=your-key-here
ANTHROPIC_API_KEY=your-key-here
GEMINI_API_KEY=your-gemini-key-here
GEMINI_MODEL=gemini-1.5-pro  # or gemini-pro

# API Configuration
API_HOST=0.0.0.0
API_PORT=8000
API_WORKERS=4
CORS_ORIGINS=*
```

### Logging

- **Log file location**: `logs/rag_system.log`
- **Log level**: INFO (configurable in `logging_config.py`)
- **Includes**: API requests, errors, RAG operations, database queries

**Monitor logs:**
```bash
tail -f logs/rag_system.log
```

## Troubleshooting

### Common Issues

#### 1. "RAG system not initialized"

**Cause:** Database or FAISS index not found, or models failed to load.

**Solution:**
```bash
# Check database exists
ls -l path/to/database.db

# Check FAISS index exists
ls -l path/to/faiss_index.bin

# Verify configuration
python config.py
```

---

#### 2. "Semantic search unavailable"

**Cause:** FAISS index not created.

**Solution:**
```bash
# Create FAISS index
python create_index.py
```

---

#### 3. "Database unavailable"

**Cause:** Database path incorrect or file permissions issue.

**Solution:**
```bash
# Check DATABASE_PATH in .env
cat .env | grep DATABASE_PATH

# Check file permissions
ls -la path/to/database.db

# Test database connection
python -c "from search_database import JudgmentSearch; s = JudgmentSearch(); print(s.get_stats())"
```

---

#### 4. "LLM API errors"

**Cause:** Missing or invalid API keys.

**Solution:**
```bash
# Check API keys in .env
cat .env | grep API_KEY

# Verify API key is valid
python -c "from llm_client import LLMClient; c = LLMClient(); print(c.test_connection())"
```

For Gemini-specific errors, see detailed troubleshooting in `README_RAG.md`:
- Content blocked by safety filters
- google-generativeai not installed
- Invalid Gemini API key

---

#### 5. CORS Errors in Browser

**Cause:** Frontend origin not allowed in CORS configuration.

**Solution:**
```python
# In api/main.py, update CORS origins
allow_origins=["http://localhost:3000", "https://yourdomain.com"]
```

---

## Integration with Frontend

### Base Configuration

```javascript
const API_BASE_URL = 'http://localhost:8000';
const API_HEADERS = {
  'Content-Type': 'application/json'
};
```

### Session Management

```javascript
// Store session_id in localStorage or state management
const sessionId = localStorage.getItem('sessionId');
if (!sessionId) {
  const response = await fetch(`${API_BASE_URL}/api/session`, {
    method: 'POST',
    headers: API_HEADERS
  });
  const data = await response.json();
  localStorage.setItem('sessionId', data.session_id);
}
```

### Streaming with Fetch API (POST SSE)

```javascript
// POST endpoint requires fetch API with streaming, not EventSource
const response = await fetch(`${API_BASE_URL}/api/chat/stream`, {
  method: 'POST',
  headers: API_HEADERS,
  body: JSON.stringify({ query, session_id })
});

const reader = response.body.getReader();
const decoder = new TextDecoder();

while (true) {
  const { done, value } = await reader.read();
  if (done) break;
  
  const chunk = decoder.decode(value);
  const lines = chunk.split('\n');
  
  for (const line of lines) {
    if (line.startsWith('data: ')) {
      const data = JSON.parse(line.slice(6));
      if (data.type === 'token') {
        appendToUI(data.token);
      } else if (data.type === 'complete') {
        displayFinalResponse(data.response);
        break;
      }
    }
  }
}
```

## Testing

### Manual Testing with Swagger UI

1. Start the API: `uvicorn api.main:app --reload`
2. Open browser: http://localhost:8000/api/docs
3. Test endpoints interactively with sample data

### Automated Testing (Future)

```python
# tests/test_api.py
import pytest
from fastapi.testclient import TestClient
from api.main import app

client = TestClient(app)

def test_health_check():
    response = client.get("/api/health")
    assert response.status_code == 200
    assert response.json()["status"] in ["healthy", "degraded"]

def test_create_session():
    response = client.post("/api/session", json={})
    assert response.status_code == 201
    assert "session_id" in response.json()
```

**Run tests:**
```bash
pytest tests/test_api.py -v
```

### Load Testing (Future)

```bash
# Install locust
pip install locust

# Create locustfile.py
# Run load test
locust -f locustfile.py --host=http://localhost:8000
```

## API Versioning

**Current Version:** 1.0.0

**Future Versioning Strategy:**
- Use URL prefix for major versions: `/api/v1/chat`, `/api/v2/chat`
- Maintain backward compatibility for at least one version
- Deprecation warnings in response headers

## Monitoring

### Health Monitoring

```bash
# Automated health check script
while true; do
  curl http://localhost:8000/api/health
  sleep 60
done
```

### Log Monitoring

```bash
# Monitor logs in real-time
tail -f logs/rag_system.log | grep ERROR
```

### Metrics (Future Enhancement)

**Add Prometheus metrics:**
```bash
pip install prometheus-fastapi-instrumentator

# In api/main.py
from prometheus_fastapi_instrumentator import Instrumentator
Instrumentator().instrument(app).expose(app)
```

**Metrics available at:** http://localhost:8000/metrics

## Security

### Production Security Checklist

- [ ] **HTTPS**: Enable TLS/SSL for all API traffic
- [ ] **API Key Authentication**: Implement API key or JWT authentication
- [ ] **CORS**: Restrict `allow_origins` to specific frontend domains
- [ ] **Rate Limiting**: Add rate limiting to prevent abuse
- [ ] **Input Validation**: Pydantic models validate all inputs
- [ ] **SQL Injection**: Parameterized queries used (protected)
- [ ] **Secrets Management**: Use environment variables, never commit `.env`
- [ ] **Logging**: Sanitize logs to avoid logging sensitive data
- [ ] **Dependencies**: Regularly update dependencies for security patches

### HTTPS Configuration

```bash
# Generate SSL certificate (development)
openssl req -x509 -newkey rsa:4096 -keyout key.pem -out cert.pem -days 365

# Run with HTTPS
uvicorn api.main:app --ssl-keyfile=key.pem --ssl-certfile=cert.pem
```

## Contributing

### Adding New Endpoints

1. **Define Pydantic models** in `api/models.py`
2. **Create route handler** in appropriate file under `api/routes/`
3. **Add endpoint function** with proper decorators and type hints
4. **Include router** in `api/main.py` if new router file
5. **Update this README** with endpoint documentation
6. **Test endpoint** in Swagger UI

### Code Structure

```
api/
├── __init__.py
├── main.py              # FastAPI app entry point
├── models.py            # Pydantic request/response models
├── dependencies.py      # Dependency injection functions
└── routes/
    ├── __init__.py
    ├── search.py        # Search endpoints
    ├── chat.py          # Chat/RAG endpoints
    ├── session.py       # Session management
    └── utility.py       # Health check, stats
```

### Testing Guidelines

- Test all endpoints with valid and invalid data
- Check error handling and status codes
- Verify response schemas match Pydantic models
- Test session persistence and cleanup
- Load test with realistic concurrent requests

## License and Support

### License

This project inherits the license from the main repository.

### Support

For issues, questions, or feature requests:
- Check this documentation first
- Review existing issues in the repository
- Check logs in `logs/rag_system.log`
- Open a new issue with detailed information

### Additional Resources

- **FastAPI Documentation**: https://fastapi.tiangolo.com/
- **Pydantic Documentation**: https://docs.pydantic.dev/
- **Uvicorn Documentation**: https://www.uvicorn.org/

---

**API Version:** 1.0.0  
**Last Updated:** October 12, 2025  
**Maintained By:** Bangladesh Supreme Court RAG Project Team
