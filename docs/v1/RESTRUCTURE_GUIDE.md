# Dozu Embedding Service - Restructured

This project has been restructured following the Scalable FastAPI Architecture pattern with a hybrid src layout.

## New Project Structure

```
.
├── src/
│   └── app/
│       ├── __init__.py
│       ├── main.py                   # App entrypoint (FastAPI app)
│       ├── api/                      # API layer (routers)
│       │   ├── __init__.py
│       │   ├── deps.py               # Dependencies (embed functions)
│       │   ├── api.py                # Central router
│       │   └── endpoints/
│       │       ├── __init__.py
│       │       ├── health.py         # Health check endpoint
│       │       ├── youtube.py        # YouTube transcript endpoints
│       │       ├── embedding.py      # Embedding endpoints
│       │       └── pdf.py            # PDF processing endpoints
│       ├── core/                     # Config & model management
│       │   ├── __init__.py
│       │   ├── config.py             # Settings from env
│       │   └── model_manager.py      # Model loading & embedding
│       ├── services/                 # Business logic
│       │   ├── __init__.py
│       │   ├── youtube_service.py    # YouTube processing logic
│       │   ├── embedding_service.py  # Embedding logic
│       │   └── pdf_service.py        # PDF processing logic
│       └── schemas/                  # Pydantic models (req/resp)
│           ├── __init__.py
│           ├── common_schema.py      # Shared schemas
│           ├── youtube_schema.py     # YouTube schemas
│           ├── embedding_schema.py   # Embedding schemas
│           └── pdf_schema.py         # PDF schemas
├── youtube_pipeline.py               # YouTube pipeline utilities
├── file_pdf_pipeline.py              # PDF pipeline utilities
├── app.py                            # (OLD - kept for reference)
├── run.py                            # Entry point script
├── requirements.txt
├── Dockerfile
├── Architecture.md
└── README.md

```

## Key Changes

### Separation of Concerns

1. **Core Layer** (`src/app/core/`):

   - `config.py`: Environment-based configuration
   - `model_manager.py`: Centralized model loading and embedding functions

2. **Schemas Layer** (`src/app/schemas/`):

   - Separated by feature: YouTube, Embedding, PDF
   - Common schemas for shared types (e.g., ProxyConfig)

3. **Services Layer** (`src/app/services/`):

   - Business logic separated from endpoints
   - `youtube_service.py`: YouTube transcript processing
   - `embedding_service.py`: Text embedding operations
   - `pdf_service.py`: PDF processing operations

4. **API Layer** (`src/app/api/endpoints/`):

   - Each feature has its own endpoint file
   - `health.py`: Health check endpoint
   - `youtube.py`: YouTube-related endpoints
   - `embedding.py`: Embedding-related endpoints
   - `pdf.py`: PDF-related endpoints

5. **Dependencies** (`src/app/api/deps.py`):
   - Centralized dependency injection
   - Provides embed functions to endpoints

## Running the Application

### Option 1: Using run.py (Recommended)

```bash
python run.py
```

### Option 2: Using uvicorn directly

```bash
uvicorn src.app.main:app --reload --host 0.0.0.0 --port 8000
```

### Option 3: Using Docker

```bash
docker build -t dozu-embedding-service .
docker run -p 8000:8000 dozu-embedding-service
```

## API Endpoints

### Health

- `GET /health` - Check service health and model status

### YouTube Endpoints

- `POST /youtube/segments` - Process YouTube transcript
- `POST /youtube/segments/embedding` - Generate embeddings for segments
- `POST /youtube/segments/similarity` - Calculate similarity with query

### Embedding Endpoints

- `POST /single/text/embedding` - Embed single text
- `POST /segments/embedding` - Embed batch of texts
- `POST /embedding/compare` - Compare two texts

### PDF Endpoints

- `POST /pdf/page/embedding` - Process PDF and generate page embeddings

## Benefits of New Structure

1. **Scalability**: Easy to add new features by creating new endpoint files
2. **Maintainability**: Clear separation of concerns makes code easier to understand
3. **Testability**: Services can be tested independently of endpoints
4. **Reusability**: Business logic in services can be reused across endpoints
5. **Following Best Practices**: Aligns with SOLID principles and FastAPI recommendations

## Migration Notes

The original `app.py` has been kept for reference but is no longer used. All functionality has been migrated to the new structure while maintaining the same API interface.
