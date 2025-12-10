# Dozu Embedding Service

A high-performance FastAPI service for generating sentence embeddings using Sentence-Transformers. Provides text embedding, YouTube transcript processing, PDF document embedding, and semantic similarity search.

## Features

- **Text Embedding**: Single text or batch embedding generation
- **YouTube Transcript Processing**: Automatic transcript fetching, sentence segmentation, and embedding
- **PDF Processing**: Extract and embed text from PDF documents by page
- **Semantic Search**: Cosine similarity search across embedded content
- **Multi-device Support**: Automatic detection of CUDA, Apple MPS, or CPU
- **Docker Ready**: Optimized Docker image (~2-3GB) with security best practices

## Quick Start

### Using Docker (Recommended)

```bash
# Pull and run the latest image
docker pull dozuu/dozu-embedding-service:latest

docker run --name dozu-embedding-service \
  -p 8686:8686 \
  -e MODEL_NAME=paraphrase-multilingual-MiniLM-L12-v2 \
  -e DEVICE=auto \
  -d dozuu/dozu-embedding-service:latest
```

Check health:

```bash
curl http://localhost:8686/health
```

### Build from Source

```bash
# Build the image
docker build -t dozu-embedding-service .

# Run the container
docker run --rm -p 8686:8686 \
  -e MODEL_NAME=paraphrase-multilingual-MiniLM-L12-v2 \
  -e DEVICE=auto \
  dozu-embedding-service
```

### Local Development

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# (Optional) Install spaCy model for sentence refinement
python -m spacy download en_core_web_sm

# Run the service
uvicorn app:app --host 0.0.0.0 --port 8686 --reload
```

## Configuration

### Environment Variables

| Variable                | Default                                 | Description                                              |
| ----------------------- | --------------------------------------- | -------------------------------------------------------- |
| `MODEL_NAME`            | `paraphrase-multilingual-MiniLM-L12-v2` | SentenceTransformer model name                           |
| `HUGGINGFACE_HUB_TOKEN` | -                                       | HuggingFace API token (required for gated models)        |
| `DEVICE`                | `auto`                                  | Device selection: `auto`, `cpu`, `cuda`, `cuda:0`, `mps` |
| `HF_HOME`               | `/app/.cache/huggingface`               | HuggingFace cache directory                              |

### Recommended Models

- `paraphrase-multilingual-MiniLM-L12-v2` - Multilingual, balanced performance (default)
- `sentence-transformers/all-MiniLM-L6-v2` - Fast, English-only
- `sentence-transformers/all-mpnet-base-v2` - High quality, English-only

## API Endpoints

### Health Check

```bash
GET /health
```

Returns service status, loaded model, and device information.

### Single Text Embedding

```bash
POST /single/text/embedding
Content-Type: application/json

{
  "query": "Your text here"
}
```

### Batch Text Embedding

```bash
POST /segments/embedding
Content-Type: application/json

{
  "segments": [
    {"text": "First sentence"},
    {"text": "Second sentence"}
  ]
}
```

### YouTube Transcript Processing

```bash
POST /youtube/segments
Content-Type: application/json

{
  "video_id": "dQw4w9WgXcQ",
  "languages": ["en"],
  "refine": false,
  "max_gap": 1.5,
  "min_length": 5,
  "query": "search query"
}
```

### YouTube Segment Similarity

```bash
POST /youtube/segments/similarity
Content-Type: application/json

{
  "segments": [...],
  "query": "search query",
  "top_k": 5
}
```

### PDF Page Embedding

```bash
POST /pdf/page/embedding
Content-Type: application/json

{
  "fileUrl": "https://example.com/document.pdf"
}
```

## Architecture

```
dozu-embedding-service/
├── app.py                    # FastAPI application and endpoints
├── youtube_pipeline.py       # YouTube transcript processing pipeline
├── file_pdf_pipeline.py      # PDF processing pipeline
├── requirements.txt          # Python dependencies
├── Dockerfile               # Optimized Docker build
└── .dockerignore            # Docker build exclusions
```

### Key Components

- **app.py**: Main FastAPI application with endpoint definitions and model management
- **youtube_pipeline.py**: Transcript fetching, sentence segmentation, embedding, and search
- **file_pdf_pipeline.py**: PDF text extraction and page-by-page embedding

## Development

### Running Tests

```bash
# Run specific test files (excluded from Docker build)
python test_youtube_pipeline.py
```

### Generating Sample Data

```bash
python generate_sample_test.py
```

## Docker Optimization

The Docker image is optimized for size and security:

- **Multi-stage build**: Separates build dependencies from runtime
- **Minimal base**: Uses `python:3.13-slim`
- **Non-root user**: Runs as `appuser` (UID 1000)
- **Cache optimization**: Proper `.dockerignore` prevents large files from being copied
- **Layer caching**: Efficient layer ordering for faster rebuilds

**Image size**: ~2-3GB (includes Python, dependencies, and runtime model cache)

## Network Deployment

### Docker Network Setup

```bash
# Create network
docker network create dozu-app-network

# Run with network
docker run --name dozu-embedding-service \
  --network dozu-app-network \
  -p 8686:8686 \
  -d dozuu/dozu-embedding-service:latest
```

### Update Deployment

```bash
docker pull dozuu/dozu-embedding-service:latest && \
docker stop dozu-embedding-service && \
docker rm dozu-embedding-service && \
docker run --name dozu-embedding-service  --network dozu-app-network  -p 8686:8686 -d dozuu/dozu-embedding-service:latest
```

## Requirements

- **Python**: 3.10+
- **Memory**: 2GB minimum, 4GB+ recommended
- **GPU**: Optional (CUDA-compatible for faster inference)

### Python Dependencies

- `fastapi` & `uvicorn` - Web framework
- `sentence-transformers` - Embedding models
- `transformers` & `torch` - Model backend
- `youtube-transcript-api` - YouTube transcript fetching
- `pypdf` - PDF text extraction

## License

[Your License Here]

## Support

For issues and questions, please open an issue on the repository.
