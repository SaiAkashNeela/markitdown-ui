# md-converted

An Enterprise Multi-Engine Document to Markdown & RAG Vector Ingestion Platform.

## Conversion Engines Included

1. **Google Document AI AST Engine** (`/google-docai-api/`)
   - Uses `layout_to_markdown.py` AST & Chunking engine.
   - Converts Document AI JSON / PDFs into GFM Markdown and token-aware RAG vector chunks (`embedding_text`, graph relationships, bounding boxes).

2. **Microsoft MarkItDown** (`/markitdown-api/`)
   - Microservice wrapping `MarkItDown()` for converting Office documents (DOCX, XLSX, PPTX) and HTML into clean Markdown.

3. **IBM Docling** (`/docling-api/`)
   - Containerized Docling Serve microservice for advanced document layout parsing.

---

## Directory Structure

```
md-converted/
├── docker-compose.yml           # Unified Compose orchestrating docling, markitdown, google-docai, and ui
├── nginx.conf                  # Reverse Proxy routing API calls
├── Dockerfile                  # Nginx Web UI image
├── index.html                  # Web UI with engine selector & tabbed RAG chunk viewer
├── layout_to_markdown.py       # Production AST & RAG Chunking Library for Google Document AI
├── markitdown_app/             # Microsoft MarkItDown FastAPI microservice
│   ├── main.py
│   ├── requirements.txt
│   └── Dockerfile
└── google_docai_app/           # Google Document AI FastAPI microservice
    ├── main.py
    ├── requirements.txt
    └── Dockerfile
```

---

## Quick Start with Docker

```bash
docker compose up --build -d
```

Open `http://localhost:8080` or `http://localhost:8003` in your browser.
