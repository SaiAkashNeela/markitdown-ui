# Docling API Service

This repository provides the Docling Document Processing API service configured for deployment to Google Cloud Run or local execution via Docker.

## Environment Variables

Copy `.env.example` to `.env` and set your static API secret key:

```bash
cp .env.example .env
```

`.env` configuration:
- `DOCLING_SERVE_API_KEY`: The static API secret key required in the `X-Api-Key` HTTP header.
- `DOCLING_SERVE_MAX_NUM_PAGES`: Maximum pages per document (default: 10000).
- `DOCLING_SERVE_MAX_DOCUMENT_TIMEOUT`: Max document processing timeout in seconds (default: 600).
- `UVICORN_PORT`: Port to run the Uvicorn server on (default: 8080).

## Local Development with Docker Compose

Run the API service locally:

```bash
docker compose up --build
```

The API will be available at `http://localhost:8080`.

## API Authentication

All requests to the API endpoints (except `/health` and `/ready`) require the `X-Api-Key` header:

```bash
curl -X POST http://localhost:8080/v1/convert/source \
  -H "X-Api-Key: docling_secret_api_key_2026" \
  -H "Content-Type: application/json" \
  -d '{
    "sources": [{"kind": "http", "url": "https://arxiv.org/pdf/2206.01062"}]
  }'
```

## Deploying to Google Cloud Run

Deploy directly using the `deploy-cloudrun.sh` script or `gcloud`:

```bash
./deploy-cloudrun.sh
```

Or run `gcloud` manually:

```bash
gcloud run deploy docling-api \
  --source . \
  --region us-central1 \
  --allow-unauthenticated \
  --port 8080 \
  --memory 4Gi \
  --cpu 2 \
  --set-env-vars "DOCLING_SERVE_API_KEY=your_static_api_secret_key_here,DOCLING_SERVE_ENABLE_UI=false,UVICORN_PORT=8080"
```
