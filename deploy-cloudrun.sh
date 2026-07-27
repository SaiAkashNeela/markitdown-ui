#!/usr/bin/env bash
set -e

# Load environment variables from .env if present
if [ -f .env ]; then
  export $(grep -v '^#' .env | xargs)
fi

SERVICE_NAME="${SERVICE_NAME:-docling-api}"
REGION="${REGION:-us-central1}"
API_KEY="${DOCLING_SERVE_API_KEY:-pmKnnGoF3QchPzB5Yva5qM5MUPhvyuZX}"
DO_OCR="${DOCLING_SERVE_DO_OCR:-false}"

echo "Deploying ${SERVICE_NAME} to Google Cloud Run in region ${REGION}..."

gcloud run deploy "${SERVICE_NAME}" \
  --source . \
  --region "${REGION}" \
  --allow-unauthenticated \
  --port 8080 \
  --memory 4Gi \
  --cpu 2 \
  --set-env-vars "DOCLING_SERVE_API_KEY=${API_KEY},DOCLING_SERVE_DO_OCR=${DO_OCR},DOCLING_SERVE_MAX_NUM_PAGES=10000,DOCLING_SERVE_MAX_DOCUMENT_TIMEOUT=600,DOCLING_SERVE_ENABLE_UI=false,UVICORN_PORT=8080"

echo "Deployment finished successfully!"
