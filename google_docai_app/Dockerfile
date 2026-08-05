FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/*

COPY google_docai_app/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY layout_to_markdown.py .
COPY google_docai_app/main.py .

EXPOSE 8004

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8004"]
