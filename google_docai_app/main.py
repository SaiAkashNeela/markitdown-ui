import os
import sys
import json
import base64
import tempfile
import urllib.request
import urllib.error
from typing import Optional, Dict, Any
from fastapi import FastAPI, File, UploadFile, HTTPException, Form, Body
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# Import AST & Chunking engine
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
try:
    from layout_to_markdown import parse_docai, render_markdown, build_chunks
except ImportError:
    from ..layout_to_markdown import parse_docai, render_markdown, build_chunks

app = FastAPI(
    title="Google Document AI Layout & RAG Engine",
    version="1.3.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

GCP_CREDS_PATH = os.environ.get("GOOGLE_APPLICATION_CREDENTIALS", "/Users/saiakashneela/github/arivulabs/server/gcp-creds.json")
GCP_PROJECT_ID = os.environ.get("GCP_PROJECT_ID", "94558928138")
GCP_LOCATION = os.environ.get("GCP_LOCATION", "eu")
GCP_PROCESSOR_ID = os.environ.get("GCP_PROCESSOR_ID", "7eb6658b4115ca8c")

@app.get("/health")
def health_check():
    return {"status": "ok", "service": "google_docai"}

@app.post("/v1/convert/json")
@app.post("/convert/json")
async def convert_json_payload(payload: Dict[str, Any] = Body(...)):
    """Converts direct Document AI JSON payload to Markdown & Enterprise RAG Chunks."""
    try:
        ast_nodes = parse_docai(payload, ignore_headers_footers=True)
        markdown_text = render_markdown(ast_nodes, include_page_comments=True)
        rag_chunks = build_chunks(ast_nodes, source_doc="document.json")

        return {
            "status": "success",
            "engine": "Google Document AI AST Engine",
            "markdown": markdown_text,
            "chunks": rag_chunks
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to process Document AI JSON: {str(e)}")

@app.post("/v1/convert/file")
@app.post("/convert/file")
async def convert_file(
    file: UploadFile = File(...),
    processor_id: Optional[str] = Form(None),
    location: Optional[str] = Form(None)
):
    """Processes PDF file via Google Document AI API and extracts AST Markdown & RAG Chunks."""
    try:
        content = await file.read()
        b64_doc = base64.b64encode(content).decode("utf-8")

        proc_id = processor_id or GCP_PROCESSOR_ID
        loc = location or GCP_LOCATION
        url = f"https://{loc}-documentai.googleapis.com/v1/projects/{GCP_PROJECT_ID}/locations/{loc}/processors/{proc_id}:process"

        # Obtain Access Token via Service Account Key if present
        token = ""
        if os.path.exists(GCP_CREDS_PATH):
            import subprocess
            cmd = f"gcloud auth activate-service-account --key-file={GCP_CREDS_PATH} && gcloud auth print-access-token"
            res = subprocess.run(cmd, shell=True, capture_output=True, text=True)
            token = res.stdout.strip().split("\n")[-1]

        if not token:
            token = os.environ.get("GCP_ACCESS_TOKEN", "")

        if not token:
            raise HTTPException(status_code=401, detail="Missing GCP Authentication Token or Service Account Credential File.")

        headers = {
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/json; charset=utf-8"
        }
        api_payload = {
            "rawDocument": {
                "content": b64_doc,
                "mimeType": file.content_type or "application/pdf"
            }
        }

        req = urllib.request.Request(url, data=json.dumps(api_payload).encode("utf-8"), headers=headers, method="POST")
        with urllib.request.urlopen(req) as resp:
            docai_response = json.loads(resp.read().decode("utf-8"))

        ast_nodes = parse_docai(docai_response, ignore_headers_footers=True)
        markdown_text = render_markdown(ast_nodes, include_page_comments=True)
        rag_chunks = build_chunks(ast_nodes, source_doc=file.filename or "uploaded.pdf")

        return {
            "status": "success",
            "filename": file.filename,
            "engine": "Google Document AI AST Engine",
            "markdown": markdown_text,
            "chunks": rag_chunks
        }
    except urllib.error.HTTPError as e:
        err_body = e.read().decode("utf-8") if e.fp else str(e)
        raise HTTPException(status_code=e.code, detail=f"GCP Document AI Error: {err_body}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
