import os
import tempfile
from fastapi import FastAPI, File, UploadFile, HTTPException, Form
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional
from markitdown import MarkItDown

app = FastAPI(
    title="Microsoft MarkItDown Conversion Service",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

markitdown_converter = MarkItDown()

class UrlConvertRequest(BaseModel):
    url: str

@app.get("/health")
def health_check():
    return {"status": "ok", "service": "markitdown"}

@app.post("/v1/convert/file")
@app.post("/convert/file")
async def convert_file(file: UploadFile = File(...)):
    """Converts uploaded file to Markdown using Microsoft MarkItDown."""
    try:
        suffix = os.path.splitext(file.filename)[1] if file.filename else ".tmp"
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            content = await file.read()
            tmp.write(content)
            tmp_path = tmp.name

        result = markitdown_converter.convert(tmp_path)
        os.unlink(tmp_path)

        markdown_text = result.text_content if hasattr(result, "text_content") else str(result)
        return {
            "status": "success",
            "filename": file.filename,
            "engine": "Microsoft MarkItDown",
            "markdown": markdown_text
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/v1/convert/source")
@app.post("/convert/source")
async def convert_source(request: UrlConvertRequest):
    """Converts remote URL content to Markdown using Microsoft MarkItDown."""
    try:
        result = markitdown_converter.convert(request.url)
        markdown_text = result.text_content if hasattr(result, "text_content") else str(result)
        return {
            "status": "success",
            "url": request.url,
            "engine": "Microsoft MarkItDown",
            "markdown": markdown_text
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
