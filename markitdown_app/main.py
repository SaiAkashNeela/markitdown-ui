import os
import re
import tempfile
from typing import Optional
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from markitdown import MarkItDown
from markitdown._base_converter import DocumentConverterResult
from markitdown.converters._image_converter import ImageConverter

# Custom Converter to completely ignore standalone image files (PNG, JPG, WEBP, etc.)
class IgnoreImageConverter(ImageConverter):
    def convert(self, file_stream, stream_info, **kwargs) -> DocumentConverterResult:
        return DocumentConverterResult(text_content="")

def strip_images(markdown_text: str) -> str:
    """Strips any embedded markdown/HTML image tags from converted text."""
    if not markdown_text:
        return ""
    # Strip markdown image syntax: ![alt](url)
    text = re.sub(r'!\[.*?\]\(.*?\)', '', markdown_text)
    # Strip HTML <img> tags
    text = re.sub(r'<img\b[^>]*>', '', text, flags=re.IGNORECASE)
    # Clean up excessive blank lines
    text = re.sub(r'\n{3,}', '\n\n', text)
    return text.strip()

app = FastAPI(
    title="Microsoft MarkItDown Conversion Service",
    version="1.1.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

markitdown_converter = MarkItDown()
# Register IgnoreImageConverter with highest priority to intercept image files
markitdown_converter.register_converter(IgnoreImageConverter(), priority=100.0)

class UrlConvertRequest(BaseModel):
    url: str

@app.get("/health")
def health_check():
    return {"status": "ok", "service": "markitdown", "image_processing": "ignored"}

@app.post("/v1/convert/file")
@app.post("/convert/file")
async def convert_file(file: UploadFile = File(...)):
    """Converts uploaded file to Markdown using Microsoft MarkItDown, ignoring all detected images."""
    try:
        suffix = os.path.splitext(file.filename)[1] if file.filename else ".tmp"
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            content = await file.read()
            tmp.write(content)
            tmp_path = tmp.name

        result = markitdown_converter.convert(tmp_path)
        os.unlink(tmp_path)

        raw_text = result.text_content if hasattr(result, "text_content") else str(result)
        cleaned_markdown = strip_images(raw_text)

        return {
            "status": "success",
            "filename": file.filename,
            "engine": "Microsoft MarkItDown (Images Ignored)",
            "markdown": cleaned_markdown
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/v1/convert/source")
@app.post("/convert/source")
async def convert_source(request: UrlConvertRequest):
    """Converts remote URL content to Markdown using Microsoft MarkItDown, ignoring all detected images."""
    try:
        result = markitdown_converter.convert(request.url)
        raw_text = result.text_content if hasattr(result, "text_content") else str(result)
        cleaned_markdown = strip_images(raw_text)

        return {
            "status": "success",
            "url": request.url,
            "engine": "Microsoft MarkItDown (Images Ignored)",
            "markdown": cleaned_markdown
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
