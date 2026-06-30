import asyncio
import logging
import os
import tempfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from urllib.parse import urlparse

import pdfplumber
import requests
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import JSONResponse
from markitdown import MarkItDown

try:
    from playwright.async_api import async_playwright
except Exception:  # pragma: no cover - browser fallback is optional at runtime
    async_playwright = None

app = FastAPI(title="MarkItDown Converter", version="1.0.0")

_executor = ThreadPoolExecutor(max_workers=4)
_logger = logging.getLogger(__name__)
_text_md = MarkItDown(enable_plugins=False)
_ocr_md = None
_ocr_init_attempted = False
_ocr_init_error = None
_render_timeout_ms = int(os.environ.get("RENDER_TIMEOUT_MS", "15000"))


def _build_md() -> MarkItDown:
    global _ocr_init_attempted, _ocr_init_error
    _ocr_init_attempted = True
    api_key = os.environ.get("GEMINI_API_KEY", "")
    if api_key:
        try:
            from openai import OpenAI

            llm_client = OpenAI(
                api_key=api_key,
                base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
            )
            _ocr_init_error = None
            return MarkItDown(
                enable_plugins=True,
                llm_client=llm_client,
                llm_model=os.environ.get("GEMINI_MODEL", "gemini-2.5-flash"),
            )
        except Exception:
            _ocr_init_error = "Failed to initialize Gemini OCR support"
            _logger.exception("Failed to initialize Gemini OCR support; falling back to text-only conversion")
    return MarkItDown(enable_plugins=False)


def _get_ocr_md() -> MarkItDown | None:
    global _ocr_md
    if _ocr_md is None and os.environ.get("GEMINI_API_KEY", "") and not _ocr_init_attempted:
        _ocr_md = _build_md()
    return _ocr_md


def _convert_with_markitdown(md: MarkItDown, source: str) -> str:
    result = md.convert(source)
    return result.markdown or ""


def _clean_cell(value: str | None) -> str:
    if value is None:
        return ""
    return " ".join(str(value).replace("\n", " ").split()).strip()


def _escape_md_cell(value: str) -> str:
    return value.replace("|", "\\|")


def _normalize_table(table: list[list[str | None]]) -> list[list[str]]:
    rows: list[list[str]] = []
    for raw_row in table or []:
        cleaned = [_clean_cell(cell) for cell in (raw_row or [])]
        while cleaned and not cleaned[-1]:
            cleaned.pop()
        if any(cleaned):
            rows.append(cleaned)
    return rows


def _table_score(rows: list[list[str]]) -> int:
    if len(rows) < 2:
        return 0

    width = max(len(row) for row in rows)
    if width < 2:
        return 0

    non_empty = sum(1 for row in rows for cell in row if cell)
    return (len(rows) * width) + non_empty


def _table_to_markdown(rows: list[list[str]]) -> str:
    if len(rows) < 2:
        return ""

    width = max(len(row) for row in rows)
    if width < 2:
        return ""

    normalized: list[list[str]] = []
    for row in rows:
        normalized.append(row + [""] * (width - len(row)))

    header = normalized[0]
    if not any(header):
        return ""

    lines = [
        "| " + " | ".join(_escape_md_cell(cell) for cell in header) + " |",
        "| " + " | ".join("---" for _ in header) + " |",
    ]
    for row in normalized[1:]:
        lines.append("| " + " | ".join(_escape_md_cell(cell) for cell in row) + " |")
    return "\n".join(lines)


def _extract_pdf_tables(file_path: str) -> str:
    sections: list[str] = []
    with pdfplumber.open(file_path) as pdf:
        for page_index, page in enumerate(pdf.pages, start=1):
            best_rows: list[list[str]] = []
            best_score = 0
            strategies: list[dict[str, object] | None] = [
                None,
                {"vertical_strategy": "text", "horizontal_strategy": "text", "intersection_tolerance": 5},
                {"vertical_strategy": "text", "horizontal_strategy": "lines", "intersection_tolerance": 5},
            ]
            for settings in strategies:
                try:
                    page_tables = page.extract_tables(table_settings=settings) if settings else page.extract_tables()
                except Exception:
                    continue
                for table in page_tables or []:
                    rows = _normalize_table(table)
                    score = _table_score(rows)
                    if score > best_score:
                        best_rows = rows
                        best_score = score
            if best_score >= 50:
                table_md = _table_to_markdown(best_rows)
                if table_md:
                    sections.append(f"### Page {page_index}\n\n{table_md}")

    if not sections:
        return ""
    return "## Extracted tables\n\n" + "\n\n".join(sections)


def _convert_pdf_sync(file_path: str) -> str:
    text_md = _convert_with_markitdown(_text_md, file_path)
    table_md = _extract_pdf_tables(file_path)
    if table_md:
        if text_md.strip():
            return f"{text_md.rstrip()}\n\n{table_md.lstrip()}"
        return table_md

    if text_md.strip():
        return text_md

    ocr_md = _get_ocr_md()
    if ocr_md:
        ocr_md_text = _convert_with_markitdown(ocr_md, file_path)
        if ocr_md_text.strip():
            return ocr_md_text

    return text_md


def _convert_local_sync(file_path: str) -> str:
    suffix = Path(file_path).suffix.lower()
    if suffix == ".pdf":
        return _convert_pdf_sync(file_path)
    return _convert_with_markitdown(_text_md, file_path)


def _probe_url_content_type(url: str) -> str:
    try:
        response = requests.head(url, allow_redirects=True, timeout=10)
        response.raise_for_status()
        return response.headers.get("content-type", "").lower()
    except Exception:
        return ""


def _download_url_to_tempfile(url: str) -> str:
    response = requests.get(url, allow_redirects=True, timeout=30)
    response.raise_for_status()
    suffix = ".pdf" if "application/pdf" in response.headers.get("content-type", "").lower() else ".html"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(response.content)
        tmp.flush()
        return tmp.name


async def _render_url_to_markdown(url: str) -> str:
    if async_playwright is None:
        return ""

    browser = None
    tmp_path = None
    try:
        async with async_playwright() as p:
            browser = await p.chromium.launch(headless=True)
            page = await browser.new_page(viewport={"width": 1440, "height": 2400})
            await page.goto(url, wait_until="domcontentloaded", timeout=_render_timeout_ms)
            try:
                await page.wait_for_load_state("networkidle", timeout=min(_render_timeout_ms, 10000))
            except Exception:
                pass
            rendered_html = await page.content()
            if not rendered_html.strip():
                return ""
            with tempfile.NamedTemporaryFile(delete=False, suffix=".html", mode="w", encoding="utf-8") as tmp:
                tmp.write(rendered_html)
                tmp_path = tmp.name
            loop = asyncio.get_running_loop()
            return await loop.run_in_executor(_executor, _convert_local_sync, tmp_path)
    except Exception as exc:
        _logger.info("Browser render fallback failed for %s: %s", url, exc)
        return ""
    finally:
        if tmp_path and os.path.exists(tmp_path):
            os.unlink(tmp_path)
        if browser is not None:
            await browser.close()


async def _convert_url_async(url: str) -> str:
    loop = asyncio.get_running_loop()
    path = urlparse(url).path.lower()
    content_type = await loop.run_in_executor(_executor, _probe_url_content_type, url)
    looks_like_pdf = path.endswith(".pdf") or "application/pdf" in content_type

    if looks_like_pdf:
        tmp_path = None
        try:
            tmp_path = await loop.run_in_executor(_executor, _download_url_to_tempfile, url)
            return await loop.run_in_executor(_executor, _convert_local_sync, tmp_path)
        finally:
            if tmp_path and os.path.exists(tmp_path):
                os.unlink(tmp_path)

    source_markdown = await loop.run_in_executor(_executor, _convert_with_markitdown, _text_md, url)
    if source_markdown.strip():
        return source_markdown

    rendered_markdown = await _render_url_to_markdown(url)
    if rendered_markdown.strip():
        return rendered_markdown

    return source_markdown


async def _convert_file_async(file_path: str) -> str:
    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(_executor, _convert_local_sync, file_path)


@app.get("/health")
async def health():
    return {
        "status": "ok",
        "ocr_configured": bool(os.environ.get("GEMINI_API_KEY", "")),
        "ocr_loaded": _ocr_md is not None,
        "browser_fallback": async_playwright is not None,
        "model": os.environ.get("GEMINI_MODEL", "gemini-2.5-flash") if _ocr_md is not None else None,
        "ocr_error": _ocr_init_error,
    }


@app.post("/convert")
async def convert_file(file: UploadFile = File(...)):
    suffix = os.path.splitext(file.filename or "")[1] or ".bin"
    tmp_path = None
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            content = await file.read()
            tmp.write(content)
            tmp.flush()
            tmp_path = tmp.name

        markdown = await asyncio.wait_for(_convert_file_async(tmp_path), timeout=420.0)

        return JSONResponse({
            "success": True,
            "markdown": markdown,
            "filename": file.filename,
            "size_bytes": len(markdown.encode("utf-8")),
        })
    except asyncio.TimeoutError:
        raise HTTPException(status_code=408, detail="Conversion timed out (420s)")
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    finally:
        if tmp_path and os.path.exists(tmp_path):
            os.unlink(tmp_path)


@app.post("/convert-url")
async def convert_url(url: str):
    if not url.startswith("http"):
        raise HTTPException(status_code=400, detail="URL must start with http")
    try:
        markdown = await asyncio.wait_for(_convert_url_async(url), timeout=180.0)
        return JSONResponse({"success": True, "markdown": markdown, "url": url})
    except asyncio.TimeoutError:
        raise HTTPException(status_code=408, detail="URL fetch timed out (180s)")
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc))
