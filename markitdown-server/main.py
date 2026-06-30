import asyncio
import logging
import os
import re
import tempfile
from concurrent.futures import ThreadPoolExecutor
from itertools import zip_longest
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


def _normalize_ws(text: str) -> str:
    return " ".join(text.split()).strip()


def _cluster_words_into_rows(words: list[dict[str, object]], tolerance: float = 2.0) -> list[list[dict[str, object]]]:
    sorted_words = sorted(words, key=lambda word: (float(word["top"]), float(word["x0"])))
    rows: list[list[dict[str, object]]] = []
    current_row: list[dict[str, object]] = []
    current_top: float | None = None

    for word in sorted_words:
        top = float(word["top"])
        if current_top is None or abs(top - current_top) <= tolerance:
            current_row.append(word)
            if current_top is None:
                current_top = top
            else:
                current_top = (current_top * (len(current_row) - 1) + top) / len(current_row)
        else:
            rows.append(current_row)
            current_row = [word]
            current_top = top

    if current_row:
        rows.append(current_row)

    return rows


def _row_text(row: list[dict[str, object]]) -> str:
    return " ".join(str(word["text"]) for word in sorted(row, key=lambda item: float(item["x0"]))).strip()


def _is_meal_title(text: str) -> bool:
    return bool(re.match(r"^(M\d+|S\d+)\b", text))


def _is_table_header(text: str) -> bool:
    return "Ingredient" in text and "kcal" in text and "Protein" in text


def _build_column_anchors(header_rows: list[list[dict[str, object]]]) -> list[float]:
    header_only_rows = [row for row in header_rows if not re.search(r"\d", _row_text(row))]
    sorted_words = sorted(
        (word for row in header_only_rows for word in row),
        key=lambda word: (float(word["top"]), float(word["x0"])),
    )
    header_keys = [
        ("Ingredient",),
        ("Cup", "Count", "Measure"),
        ("Weight",),
        ("kcal",),
        ("Protein",),
        ("Carbs",),
        ("Fat",),
        ("Fibre",),
        ("Salt",),
    ]
    anchors: list[float | None] = []
    for key_group in header_keys:
        matched = next(
            (
                word
                for word in sorted_words
                if any(str(word["text"]).lower().startswith(key.lower()) for key in key_group)
            ),
            None,
        )
        anchors.append(float(matched["x0"]) if matched else None)

    known_positions = [anchor for anchor in anchors if anchor is not None]
    if len(known_positions) < 2:
        return []

    filled: list[float] = []
    last_known: float | None = None
    next_known: list[float | None] = [None] * len(anchors)
    upcoming: float | None = None
    for index in range(len(anchors) - 1, -1, -1):
        if anchors[index] is not None:
            upcoming = anchors[index]
        next_known[index] = upcoming

    for index, anchor in enumerate(anchors):
        if anchor is not None:
            last_known = anchor
            filled.append(anchor)
        elif last_known is not None:
            filled.append(last_known)
        elif next_known[index] is not None:
            filled.append(next_known[index])

    return filled


def _assign_row_to_cells(row: list[dict[str, object]], anchors: list[float]) -> list[str]:
    cells = ["" for _ in anchors]
    for word in sorted(row, key=lambda item: float(item["x0"])):
        x0 = float(word["x0"])
        cell_index = min(range(len(anchors)), key=lambda index: abs(x0 - anchors[index]))
        cell_text = str(word["text"]).strip()
        if cell_text:
            cells[cell_index] = f"{cells[cell_index]} {cell_text}".strip() if cells[cell_index] else cell_text
    return cells


def _merge_rows(left: list[str], right: list[str]) -> list[str]:
    merged: list[str] = []
    for left_cell, right_cell in zip_longest(left, right, fillvalue=""):
        pieces = [piece for piece in (left_cell, right_cell) if piece]
        merged.append(" ".join(pieces).strip())
    return merged


def _extract_word_based_pdf_tables(file_path: str) -> str:
    sections: list[str] = []
    with pdfplumber.open(file_path) as pdf:
        for page_index, page in enumerate(pdf.pages, start=1):
            rows = _cluster_words_into_rows(page.extract_words(use_text_flow=True, keep_blank_chars=False) or [])
            title_indexes = [index for index, row in enumerate(rows) if _is_meal_title(_row_text(row))]
            if not title_indexes:
                continue

            page_sections: list[str] = []
            for title_pos, title_index in enumerate(title_indexes):
                next_title_index = title_indexes[title_pos + 1] if title_pos + 1 < len(title_indexes) else len(rows)
                title_text = _row_text(rows[title_index])
                header_index = None
                for candidate_index in range(title_index + 1, next_title_index):
                    if _is_table_header(_row_text(rows[candidate_index])):
                        header_index = candidate_index
                        break
                if header_index is None:
                    continue

                header_scan_rows = rows[title_index + 1 : min(next_title_index, header_index + 4)]
                anchors = _build_column_anchors(header_scan_rows)
                if len(anchors) < 2:
                    continue

                table_rows: list[list[str]] = []
                pending_prefix: list[str] | None = None

                for row in rows[header_index + 1 : next_title_index]:
                    row_text = _row_text(row)
                    if not row_text:
                        continue
                    if row_text in {"Indian", "Western", "Snack"}:
                        continue
                    if _is_meal_title(row_text):
                        break
                    if row_text.startswith("TOTAL"):
                        if pending_prefix:
                            pending_prefix = None
                        total_cells = _assign_row_to_cells(row, anchors)
                        if any(total_cells):
                            table_rows.append(total_cells)
                        break

                    cells = _assign_row_to_cells(row, anchors)
                    has_numeric_payload = any(re.search(r"\d", cell) for cell in cells[2:] if cell)
                    non_empty_cells = sum(1 for cell in cells if cell)

                    if has_numeric_payload:
                        if pending_prefix:
                            cells = _merge_rows(pending_prefix, cells)
                            pending_prefix = None
                        table_rows.append(cells)
                        continue

                    if non_empty_cells >= 2 and cells[0]:
                        pending_prefix = _merge_rows(pending_prefix, cells) if pending_prefix else cells
                        continue

                    if table_rows:
                        table_rows[-1] = _merge_rows(table_rows[-1], cells)

                if len(table_rows) >= 2:
                    table_md = _table_to_markdown(
                        [
                            [
                                "Ingredient",
                                "Cup / Count Measure",
                                "Weight",
                                "kcal",
                                "Protein",
                                "Carbs (sugar)",
                                "Fat (sat)",
                                "Fibre",
                                "Salt",
                            ]
                        ]
                        + table_rows
                    )
                    if table_md:
                        page_sections.append(f"### {title_text}\n\n{table_md}")

            if page_sections:
                sections.extend(page_sections)

    if not sections:
        return ""
    return "## Extracted tables\n\n" + "\n\n".join(sections)


def _inline_table_sections(text_md: str, table_md: str) -> str:
    table_sections: dict[str, tuple[str, str]] = {}
    current_title = ""
    current_body: list[str] = []

    for line in table_md.splitlines():
        if line.startswith("### "):
            if current_title:
                normalized_title = _normalize_ws(current_title)
                table_sections[normalized_title] = (current_title, "\n".join(current_body).strip())
            current_title = line[4:].strip()
            current_body = []
        elif line.startswith("## "):
            continue
        else:
            current_body.append(line)

    if current_title:
        normalized_title = _normalize_ws(current_title)
        table_sections[normalized_title] = (current_title, "\n".join(current_body).strip())

    if not table_sections:
        return text_md

    title_keys = sorted(table_sections.keys(), key=len, reverse=True)
    note_prefixes = ("Tip:", "Best ", "Highest ", "Lowest ", "* ", "Cup =")
    lines = text_md.splitlines()
    merged_lines: list[str] = []
    index = 0

    while index < len(lines):
        current_line = lines[index]
        normalized_line = _normalize_ws(current_line)
        matched_key = next((key for key in title_keys if normalized_line == key), None)

        if not matched_key:
            merged_lines.append(current_line)
            index += 1
            continue

        title, body = table_sections[matched_key]
        merged_lines.append(title)
        if body:
            merged_lines.append(body)

        next_title_index = index + 1
        while next_title_index < len(lines):
            candidate = _normalize_ws(lines[next_title_index])
            if any(candidate == key for key in title_keys):
                break
            next_title_index += 1

        saw_total = False
        for note_line in lines[index + 1 : next_title_index]:
            normalized_note = _normalize_ws(note_line)
            if not saw_total and normalized_note.startswith("TOTAL"):
                saw_total = True
                continue
            if saw_total and normalized_note and note_line.startswith(note_prefixes):
                merged_lines.append(note_line)

        index = next_title_index

    return "\n".join(merged_lines).strip()


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

    word_based_tables = _extract_word_based_pdf_tables(file_path)
    if word_based_tables:
        return word_based_tables

    if not sections:
        return ""
    return "## Extracted tables\n\n" + "\n\n".join(sections)


def _convert_pdf_sync(file_path: str) -> str:
    text_md = _convert_with_markitdown(_text_md, file_path)
    table_md = _extract_pdf_tables(file_path)
    if table_md:
        if text_md.strip():
            inline_md = _inline_table_sections(text_md, table_md)
            return inline_md if inline_md.strip() else f"{text_md.rstrip()}\n\n{table_md.lstrip()}"
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
