import asyncio
import contextlib
import logging
import os
import sys
import tempfile
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType
from urllib.error import URLError
from urllib.parse import urlparse
from urllib.request import Request, urlopen

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import JSONResponse
import pdfplumber
import numpy as np
import pypdfium2 as pdfium
from rapidocr import RapidOCR


def _install_docling_import_shim() -> None:
    module_name = "docling.models.stages.chart_extraction.granite_vision"
    if module_name in sys.modules:
        return

    shim = ModuleType(module_name)

    class _DisabledChartExtractionModel:
        _model_repo_folder = "disabled"

        def __init__(self, *args, **kwargs):
            self.enabled = bool(kwargs.get("enabled", False))
            self.elements_batch_size = 1

        @classmethod
        def download_models(cls, *args, **kwargs):
            return None

        def prepare_element(self, *args, **kwargs):
            return None

        def __call__(self, *args, **kwargs):
            return []

    shim.ChartExtractionModelGraniteVision = _DisabledChartExtractionModel
    shim.ChartExtractionModelGraniteVisionV4 = _DisabledChartExtractionModel
    sys.modules[module_name] = shim

    def _ensure_module(name: str) -> ModuleType:
        module = sys.modules.get(name)
        if module is None:
            module = ModuleType(name)
            sys.modules[name] = module
        return module

    _ensure_module("docling_ibm_models")
    _ensure_module("docling_ibm_models.list_item_normalizer")
    _ensure_module("docling_ibm_models.reading_order")
    _ensure_module("docling_ibm_models.layoutmodel")
    _ensure_module("docling_ibm_models.tableformer")
    _ensure_module("docling_ibm_models.tableformer.data_management")
    layout_model_module = _ensure_module("docling.models.stages.layout.layout_model")
    table_structure_model_module = _ensure_module(
        "docling.models.stages.table_structure.table_structure_model"
    )
    table_structure_granite_module = _ensure_module(
        "docling.models.stages.table_structure.table_structure_model_granite_vision"
    )
    table_structure_v2_module = _ensure_module(
        "docling.models.stages.table_structure.table_structure_model_v2"
    )

    list_marker_module = _ensure_module(
        "docling_ibm_models.list_item_normalizer.list_marker_processor"
    )
    reading_order_module = _ensure_module(
        "docling_ibm_models.reading_order.reading_order_rb"
    )
    layout_module = _ensure_module("docling_ibm_models.layoutmodel.layout_predictor")
    tableformer_common_module = _ensure_module("docling_ibm_models.tableformer.common")
    tableformer_predictor_module = _ensure_module(
        "docling_ibm_models.tableformer.data_management.tf_predictor"
    )
    tableformer_v2_module = _ensure_module("docling_ibm_models.tableformer_v2")

    class _ListItemMarkerProcessor:
        def process_list_item(self, *args, **kwargs):
            return None

    @dataclass
    class _ReadingOrderPageElement:
        cid: int
        ref: object
        text: str
        page_no: int
        page_size: object
        label: object
        l: float
        r: float
        b: float
        t: float
        coord_origin: object

    class _ReadingOrderPredictor:
        def predict_reading_order(self, page_elements, *args, **kwargs):
            return list(page_elements)

        def predict_to_captions(self, *args, **kwargs):
            return {}

        def predict_to_footnotes(self, *args, **kwargs):
            return {}

        def predict_merges(self, *args, **kwargs):
            return {}

    class _LayoutPredictor:
        def __init__(self, *args, **kwargs):
            return None

        def predict_batch(self, pages, *args, **kwargs):
            return [[] for _ in pages]

    class _TableFormerV2:
        def __init__(self, *args, **kwargs):
            return None

    class _LayoutModel:
        from docling_core.types.doc import DocItemLabel as _DocItemLabel

        TEXT_ELEM_LABELS = [
            _DocItemLabel.TEXT,
            _DocItemLabel.FOOTNOTE,
            _DocItemLabel.CAPTION,
            _DocItemLabel.CHECKBOX_UNSELECTED,
            _DocItemLabel.CHECKBOX_SELECTED,
            _DocItemLabel.SECTION_HEADER,
            _DocItemLabel.PAGE_HEADER,
            _DocItemLabel.PAGE_FOOTER,
            _DocItemLabel.CODE,
            _DocItemLabel.LIST_ITEM,
            _DocItemLabel.FORMULA,
        ]
        PAGE_HEADER_LABELS = [_DocItemLabel.PAGE_HEADER, _DocItemLabel.PAGE_FOOTER]
        TABLE_LABELS = [_DocItemLabel.TABLE, _DocItemLabel.DOCUMENT_INDEX]
        FIGURE_LABEL = _DocItemLabel.PICTURE
        FORMULA_LABEL = _DocItemLabel.FORMULA
        CONTAINER_LABELS = [
            _DocItemLabel.FORM,
            _DocItemLabel.KEY_VALUE_REGION,
        ]

        def __init__(self, *args, **kwargs):
            self.enabled = bool(kwargs.get("enabled", True))

        @classmethod
        def get_options_type(cls):
            from docling.datamodel.pipeline_options import LayoutOptions

            return LayoutOptions

        def predict_layout(self, conv_res, pages):
            from docling.datamodel.base_models import (
                Cluster,
                LayoutPrediction,
            )
            from docling_core.types.doc import BoundingBox, DocItemLabel

            predictions = []
            for page in pages:
                if page.size is None:
                    predictions.append(LayoutPrediction())
                    continue

                cluster = Cluster(
                    id=0,
                    label=DocItemLabel.TEXT,
                    confidence=0.0,
                    bbox=BoundingBox(
                        l=0.0,
                        t=0.0,
                        r=page.size.width,
                        b=page.size.height,
                    ),
                    cells=[],
                )
                predictions.append(LayoutPrediction(clusters=[cluster]))
            return predictions

        def __call__(self, conv_res, page_batch):
            pages = list(page_batch)
            predictions = self.predict_layout(conv_res, pages)
            for page, prediction in zip(pages, predictions):
                page.predictions.layout = prediction
                yield page

    class _TableStructureModel:
        def __init__(self, *args, **kwargs):
            self.enabled = bool(kwargs.get("enabled", True))

        @classmethod
        def get_options_type(cls):
            from docling.datamodel.pipeline_options import TableStructureOptions

            return TableStructureOptions

        def predict_tables(self, conv_res, pages):
            from docling.datamodel.base_models import TableStructurePrediction

            return [TableStructurePrediction() for _ in pages]

        def __call__(self, conv_res, page_batch):
            if not getattr(self, "enabled", True):
                yield from page_batch
                return
            pages = list(page_batch)
            predictions = self.predict_tables(conv_res, pages)
            for page, prediction in zip(pages, predictions):
                page.predictions.tablestructure = prediction
                yield page

    class _GraniteVisionTableStructureModel:
        def __init__(self, *args, **kwargs):
            self.enabled = bool(kwargs.get("enabled", True))

        @classmethod
        def get_options_type(cls):
            from docling.datamodel.pipeline_options import TableStructureOptions

            return TableStructureOptions

        def predict_tables(self, conv_res, pages):
            from docling.datamodel.base_models import TableStructurePrediction

            return [TableStructurePrediction() for _ in pages]

        def __call__(self, conv_res, page_batch):
            if not getattr(self, "enabled", True):
                yield from page_batch
                return
            pages = list(page_batch)
            predictions = self.predict_tables(conv_res, pages)
            for page, prediction in zip(pages, predictions):
                page.predictions.tablestructure = prediction
                yield page

    class _TableStructureModelV2:
        def __init__(self, *args, **kwargs):
            self.enabled = bool(kwargs.get("enabled", True))

        @classmethod
        def get_options_type(cls):
            from docling.datamodel.pipeline_options import TableStructureV2Options

            return TableStructureV2Options

        def predict_tables(self, conv_res, pages):
            from docling.datamodel.base_models import TableStructurePrediction

            return [TableStructurePrediction() for _ in pages]

        def __call__(self, conv_res, page_batch):
            if not getattr(self, "enabled", True):
                yield from page_batch
                return
            pages = list(page_batch)
            predictions = self.predict_tables(conv_res, pages)
            for page, prediction in zip(pages, predictions):
                page.predictions.tablestructure = prediction
                yield page

    list_marker_module.ListItemMarkerProcessor = _ListItemMarkerProcessor
    reading_order_module.PageElement = _ReadingOrderPageElement
    reading_order_module.ReadingOrderPredictor = _ReadingOrderPredictor
    layout_module.LayoutPredictor = _LayoutPredictor
    tableformer_v2_module.TableFormerV2 = _TableFormerV2
    layout_model_module.LayoutModel = _LayoutModel
    table_structure_model_module.TableStructureModel = _TableStructureModel
    table_structure_granite_module.GraniteVisionTableStructureModel = _GraniteVisionTableStructureModel
    table_structure_v2_module.TableStructureModelV2 = _TableStructureModelV2
    tableformer_common_module.__dict__.setdefault("__all__", [])
    tableformer_predictor_module.__dict__.setdefault("__all__", [])

    accelerator_utils_module = _ensure_module("docling.utils.accelerator_utils")

    def _decide_device(accelerator_device: str, supported_devices=None) -> str:
        return "cpu"

    accelerator_utils_module.decide_device = _decide_device


_install_docling_import_shim()

from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import PdfPipelineOptions, RapidOcrOptions
from docling.document_converter import DocumentConverter
from docling.document_converter import PdfFormatOption

app = FastAPI(title="Docling Converter", version="1.0.0")

_executor = ThreadPoolExecutor(max_workers=2)
_logger = logging.getLogger(__name__)
_rapidocr_engine: RapidOCR | None = None
_pdf_options = PdfPipelineOptions(
    do_ocr=True,
    do_table_structure=True,
    ocr_options=RapidOcrOptions(
        lang=["en"], backend="onnxruntime", force_full_page_ocr=True
    ),
)
_converter = DocumentConverter(
    format_options={
        InputFormat.PDF: PdfFormatOption(pipeline_options=_pdf_options),
    }
)


def _convert_source(source: str) -> str:
    temp_path = None
    try:
        source_path = source
        if _is_url(source):
            source_path = _download_url_to_tempfile(source)
            temp_path = source_path

        result = _converter.convert(source_path)
        document = getattr(result, "document", None)
        markdown = ""
        if document is not None:
            markdown = document.export_to_markdown() or ""

        if _is_pdf_file(source_path):
            table_markdown = _extract_pdf_tables(source_path)
            if table_markdown:
                if markdown.strip():
                    return f"{markdown.rstrip()}\n\n{table_markdown.lstrip()}"
                return table_markdown

            if not markdown.strip():
                ocr_markdown = _extract_pdf_ocr(source_path)
                if ocr_markdown:
                    return ocr_markdown

        if not markdown.strip() and _is_html_file(source_path):
            html_markdown = _extract_html_text(source_path)
            if html_markdown:
                return html_markdown

        return markdown
    finally:
        if temp_path and os.path.exists(temp_path):
            with contextlib.suppress(OSError):
                os.unlink(temp_path)


def _is_url(source: str) -> bool:
    return source.startswith("http://") or source.startswith("https://")


def _download_url_to_tempfile(url: str) -> str:
    request = Request(
        url,
        headers={
            "User-Agent": (
                "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/126.0.0.0 Safari/537.36"
            )
        },
    )
    try:
        with urlopen(request, timeout=120) as response:
            content_type = response.headers.get_content_type()
            path_suffix = Path(urlparse(url).path).suffix.lower()
            if content_type == "application/pdf" or path_suffix == ".pdf":
                suffix = ".pdf"
            elif content_type in {"text/html", "application/xhtml+xml", "text/xml", "application/xml"}:
                suffix = ".html"
            else:
                suffix = path_suffix if path_suffix in {".pdf", ".html", ".htm", ".xml", ".txt"} else ".html"
            payload = response.read()
    except URLError as exc:
        raise RuntimeError(f"Failed to fetch URL: {exc}") from exc

    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(payload)
        tmp.flush()
        return tmp.name


def _is_pdf_file(source: str) -> bool:
    return Path(source).suffix.lower() == ".pdf"


def _is_html_file(source: str) -> bool:
    return Path(source).suffix.lower() in {".html", ".htm"}


def _clean_cell(value: str | None) -> str:
    if value is None:
        return ""
    return " ".join(str(value).replace("\n", " ").split()).strip()


def _normalize_table(table: list[list[str | None]]) -> list[list[str]]:
    rows: list[list[str]] = []
    for raw_row in table or []:
        cleaned = [_clean_cell(cell) for cell in (raw_row or [])]
        while cleaned and not cleaned[-1]:
            cleaned.pop()
        if any(cleaned):
            rows.append(cleaned)
    return rows


def _table_to_markdown(rows: list[list[str]]) -> str:
    if len(rows) < 2:
        return ""

    width = max(len(row) for row in rows)
    if width < 2:
        return ""

    normalized = [row + [""] * (width - len(row)) for row in rows]
    header = normalized[0]
    if not any(header):
        return ""

    lines = [
        "| " + " | ".join(cell.replace("|", "\\|") for cell in header) + " |",
        "| " + " | ".join("---" for _ in header) + " |",
    ]
    for row in normalized[1:]:
        lines.append("| " + " | ".join(cell.replace("|", "\\|") for cell in row) + " |")
    return "\n".join(lines)


def _get_rapidocr_engine() -> RapidOCR:
    global _rapidocr_engine
    if _rapidocr_engine is None:
        _rapidocr_engine = RapidOCR()
    return _rapidocr_engine


def _extract_pdf_ocr(file_path: str) -> str:
    sections: list[str] = []
    try:
        document = pdfium.PdfDocument(file_path)
        engine = _get_rapidocr_engine()
        for page_index in range(len(document)):
            page = document[page_index]
            image = page.render(scale=2).to_pil()
            result = engine(np.array(image))
            lines = [text.strip() for text in getattr(result, "txts", ()) if text and text.strip()]
            if lines:
                sections.append(f"### Page {page_index + 1}\n\n" + "\n".join(lines))
    except Exception:
        _logger.exception("Docling OCR fallback failed for %s", file_path)

    if not sections:
        return ""
    return "## OCR text\n\n" + "\n\n".join(sections)


def _extract_html_text(file_path: str) -> str:
    try:
        from bs4 import BeautifulSoup

        html = Path(file_path).read_text(encoding="utf-8", errors="ignore")
        soup = BeautifulSoup(html, "html.parser")
        title = soup.title.get_text(" ", strip=True) if soup.title else ""
        lines = [
            line.strip()
            for line in soup.get_text("\n").splitlines()
            if line and line.strip()
        ]
        if not title and not lines:
            return ""

        parts: list[str] = []
        if title:
            parts.append(f"# {title}")
        if lines:
            parts.append("\n".join(lines))
        return "\n\n".join(parts)
    except Exception:
        _logger.exception("Docling HTML extraction failed for %s", file_path)
        return ""


def _extract_pdf_tables(file_path: str) -> str:
    sections: list[str] = []
    try:
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
                        width = max((len(row) for row in rows), default=0)
                        score = (len(rows) * width) + sum(1 for row in rows for cell in row if cell) if len(rows) >= 2 and width >= 2 else 0
                        if score > best_score:
                            best_rows = rows
                            best_score = score
                if best_score >= 10:
                    table_md = _table_to_markdown(best_rows)
                    if table_md:
                        sections.append(f"### Page {page_index}\n\n{table_md}")
    except Exception:
        _logger.exception("Docling table extraction failed for %s", file_path)

    if not sections:
        return ""
    return "## Extracted tables\n\n" + "\n\n".join(sections)


async def _convert_file_async(file_path: str) -> str:
    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(_executor, _convert_source, file_path)


async def _convert_url_async(url: str) -> str:
    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(_executor, _convert_source, url)


@app.get("/health")
async def health():
    return {
        "status": "ok",
        "backend": "docling",
        "markdown_conversion": True,
        "python_version": os.sys.version.split()[0],
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

        return JSONResponse(
            {
                "success": True,
                "markdown": markdown,
                "filename": file.filename,
                "size_bytes": len(markdown.encode("utf-8")),
            }
        )
    except asyncio.TimeoutError:
        raise HTTPException(status_code=408, detail="Conversion timed out (420s)")
    except Exception as exc:
        _logger.exception("Docling file conversion failed")
        raise HTTPException(status_code=400, detail=str(exc))
    finally:
        if tmp_path and os.path.exists(tmp_path):
            os.unlink(tmp_path)


@app.post("/convert-url")
async def convert_url(url: str):
    if not url.startswith("http"):
        raise HTTPException(status_code=400, detail="URL must start with http")
    try:
        markdown = await asyncio.wait_for(_convert_url_async(url), timeout=420.0)
        return JSONResponse({"success": True, "markdown": markdown, "url": url})
    except asyncio.TimeoutError:
        raise HTTPException(status_code=408, detail="URL fetch timed out (420s)")
    except Exception as exc:
        _logger.exception("Docling URL conversion failed")
        raise HTTPException(status_code=400, detail=str(exc))
