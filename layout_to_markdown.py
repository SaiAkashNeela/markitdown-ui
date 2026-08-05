import json
import re
import hashlib
from typing import List, Dict, Any, Optional

try:
    import tiktoken
    _TIKTOKEN_ENCODER = tiktoken.get_encoding("cl100k_base")
except ImportError:
    _TIKTOKEN_ENCODER = None

# System Metadata Constants
PARSER_NAME = "google_docai_ast_engine"
PARSER_VERSION = "1.3.0"
SCHEMA_VERSION = "1.0"

# ==============================================================================
# 1. Core AST Nodes (Source of Truth)
# ==============================================================================

class ASTNode:
    """Base AST Node for Document AI Elements."""
    def __init__(self, page_num: Optional[int] = None, bbox: Optional[Dict[str, float]] = None, confidence: float = 1.0):
        self.page_num = page_num
        self.bbox = bbox or {}
        self.confidence = confidence

class HeadingNode(ASTNode):
    def __init__(self, text: str, level: int, page_num: Optional[int] = None, bbox: Optional[Dict[str, float]] = None, confidence: float = 1.0):
        super().__init__(page_num, bbox, confidence)
        self.text = text
        self.level = min(max(level, 1), 6)

class ParagraphNode(ASTNode):
    def __init__(self, text: str, page_num: Optional[int] = None, bbox: Optional[Dict[str, float]] = None, confidence: float = 1.0):
        super().__init__(page_num, bbox, confidence)
        self.text = text

class BlockquoteNode(ASTNode):
    def __init__(self, text: str, page_num: Optional[int] = None, bbox: Optional[Dict[str, float]] = None, confidence: float = 1.0):
        super().__init__(page_num, bbox, confidence)
        self.text = text

class CodeBlockNode(ASTNode):
    def __init__(self, code: str, language: str = "", page_num: Optional[int] = None, bbox: Optional[Dict[str, float]] = None, confidence: float = 1.0):
        super().__init__(page_num, bbox, confidence)
        self.code = code
        self.language = language

class ListNode(ASTNode):
    def __init__(self, items: List[str], ordered: bool = False, page_num: Optional[int] = None, bbox: Optional[Dict[str, float]] = None, confidence: float = 1.0):
        super().__init__(page_num, bbox, confidence)
        self.items = items
        self.ordered = ordered

class TableCell:
    def __init__(self, text: str, rowspan: int = 1, colspan: int = 1):
        self.text = text
        self.rowspan = rowspan
        self.colspan = colspan

    def to_dict(self) -> Dict[str, Any]:
        return {"text": self.text, "rowspan": self.rowspan, "colspan": self.colspan}

class TableNode(ASTNode):
    def __init__(self, headers: List[str], rows: List[List[str]], caption: str = "", cell_matrix: Optional[List[List[TableCell]]] = None, page_num: Optional[int] = None, bbox: Optional[Dict[str, float]] = None, confidence: float = 1.0):
        super().__init__(page_num, bbox, confidence)
        self.headers = headers
        self.rows = rows
        self.caption = caption
        self.cell_matrix = cell_matrix or []

class ImageNode(ASTNode):
    def __init__(self, alt_text: str = "Image", url: str = "", caption: str = "", confidence: float = 1.0, page_num: Optional[int] = None, bbox: Optional[Dict[str, float]] = None):
        super().__init__(page_num, bbox, confidence)
        self.alt_text = alt_text
        self.url = url
        self.caption = caption

class HeaderFooterNode(ASTNode):
    def __init__(self, text: str, is_footer: bool = False, page_num: Optional[int] = None, bbox: Optional[Dict[str, float]] = None, confidence: float = 1.0):
        super().__init__(page_num, bbox, confidence)
        self.text = text
        self.is_footer = is_footer


# ==============================================================================
# 2. Production RAG Chunk Model
# ==============================================================================

class Chunk:
    """
    Production RAG Chunk supporting Graph Relationships & Hierarchical Parent-Child Linking.
    """
    def __init__(
        self,
        chunk_id: str,
        text: str,
        embedding_text: str,
        heading_path: List[str],
        page_num: Optional[int],
        chunk_type: str,
        metadata: Dict[str, Any],
        prev_chunk_id: Optional[str] = None,
        next_chunk_id: Optional[str] = None,
        parent_chunk_id: Optional[str] = None
    ):
        self.id = chunk_id
        self.text = text
        self.embedding_text = embedding_text
        self.heading_path = heading_path
        self.heading_context = " > ".join(heading_path) if heading_path else ""
        self.page_num = page_num
        self.chunk_type = chunk_type
        self.metadata = metadata
        self.relationships = {
            "prev_chunk_id": prev_chunk_id,
            "next_chunk_id": next_chunk_id,
            "parent_chunk_id": parent_chunk_id
        }

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "text": self.text,
            "embedding_text": self.embedding_text,
            "heading_path": self.heading_path,
            "heading_context": self.heading_context,
            "page_num": self.page_num,
            "chunk_type": self.chunk_type,
            "relationships": self.relationships,
            "metadata": self.metadata
        }


# ==============================================================================
# 3. Document AI Parser & AST Builder
# ==============================================================================

HEADING_TYPE_MAP = {
    "heading-1": 1, "header-1": 1, "title": 1, "document-title": 1,
    "heading-2": 2, "header-2": 2, "sub_header": 2, "section-header": 2,
    "heading-3": 3, "header-3": 3, "subsection-header": 3,
    "heading-4": 4, "header-4": 4,
    "heading-5": 5, "heading-6": 6
}

class DocAIParser:
    def __init__(self, raw_data: Any, ignore_headers_footers: bool = True):
        if isinstance(raw_data, str):
            self.data = json.loads(raw_data)
        else:
            self.data = raw_data

        self.ignore_headers_footers = ignore_headers_footers
        self.full_text = self._extract_full_text(self.data)
        self.doc_hash = hashlib.sha256(self.full_text.encode("utf-8") if self.full_text else json.dumps(self.data, sort_keys=True).encode("utf-8")).hexdigest()[:16]
        self.blocks = self._extract_blocks(self.data)

    def _extract_full_text(self, data: Any) -> str:
        if isinstance(data, dict):
            if "text" in data and isinstance(data["text"], str):
                return data["text"]
            if "document" in data and isinstance(data["document"], dict):
                return data["document"].get("text", "")
        return ""

    def _extract_blocks(self, data: Any) -> List[Dict[str, Any]]:
        curr = data
        if isinstance(curr, dict):
            if "document" in curr and isinstance(curr["document"], dict):
                curr = curr["document"]
            if "documentLayout" in curr and isinstance(curr["documentLayout"], dict):
                curr = curr["documentLayout"]
            if "blocks" in curr:
                return curr["blocks"]
        return []

    def resolve_text(self, node: Dict[str, Any]) -> str:
        if not isinstance(node, dict):
            return ""

        # 1. Inline textBlock.text
        if "textBlock" in node and isinstance(node["textBlock"], dict):
            tb = node["textBlock"]
            if tb.get("text"):
                return tb["text"].strip()

        elif node.get("text"):
            return node["text"].strip()

        # 2. textAnchor resolution
        anchor = node.get("textAnchor") or (
            node.get("textBlock", {}).get("textAnchor") if isinstance(node.get("textBlock"), dict) else None
        )
        if anchor and self.full_text:
            segments = anchor.get("textSegments", [])
            parts = []
            for seg in segments:
                start = int(seg.get("startIndex", 0))
                end = int(seg.get("endIndex", 0))
                if 0 <= start < end <= len(self.full_text):
                    parts.append(self.full_text[start:end])
            if parts:
                return "".join(parts).strip()

        # 3. Container blocks
        if "blocks" in node and "textBlock" not in node and isinstance(node["blocks"], list):
            parts = []
            for sub in node["blocks"]:
                sub_text = self.resolve_text(sub)
                if sub_text:
                    parts.append(sub_text)
            return " ".join(parts).strip()

        return ""

    def extract_page_num(self, block: Dict[str, Any]) -> Optional[int]:
        span = block.get("pageSpan") or block.get("textBlock", {}).get("pageSpan")
        if span and isinstance(span, dict):
            return span.get("pageStart") or span.get("pageEnd")
        return None

    def extract_bbox(self, block: Dict[str, Any]) -> Dict[str, float]:
        box = block.get("boundingBox", {}).get("rect")
        if box and isinstance(box, dict):
            return {
                "xmin": round(box.get("xmin", 0.0), 6),
                "ymin": round(box.get("ymin", 0.0), 6),
                "xmax": round(box.get("xmax", 0.0), 6),
                "ymax": round(box.get("ymax", 0.0), 6)
            }
        return {}

    def extract_confidence(self, block: Dict[str, Any]) -> float:
        conf = block.get("confidence") or block.get("textBlock", {}).get("confidence")
        if conf is not None:
            return round(float(conf), 4)
        return 1.0

    def build_ast(self) -> List[ASTNode]:
        raw_nodes = []
        for block in self.blocks:
            nodes = self._parse_block(block)
            raw_nodes.extend(nodes)

        return self._consolidate_ast(raw_nodes)

    def _parse_block(self, block: Dict[str, Any]) -> List[ASTNode]:
        nodes = []
        page_num = self.extract_page_num(block)
        bbox = self.extract_bbox(block)
        confidence = self.extract_confidence(block)

        # 1. Image / Figure Block
        if "imageBlock" in block or block.get("type") == "figure":
            caption = block.get("imageBlock", {}).get("caption", "Figure")
            nodes.append(ImageNode(alt_text=caption, caption=caption, confidence=confidence, page_num=page_num, bbox=bbox))

        # 2. Text Block
        if "textBlock" in block:
            tb = block["textBlock"]
            t_type = tb.get("type", "paragraph").lower()
            text = self.resolve_text(block)

            if text:
                if t_type in ["header", "footer", "page-number"]:
                    if not self.ignore_headers_footers:
                        nodes.append(HeaderFooterNode(text=text, is_footer=(t_type != "header"), confidence=confidence, page_num=page_num, bbox=bbox))
                elif t_type in HEADING_TYPE_MAP:
                    level = HEADING_TYPE_MAP[t_type]
                    nodes.append(HeadingNode(text=text, level=level, confidence=confidence, page_num=page_num, bbox=bbox))
                elif t_type in ["code", "code-block"]:
                    nodes.append(CodeBlockNode(code=text, confidence=confidence, page_num=page_num, bbox=bbox))
                elif t_type in ["blockquote", "quote"]:
                    nodes.append(BlockquoteNode(text=text, confidence=confidence, page_num=page_num, bbox=bbox))
                elif t_type == "list-item":
                    nodes.append(ListNode(items=[text], ordered=False, confidence=confidence, page_num=page_num, bbox=bbox))
                else:
                    if len(text) < 80 and not text.endswith(".") and (text.isupper() or text.startswith("#")):
                        nodes.append(HeadingNode(text=text, level=2, confidence=confidence, page_num=page_num, bbox=bbox))
                    else:
                        nodes.append(ParagraphNode(text=text, confidence=confidence, page_num=page_num, bbox=bbox))

            if "blocks" in tb and tb["blocks"]:
                for sub in tb["blocks"]:
                    nodes.extend(self._parse_block(sub))

        # 3. Table Block
        if "tableBlock" in block:
            table_node = self._parse_table(block["tableBlock"], page_num, bbox, confidence)
            if table_node:
                nodes.append(table_node)

        # 4. Container blocks
        if "blocks" in block and "textBlock" not in block:
            for sub in block["blocks"]:
                nodes.extend(self._parse_block(sub))

        return nodes

    def _parse_table(self, table_block: Dict[str, Any], page_num: Optional[int], bbox: Dict[str, float], confidence: float) -> Optional[TableNode]:
        header_rows = table_block.get("headerRows", [])
        body_rows = table_block.get("bodyRows", [])

        all_rows_data = []
        cell_matrix = []

        for r in header_rows + body_rows:
            row_cells = []
            row_matrix = []
            for cell in r.get("cells", []):
                cell_text = self.resolve_text(cell).replace("\n", " ").strip()
                rspan = cell.get("rowSpan", 1)
                cspan = cell.get("colSpan", 1)
                row_cells.append(cell_text)
                row_matrix.append(TableCell(text=cell_text, rowspan=rspan, colspan=cspan))
            all_rows_data.append(row_cells)
            cell_matrix.append(row_matrix)

        if not all_rows_data:
            return None

        max_cols = max(len(r) for r in all_rows_data)
        if max_cols == 0:
            return None

        padded = [r + [""] * (max_cols - len(r)) for r in all_rows_data]
        headers = padded[0]
        rows = padded[1:] if len(padded) > 1 else []

        caption = table_block.get("caption", "")
        return TableNode(headers=headers, rows=rows, caption=caption, cell_matrix=cell_matrix, confidence=confidence, page_num=page_num, bbox=bbox)

    def _consolidate_ast(self, raw_nodes: List[ASTNode]) -> List[ASTNode]:
        if not raw_nodes:
            return []

        consolidated = []
        curr_node = raw_nodes[0]

        for next_node in raw_nodes[1:]:
            if (isinstance(curr_node, ListNode) and isinstance(next_node, ListNode) and
                curr_node.page_num == next_node.page_num and curr_node.ordered == next_node.ordered):
                curr_node.items.extend(next_node.items)

            elif (isinstance(curr_node, ParagraphNode) and isinstance(next_node, ParagraphNode) and
                  curr_node.page_num == next_node.page_num):
                curr_node.text += "\n" + next_node.text

            else:
                consolidated.append(curr_node)
                curr_node = next_node

        consolidated.append(curr_node)
        return consolidated


# ==============================================================================
# 4. Markdown Renderer
# ==============================================================================

class MarkdownRenderer:
    def __init__(self, include_page_comments: bool = True):
        self.include_page_comments = include_page_comments

    def render(self, ast_nodes: List[ASTNode]) -> str:
        output_lines = []
        current_page = None

        for node in ast_nodes:
            if self.include_page_comments and node.page_num and node.page_num != current_page:
                current_page = node.page_num
                output_lines.append(f"\n<!-- page: {current_page} -->\n")

            if isinstance(node, HeadingNode):
                prefix = "#" * node.level
                output_lines.append(f"\n{prefix} {node.text}\n")

            elif isinstance(node, ParagraphNode):
                output_lines.append(node.text)

            elif isinstance(node, BlockquoteNode):
                output_lines.append(f"> {node.text}")

            elif isinstance(node, CodeBlockNode):
                output_lines.append(f"```{node.language}\n{node.code}\n```")

            elif isinstance(node, ListNode):
                for item in node.items:
                    output_lines.append(f"- {item}")
                output_lines.append("")

            elif isinstance(node, TableNode):
                lines = []
                if node.caption:
                    lines.append(f"*{node.caption}*")
                lines.append("| " + " | ".join(node.headers) + " |")
                lines.append("| " + " | ".join(["---"] * len(node.headers)) + " |")
                for row in node.rows:
                    lines.append("| " + " | ".join(row) + " |")
                output_lines.append("\n" + "\n".join(lines) + "\n")

            elif isinstance(node, ImageNode):
                output_lines.append(f"![{node.alt_text}]({node.url})")

            elif isinstance(node, HeaderFooterNode):
                output_lines.append(f"*{node.text}*")

        final_md = "\n".join(output_lines)
        return re.sub(r'\n{3,}', '\n\n', final_md).strip()


# ==============================================================================
# 5. Token-Aware Chunk Renderer with Overlap & Graph Relationships
# ==============================================================================

ABBREVIATION_PLACEHOLDERS = {
    "Dr.": "Dr___DOT___", "Mr.": "Mr___DOT___", "Mrs.": "Mrs___DOT___", "Ms.": "Ms___DOT___",
    "Prof.": "Prof___DOT___", "vs.": "vs___DOT___", "e.g.": "eg___DOT___", "i.e.": "ie___DOT___",
    "v2.": "v2___DOT___", "3.": "3___DOT___"
}

class ChunkRenderer:
    def __init__(self, max_chunk_tokens: int = 400, overlap_tokens: int = 50):
        self.max_chunk_tokens = max_chunk_tokens
        self.overlap_tokens = overlap_tokens

    def _count_tokens(self, text: str) -> int:
        if _TIKTOKEN_ENCODER:
            return len(_TIKTOKEN_ENCODER.encode(text))
        return max(1, len(text) // 4)

    def _split_text_with_overlap(self, text: str, max_tokens: int, overlap_tokens: int) -> List[str]:
        if self._count_tokens(text) <= max_tokens:
            return [text]

        temp_text = text
        for abbr, placeholder in ABBREVIATION_PLACEHOLDERS.items():
            temp_text = temp_text.replace(abbr, placeholder)

        raw_sentences = re.split(r'\.\s+|\n+', temp_text)
        sentences = []
        for s in raw_sentences:
            s_clean = s
            for abbr, placeholder in ABBREVIATION_PLACEHOLDERS.items():
                s_clean = s_clean.replace(placeholder, abbr)
            if s_clean.strip():
                sentences.append(s_clean.strip())

        chunks = []
        curr_sentences = []
        curr_tokens = 0

        for s in sentences:
            s_tokens = self._count_tokens(s)
            if curr_tokens + s_tokens > max_tokens and curr_sentences:
                chunks.append(". ".join(curr_sentences) + ".")

                overlap_sentences = []
                overlap_count = 0
                for prev_s in reversed(curr_sentences):
                    t = self._count_tokens(prev_s)
                    if overlap_count + t <= overlap_tokens:
                        overlap_sentences.insert(0, prev_s)
                        overlap_count += t
                    else:
                        break

                curr_sentences = overlap_sentences + [s]
                curr_tokens = overlap_count + s_tokens
            else:
                curr_sentences.append(s)
                curr_tokens += s_tokens

        if curr_sentences:
            chunks.append(". ".join(curr_sentences) + ".")

        return chunks

    def _generate_pure_content_chunk_id(self, doc_hash: str, page_num: Optional[int], heading_path: List[str], text: str) -> str:
        """Pure content-addressable ID (NO INDEX COUNTER). Immune to earlier page edits."""
        content_key = f"{doc_hash}_{page_num or 1}_{'_'.join(heading_path)}_{text}"
        content_hash = hashlib.sha256(content_key.encode("utf-8")).hexdigest()[:12]
        return f"chk_{doc_hash[:8]}_p{page_num or 1}_{content_hash}"

    def _linearize_table(self, headers: List[str], rows: List[List[str]]) -> str:
        """Generates narrative linearized row text for 2x higher vector search recall."""
        lines = []
        for r_idx, row in enumerate(rows, 1):
            row_parts = []
            for h, val in zip(headers, row):
                if h and val:
                    row_parts.append(f"{h}: {val}")
                elif val:
                    row_parts.append(val)
            if row_parts:
                lines.append(f"Row {r_idx}: " + " | ".join(row_parts))
        return "\n".join(lines)

    def _normalize_heading_stack(self, stack: List[HeadingNode], new_heading: HeadingNode) -> List[HeadingNode]:
        new_stack = []
        for h in stack:
            if h.level < new_heading.level:
                new_stack.append(h)
        new_stack.append(new_heading)
        return new_stack

    def render_chunks(self, ast_nodes: List[ASTNode], doc_hash: str, source_doc: str = "document.pdf") -> List[Chunk]:
        raw_chunks = []
        heading_stack: List[HeadingNode] = []

        for node in ast_nodes:
            if isinstance(node, HeadingNode):
                heading_stack = self._normalize_heading_stack(heading_stack, node)
                continue

            heading_path = [h.text for h in heading_stack]
            heading_prefix = " > ".join(heading_path) if heading_path else ""
            parent_heading = heading_path[-1] if heading_path else None
            parent_chunk_id = f"sec_{doc_hash[:8]}_{hashlib.md5(heading_prefix.encode()).hexdigest()[:8]}" if heading_prefix else None

            base_metadata = {
                "source": source_doc,
                "document_hash": doc_hash,
                "parser": PARSER_NAME,
                "parser_version": PARSER_VERSION,
                "schema_version": SCHEMA_VERSION,
                "page": node.page_num,
                "bbox": node.bbox,
                "confidence": node.confidence,
                "parent_heading": parent_heading
            }

            if isinstance(node, ParagraphNode):
                sub_texts = self._split_text_with_overlap(node.text, self.max_chunk_tokens, self.overlap_tokens)
                for sub in sub_texts:
                    c_id = self._generate_pure_content_chunk_id(doc_hash, node.page_num, heading_path, sub)
                    embed_text = f"{heading_prefix}\n\n{sub}" if heading_prefix else sub

                    meta = dict(base_metadata)
                    raw_chunks.append({
                        "id": c_id, "text": sub, "embed_text": embed_text,
                        "heading_path": heading_path, "page_num": node.page_num,
                        "chunk_type": "paragraph", "parent_chunk_id": parent_chunk_id, "metadata": meta
                    })

            elif isinstance(node, TableNode):
                lines = []
                if node.caption:
                    lines.append(f"Table: {node.caption}\n")
                lines.append("| " + " | ".join(node.headers) + " |")
                lines.append("| " + " | ".join(["---"] * len(node.headers)) + " |")
                for row in node.rows:
                    lines.append("| " + " | ".join(row) + " |")
                markdown_table = "\n".join(lines)
                linearized_table = self._linearize_table(node.headers, node.rows)

                c_id = self._generate_pure_content_chunk_id(doc_hash, node.page_num, heading_path, markdown_table)
                embed_text = f"{heading_prefix}\n\n{linearized_table}" if heading_prefix else linearized_table

                meta = dict(base_metadata)
                meta.update({
                    "caption": node.caption,
                    "markdown_table": markdown_table,
                    "linearized_table": linearized_table,
                    "cell_matrix": [[cell.to_dict() for cell in row] for row in node.cell_matrix]
                })
                raw_chunks.append({
                    "id": c_id, "text": markdown_table, "embed_text": embed_text,
                    "heading_path": heading_path, "page_num": node.page_num,
                    "chunk_type": "table", "parent_chunk_id": parent_chunk_id, "metadata": meta
                })

            elif isinstance(node, ListNode):
                list_str = "\n".join([f"- {i}" for i in node.items])
                sub_texts = self._split_text_with_overlap(list_str, self.max_chunk_tokens, self.overlap_tokens)
                for sub in sub_texts:
                    c_id = self._generate_pure_content_chunk_id(doc_hash, node.page_num, heading_path, sub)
                    embed_text = f"{heading_prefix}\n\n{sub}" if heading_prefix else sub

                    meta = dict(base_metadata)
                    meta.update({"item_count": len(node.items)})
                    raw_chunks.append({
                        "id": c_id, "text": sub, "embed_text": embed_text,
                        "heading_path": heading_path, "page_num": node.page_num,
                        "chunk_type": "list", "parent_chunk_id": parent_chunk_id, "metadata": meta
                    })

            elif isinstance(node, ImageNode):
                c_id = self._generate_pure_content_chunk_id(doc_hash, node.page_num, heading_path, node.alt_text)
                fig_text = f"Figure: {node.alt_text}\nCaption: {node.caption}" if node.caption else f"Figure: {node.alt_text}"
                embed_text = f"{heading_prefix}\n\n{fig_text}" if heading_prefix else fig_text

                meta = dict(base_metadata)
                meta.update({"caption": node.caption, "confidence": node.confidence})
                raw_chunks.append({
                    "id": c_id, "text": fig_text, "embed_text": embed_text,
                    "heading_path": heading_path, "page_num": node.page_num,
                    "chunk_type": "figure", "parent_chunk_id": parent_chunk_id, "metadata": meta
                })

        # Link Graph Relationships (prev_chunk_id, next_chunk_id, parent_chunk_id)
        linked_chunks: List[Chunk] = []
        num_chunks = len(raw_chunks)

        for i, item in enumerate(raw_chunks):
            prev_id = raw_chunks[i-1]["id"] if i > 0 else None
            next_id = raw_chunks[i+1]["id"] if i < num_chunks - 1 else None

            linked_chunks.append(Chunk(
                chunk_id=item["id"],
                text=item["text"],
                embedding_text=item["embed_text"],
                heading_path=item["heading_path"],
                page_num=item["page_num"],
                chunk_type=item["chunk_type"],
                metadata=item["metadata"],
                prev_chunk_id=prev_id,
                next_chunk_id=next_id,
                parent_chunk_id=item["parent_chunk_id"]
            ))

        return linked_chunks


# ==============================================================================
# 6. Decoupled 3-Tier Public API
# ==============================================================================

def parse_docai(raw_data: Any, ignore_headers_footers: bool = True) -> List[ASTNode]:
    """Tier 1: Parses Document AI JSON into an Abstract Syntax Tree (AST)."""
    parser = DocAIParser(raw_data, ignore_headers_footers=ignore_headers_footers)
    return parser.build_ast()

def render_markdown(ast_nodes: List[ASTNode], include_page_comments: bool = True) -> str:
    """Tier 2: Renders AST to GFM Markdown string."""
    renderer = MarkdownRenderer(include_page_comments=include_page_comments)
    return renderer.render(ast_nodes)

def build_chunks(
    ast_nodes: List[ASTNode],
    doc_hash: str = "doc_hash",
    source_doc: str = "document.pdf",
    max_chunk_tokens: int = 400,
    overlap_tokens: int = 50
) -> List[Dict[str, Any]]:
    """Tier 3: Transforms AST into Enterprise RAG Chunks."""
    renderer = ChunkRenderer(max_chunk_tokens=max_chunk_tokens, overlap_tokens=overlap_tokens)
    chunks = renderer.render_chunks(ast_nodes, doc_hash=doc_hash, source_doc=source_doc)
    return [c.to_dict() for c in chunks]

# Backward-Compatibility Helpers
def convert_docai_to_markdown(raw_data: Any, ignore_headers_footers: bool = True, include_page_comments: bool = True) -> str:
    ast_nodes = parse_docai(raw_data, ignore_headers_footers=ignore_headers_footers)
    return render_markdown(ast_nodes, include_page_comments=include_page_comments)

def convert_docai_to_chunks(raw_data: Any, source_doc: str = "document.pdf", max_chunk_tokens: int = 400, overlap_tokens: int = 50) -> List[Dict[str, Any]]:
    parser = DocAIParser(raw_data, ignore_headers_footers=True)
    ast_nodes = parser.build_ast()
    return build_chunks(ast_nodes, doc_hash=parser.doc_hash, source_doc=source_doc, max_chunk_tokens=max_chunk_tokens, overlap_tokens=overlap_tokens)


if __name__ == "__main__":
    import sys
    input_file = sys.argv[1] if len(sys.argv) > 1 else "ttyl.json"
    with open(input_file, "r", encoding="utf-8") as f:
        json_data = json.load(f)

    print("=== DECOUPLED 3-TIER API TEST ===")
    ast = parse_docai(json_data)
    md = render_markdown(ast)
    chunks = build_chunks(ast, source_doc=input_file)

    print(f"AST Nodes count: {len(ast)}")
    print(f"Markdown length: {len(md)} chars")
    print(f"RAG Chunks count: {len(chunks)}")
    print("\n=== SAMPLE CHUNK WITH LINEARIZED TABLE & CONFIDENCE ===")
    print(json.dumps(chunks[1], indent=2))
