from __future__ import annotations

from abc import ABC, abstractmethod
import importlib.util
import math
from pathlib import Path

import fitz

from ..models import DocumentProfile, ExtractedDocument
from ..models import ExtractedPage, ProvenanceRef, TextBlock


class LayoutAdapter(ABC):
    name: str = "layout-adapter"

    @property
    @abstractmethod
    def available(self) -> bool:
        raise NotImplementedError

    @abstractmethod
    def extract(self, pdf_path: Path, profile: DocumentProfile) -> ExtractedDocument:
        raise NotImplementedError


class DoclingAdapter(LayoutAdapter):
    name = "docling"

    @property
    def available(self) -> bool:
        return importlib.util.find_spec("docling") is not None

    def extract(self, pdf_path: Path, profile: DocumentProfile) -> ExtractedDocument:
        if not self.available:
            raise RuntimeError("Docling is not installed")

        try:
            from docling.document_converter import DocumentConverter
        except Exception as exc:
            raise RuntimeError("Docling runtime import failed") from exc

        converter = DocumentConverter()
        result = converter.convert(str(pdf_path))
        document = getattr(result, "document", None)
        if document is None:
            raise RuntimeError("Docling conversion returned no document object")

        markdown_text = ""
        if hasattr(document, "export_to_markdown"):
            markdown_text = str(document.export_to_markdown() or "").strip()
        if not markdown_text:
            fallback_markdown = getattr(result, "markdown", "")
            markdown_text = str(fallback_markdown or "").strip()
        if not markdown_text:
            raise RuntimeError("Docling conversion produced empty markdown")

        paragraphs = [p.strip() for p in markdown_text.split("\n\n") if p.strip()]
        if not paragraphs:
            raise RuntimeError("Docling produced no paragraph content")

        with fitz.open(pdf_path) as doc:
            page_count = max(1, doc.page_count)
            page_dims = [(float(doc.load_page(i).rect.width), float(doc.load_page(i).rect.height)) for i in range(page_count)]

        chunks_per_page = max(1, math.ceil(len(paragraphs) / page_count))
        pages: list[ExtractedPage] = []
        reading_order = 0

        for page_number in range(1, page_count + 1):
            width, height = page_dims[page_number - 1]
            start = (page_number - 1) * chunks_per_page
            end = min(len(paragraphs), start + chunks_per_page)
            page_blocks: list[TextBlock] = []
            for local_idx, text in enumerate(paragraphs[start:end]):
                y0 = min(height - 1.0, float(local_idx) * 32.0)
                y1 = min(height, y0 + 28.0)
                bbox = (0.0, y0, width, y1)
                page_blocks.append(
                    TextBlock(
                        text=text,
                        bbox=bbox,
                        reading_order=reading_order,
                        confidence=0.86,
                        provenance=ProvenanceRef(
                            doc_name=profile.doc_name,
                            ref_type="pdf_bbox",
                            page_number=page_number,
                            bbox=bbox,
                            content_hash="pending",
                        ),
                    )
                )
                reading_order += 1

            pages.append(
                ExtractedPage(
                    page_number=page_number,
                    width=width,
                    height=height,
                    blocks=page_blocks,
                    tables=[],
                    figures=[],
                )
            )

        return ExtractedDocument(doc_id=profile.doc_id, doc_name=profile.doc_name, pages=pages)


from .docx_adapter import DocxAdapter
from .excel_adapter import ExcelAdapter
from .image_adapter import ImageAdapter
from .markdown_adapter import MarkdownAdapter

__all__ = [
    "LayoutAdapter",
    "DoclingAdapter",
    "DocxAdapter",
    "MarkdownAdapter",
    "ImageAdapter",
    "ExcelAdapter",
]
