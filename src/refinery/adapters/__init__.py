from __future__ import annotations

from abc import ABC, abstractmethod
import importlib.util
from pathlib import Path

from ..models import DocumentProfile, ExtractedDocument


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
        raise RuntimeError("Docling adapter interface ready, but runtime extraction is not enabled in this local build")


class MineruAdapter(LayoutAdapter):
    name = "mineru"

    @property
    def available(self) -> bool:
        return importlib.util.find_spec("mineru") is not None

    def extract(self, pdf_path: Path, profile: DocumentProfile) -> ExtractedDocument:
        raise RuntimeError("Mineru adapter interface ready, but runtime extraction is not enabled in this local build")


from .docx_adapter import DocxAdapter
from .excel_adapter import ExcelAdapter
from .image_adapter import ImageAdapter
from .markdown_adapter import MarkdownAdapter

__all__ = [
    "LayoutAdapter",
    "DoclingAdapter",
    "MineruAdapter",
    "DocxAdapter",
    "MarkdownAdapter",
    "ImageAdapter",
    "ExcelAdapter",
]
