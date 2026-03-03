from __future__ import annotations

import mimetypes
from pathlib import Path

from .adapters import DocxAdapter, ExcelAdapter, ImageAdapter, MarkdownAdapter
from .config import Settings
from .models import ExtractedDocument


class FileRouter:
    def __init__(self, settings: Settings):
        self.settings = settings
        self.docx = DocxAdapter()
        self.md = MarkdownAdapter()
        self.image = ImageAdapter(settings)
        self.excel = ExcelAdapter()

    def detect_type(self, path: Path) -> str:
        ext = path.suffix.lower()
        if ext == ".pdf":
            return "pdf"
        if ext == ".docx":
            return "docx"
        if ext == ".md":
            return "markdown"
        if ext in {".png", ".jpg", ".jpeg"}:
            return "image"
        if ext == ".xlsx":
            return "excel"
        mime, _ = mimetypes.guess_type(str(path))
        if mime == "application/pdf":
            return "pdf"
        if mime == "application/vnd.openxmlformats-officedocument.wordprocessingml.document":
            return "docx"
        if mime in {"text/markdown", "text/plain"} and ext == ".md":
            return "markdown"
        if mime and mime.startswith("image/"):
            return "image"
        if mime == "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet":
            return "excel"
        return "unsupported"

    def extract_non_pdf(self, path: Path, doc_id: str, profile_language: str = "unknown", profile_conf: float = 0.0) -> tuple[ExtractedDocument, float, str]:
        ftype = self.detect_type(path)
        if ftype == "docx":
            return self.docx.extract(path, doc_id), 0.90, self.docx.name
        if ftype == "markdown":
            return self.md.extract(path, doc_id), 0.92, self.md.name
        if ftype == "image":
            extracted, conf, strategy = self.image.extract(path, doc_id, profile_language=profile_language, profile_conf=profile_conf)
            return extracted, conf, strategy
        if ftype == "excel":
            return self.excel.extract(path, doc_id), 0.95, self.excel.name
        raise ValueError(f"Unsupported file type for {path}")
