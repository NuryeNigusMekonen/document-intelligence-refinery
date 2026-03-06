from __future__ import annotations

from pathlib import Path

from ..config import Settings
from ..lang_detect import select_ocr_lang
from ..models import ExtractedDocument, ExtractedPage, ProvenanceRef, TextBlock


class ImageAdapter:
    name = "image_adapter"

    def __init__(self, settings: Settings):
        self.settings = settings

    def extract(self, file_path: Path, doc_id: str, profile_language: str = "unknown", profile_conf: float = 0.0) -> tuple[ExtractedDocument, float, str]:
        if not self.settings.ocr_enabled or self.settings.ocr_engine == "off":
            prov = ProvenanceRef(doc_name=file_path.name, ref_type="image_bbox", bbox=(0.0, 0.0, 1000.0, 1000.0), content_hash="pending")
            page = ExtractedPage(
                page_number=1,
                width=1000.0,
                height=1000.0,
                blocks=[TextBlock(text="[UNRESOLVED_IMAGE_TEXT] OCR disabled", bbox=(0.0, 0.0, 1000.0, 1000.0), reading_order=0, confidence=0.2, provenance=prov)],
                tables=[],
                figures=[],
                unresolved_needs_vision=True,
            )
            return ExtractedDocument(doc_id=doc_id, doc_name=file_path.name, pages=[page]), 0.2, "ocr_disabled"

        ocr_lang = select_ocr_lang(
            profile_language,
            profile_conf,
            ocr_amharic_enabled=self.settings.ocr_amharic_enabled,
            ocr_lang_default=self.settings.ocr_lang_default,
            ocr_lang_fallback=self.settings.ocr_lang_fallback,
        )
        try:
            from PIL import Image
            import pytesseract

            img = Image.open(file_path)
            width, height = img.size
            note = f"tesseract_ocr ocr_lang_used={ocr_lang}"
            try:
                data = pytesseract.image_to_data(img, lang=ocr_lang, output_type=pytesseract.Output.DICT)
            except Exception:
                data = pytesseract.image_to_data(img, lang=self.settings.ocr_lang_default, output_type=pytesseract.Output.DICT)
                note = f"tesseract_ocr ocr_lang_used={self.settings.ocr_lang_default} fallback_from={ocr_lang}"
            blocks: list[TextBlock] = []
            conf_scores: list[float] = []
            non_empty = 0
            for i in range(len(data.get("text", []))):
                txt = (data["text"][i] or "").strip()
                if not txt:
                    continue
                x = float(data["left"][i])
                y = float(data["top"][i])
                w = float(data["width"][i])
                h = float(data["height"][i])
                c = float(data.get("conf", [0])[i]) if str(data.get("conf", [0])[i]).strip() not in {"", "-1"} else 0.0
                conf_scores.append(max(0.0, min(100.0, c)))
                non_empty += 1
                prov = ProvenanceRef(doc_name=file_path.name, ref_type="image_bbox", bbox=(x, y, x + w, y + h), content_hash="pending")
                blocks.append(TextBlock(text=txt, bbox=(x, y, x + w, y + h), reading_order=i, confidence=max(0.05, min(1.0, c / 100.0)), provenance=prov))
            mean_conf = (sum(conf_scores) / max(len(conf_scores), 1)) / 100.0
            coverage = min(1.0, non_empty / 100.0)
            page_score = max(0.1, min(0.95, 0.5 * mean_conf + 0.5 * coverage))
            page = ExtractedPage(page_number=1, width=float(width), height=float(height), blocks=blocks, tables=[], figures=[])
            return ExtractedDocument(doc_id=doc_id, doc_name=file_path.name, pages=[page]), page_score, note
        except Exception:
            low_note = "vision_disabled_low_confidence ocr_unavailable"
            fallback_text = "[UNRESOLVED_IMAGE_TEXT] OCR unavailable"
            prov = ProvenanceRef(doc_name=file_path.name, ref_type="image_bbox", bbox=(0.0, 0.0, 1000.0, 1000.0), content_hash="pending")
            page = ExtractedPage(
                page_number=1,
                width=1000.0,
                height=1000.0,
                blocks=[TextBlock(text=fallback_text, bbox=(0.0, 0.0, 1000.0, 1000.0), reading_order=0, confidence=0.2, provenance=prov)],
                tables=[],
                figures=[],
                unresolved_needs_vision=True,
            )
            return ExtractedDocument(doc_id=doc_id, doc_name=file_path.name, pages=[page]), 0.2, low_note
