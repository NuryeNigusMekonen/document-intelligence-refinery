from pathlib import Path
import re

import pytest
import fitz
from PIL import Image

from refinery.config import Settings
from refinery.extraction import VisionExtractor
from refinery.models import DocumentProfile, LanguageHint


def _word(idx: int, text: str, x: float, y: float, w: float = 40.0, h: float = 12.0, conf: float = 90.0) -> dict:
    return {
        "index": idx,
        "text": text,
        "x": x,
        "y": y,
        "w": w,
        "h": h,
        "conf": conf,
        "x_center": x + w / 2.0,
        "y_center": y + h / 2.0,
    }


def test_reconstruct_table_from_synthetic_ocr_words(tmp_path: Path):
    settings = Settings(workspace_root=tmp_path)
    extractor = VisionExtractor(settings)

    words = []
    idx = 0
    rows = [
        ["Col A", "Col B", "Col C"],
        ["R1A", "R1B", "R1C"],
        ["R2A", "R2B", "R2C"],
        ["R3A", "R3B", "R3C"],
    ]
    y_positions = [10.0, 35.0, 60.0, 85.0]
    x_positions = [10.0, 110.0, 210.0]

    for r, y in enumerate(y_positions):
        for c, x in enumerate(x_positions):
            words.append(_word(idx, rows[r][c], x, y))
            idx += 1

    table, used = extractor._reconstruct_table_from_words(
        words,
        page_number=1,
        doc_name="synthetic.pdf",
        page_width=300.0,
        page_height=200.0,
    )

    assert table is not None
    assert len(used) == 12
    assert len(table.headers) == 3
    assert len(table.rows) == 3
    assert all(len(r) == 3 for r in table.rows)

    x0, y0, x1, y1 = table.bbox
    assert x0 <= 10.0
    assert y0 <= 10.0
    assert x1 >= 250.0
    assert y1 >= 97.0


def test_pixel_bbox_to_pdf_bbox_conversion(tmp_path: Path):
    settings = Settings(workspace_root=tmp_path)
    extractor = VisionExtractor(settings)
    bbox = extractor.pixel_bbox_to_pdf_bbox((200.0, 300.0, 600.0, 900.0), 1200.0, 1800.0, 600.0, 900.0)
    assert bbox == (100.0, 150.0, 300.0, 450.0)


def test_ethiopic_chars_preserved_with_table_reconstruction(tmp_path: Path):
    pytesseract = pytest.importorskip("pytesseract")
    data_pdf = Path("data/2013-E.C-Assigned-regular-budget-and-expense.pdf")
    if not data_pdf.exists():
        pytest.skip("regression pdf not available")

    settings = Settings(workspace_root=tmp_path, ocr_enabled=True, ocr_engine="tesseract", ocr_lang_fallback="eng+amh")
    extractor = VisionExtractor(settings)
    profile = DocumentProfile(
        doc_id="doc_reg",
        doc_name=data_pdf.name,
        sha256="dummy",
        origin_type="scanned_image",
        layout_complexity="figure_heavy",
        language_hint=LanguageHint(language="unknown", confidence=0.0),
        domain_hint="general",
        estimated_extraction_cost="needs_vision_model",
        page_count=3,
        avg_char_density=0.0,
        image_area_ratio=1.0,
        whitespace_ratio=1.0,
    )

    baseline_chars = 0
    with fitz.open(data_pdf) as doc:
        for pidx in range(1, doc.page_count + 1):
            page = doc.load_page(pidx - 1)
            pix = page.get_pixmap(dpi=200)
            img = Image.frombytes("RGB", (pix.width, pix.height), pix.samples)
            data = pytesseract.image_to_data(img, lang="eng+amh", output_type=pytesseract.Output.DICT)
            for txt in data.get("text", []) or []:
                baseline_chars += len(re.findall(r"[\u1200-\u137F]", str(txt or "")))

    assert baseline_chars > 200

    result = extractor._run_tesseract_ocr(data_pdf, profile)
    assert result is not None
    extracted = result[0]
    post_chars = 0
    for page in extracted.pages:
        for block in page.blocks:
            post_chars += len(re.findall(r"[\u1200-\u137F]", block.text or ""))

    assert post_chars >= int(0.7 * baseline_chars)
