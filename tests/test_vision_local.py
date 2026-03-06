from pathlib import Path

from refinery.config import Settings
from refinery.extraction import VisionExtractor
from refinery.models import DocumentProfile, ExtractedDocument, ExtractedPage, LanguageHint


def _profile() -> DocumentProfile:
    return DocumentProfile(
        doc_id="doc_v",
        doc_name="v.pdf",
        sha256="abc",
        origin_type="scanned_image",
        layout_complexity="figure_heavy",
        language_hint=LanguageHint(language="en", confidence=0.9),
        domain_hint="general",
        estimated_extraction_cost="needs_vision_model",
        page_count=1,
        avg_char_density=0.00001,
        image_area_ratio=0.8,
        whitespace_ratio=0.95,
    )


def test_default_route_uses_local_ocr(monkeypatch, tmp_path: Path):
    settings = Settings(workspace_root=tmp_path, enable_vision=True)
    extractor = VisionExtractor(settings)

    monkeypatch.setattr(
        extractor,
        "_run_local_strategy_c",
        lambda pdf_path, profile: (
            ExtractedDocument(
                doc_id=profile.doc_id,
                doc_name=profile.doc_name,
                pages=[
                    ExtractedPage(
                        page_number=1,
                        width=100,
                        height=100,
                        blocks=[],
                        tables=[],
                        figures=[],
                        unresolved_needs_vision=False,
                    )
                ],
            ),
            0.76,
            0.0,
            "ocr_engine=tesseract ocr_lang_used=eng",
        ),
    )
    extracted, confidence, cost, notes = extractor.extract(Path("dummy.pdf"), _profile())
    assert extracted.doc_id == "doc_v"
    assert confidence == 0.76
    assert cost == 0.0
    assert notes.startswith("ocr_engine=tesseract")


def test_local_route_is_returned_even_when_unresolved(monkeypatch, tmp_path: Path):
    settings = Settings(workspace_root=tmp_path, enable_vision=True)
    extractor = VisionExtractor(settings)

    monkeypatch.setattr(
        extractor,
        "_run_local_strategy_c",
        lambda pdf_path, profile: (
            ExtractedDocument(
                doc_id=profile.doc_id,
                doc_name=profile.doc_name,
                pages=[
                    ExtractedPage(
                        page_number=1,
                        width=100,
                        height=100,
                        blocks=[],
                        tables=[],
                        figures=[],
                        unresolved_needs_vision=True,
                    )
                ],
            ),
            0.3,
            0.0,
            "local_ocr_unavailable",
        ),
    )

    extracted, confidence, cost, notes = extractor.extract(Path("dummy.pdf"), _profile())
    assert extracted.doc_id == "doc_v"
    assert confidence == 0.3
    assert cost == 0.0
    assert notes == "local_ocr_unavailable"
