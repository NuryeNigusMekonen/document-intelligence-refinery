from pathlib import Path
from typing import Literal

from refinery.config import Settings
from refinery.extraction import ExtractionRouter
from refinery.models import DocumentProfile, ExtractedDocument, ExtractedPage, LanguageHint
from refinery.storage import ArtifactStore


def _profile(cost: Literal["fast_text_sufficient", "needs_layout_model", "needs_vision_model"]) -> DocumentProfile:
    return DocumentProfile(
        doc_id="doc_x",
        doc_name="x.pdf",
        sha256="abc",
        origin_type="native_digital",
        layout_complexity="single_column",
        language_hint=LanguageHint(language="en", confidence=0.7),
        domain_hint="general",
        estimated_extraction_cost=cost,
        page_count=1,
        avg_char_density=0.0003,
        image_area_ratio=0.1,
        whitespace_ratio=0.4,
    )


def _doc() -> ExtractedDocument:
    return ExtractedDocument(doc_id="doc_x", doc_name="x.pdf", pages=[ExtractedPage(page_number=1, width=100, height=100)])


def test_router_escalates_to_layout(tmp_path: Path):
    settings = Settings(workspace_root=tmp_path)
    store = ArtifactStore(settings)
    router = ExtractionRouter(settings, store)

    router.fast.extract = lambda pdf_path, profile: (_doc(), 0.4, 0.0, "low")
    router.layout.extract = lambda pdf_path, profile: (_doc(), 0.8, 0.1, "layout")

    router.extract(Path("dummy.pdf"), _profile("needs_layout_model"))
    ledger = store.read_jsonl(store.ledger_file)
    assert ledger[-1]["strategy_used"] == "layout_lite"
    assert "fast_to_layout" in ledger[-1]["escalations"]
