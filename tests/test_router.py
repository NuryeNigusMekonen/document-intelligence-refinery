from pathlib import Path
from typing import Literal

from refinery.config import Settings
from refinery.extraction import ExtractionRouter
from refinery.models import (
    DocumentProfile,
    ExtractedDocument,
    ExtractedPage,
    FigureObject,
    LanguageHint,
    ProvenanceRef,
    TableObject,
)
from refinery.storage import ArtifactStore


def _profile(
    cost: Literal["fast_text_sufficient", "needs_layout_model", "needs_vision_model"],
    *,
    origin: Literal["native_digital", "scanned_image", "mixed", "form_fillable"] = "native_digital",
    layout: Literal["single_column", "multi_column", "table_heavy", "figure_heavy", "mixed"] = "single_column",
) -> DocumentProfile:
    return DocumentProfile(
        doc_id="doc_x",
        doc_name="x.pdf",
        sha256="abc",
        origin_type=origin,
        layout_complexity=layout,
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

    router.extract(Path("dummy.pdf"), _profile("fast_text_sufficient"))
    ledger = store.read_jsonl(store.ledger_file)
    assert ledger[-1]["strategy_used"] == "layout_lite"
    assert "fast_to_layout" in ledger[-1]["escalations"]


def test_router_scanned_prefers_vision(tmp_path: Path):
    settings = Settings(workspace_root=tmp_path)
    store = ArtifactStore(settings)
    router = ExtractionRouter(settings, store)

    router.fast.extract = lambda pdf_path, profile: (_doc(), 0.95, 0.0, "fast")
    router.layout.extract = lambda pdf_path, profile: (_doc(), 0.9, 0.1, "layout")
    router.vision.extract = lambda pdf_path, profile: (_doc(), 0.7, 0.2, "vision")

    router.extract(Path("dummy.pdf"), _profile("needs_vision_model", origin="scanned_image", layout="figure_heavy"))
    ledger = store.read_jsonl(store.ledger_file)
    assert ledger[-1]["strategy_used"] in {"vision", "vision_disabled"}


def test_router_fast_single_column_stays_fast_when_confident(tmp_path: Path):
    settings = Settings(workspace_root=tmp_path)
    store = ArtifactStore(settings)
    router = ExtractionRouter(settings, store)

    router.fast.extract = lambda pdf_path, profile: (_doc(), 0.92, 0.0, "fast_good")
    router.layout.extract = lambda pdf_path, profile: (_doc(), 0.8, 0.1, "layout")

    router.extract(Path("dummy.pdf"), _profile("fast_text_sufficient", origin="native_digital", layout="single_column"))
    ledger = store.read_jsonl(store.ledger_file)
    assert ledger[-1]["strategy_used"] == "fast_text"
    assert "fast_to_layout" not in ledger[-1]["escalations"]


def test_router_figure_heavy_uses_layout_not_fast(tmp_path: Path):
    settings = Settings(workspace_root=tmp_path)
    store = ArtifactStore(settings)
    router = ExtractionRouter(settings, store)

    def _should_not_run_fast(*_args, **_kwargs):
        raise AssertionError("fast extractor should not be used for non-single-column layout")

    router.fast.extract = _should_not_run_fast
    router.layout.extract = lambda pdf_path, profile: (_doc(), 0.83, 0.1, "layout")

    router.extract(Path("dummy.pdf"), _profile("needs_layout_model", origin="native_digital", layout="figure_heavy"))
    ledger = store.read_jsonl(store.ledger_file)
    assert ledger[-1]["strategy_used"] == "layout_lite"


def test_router_docling_is_enriched_with_layout_structures(tmp_path: Path, monkeypatch):
    settings = Settings(workspace_root=tmp_path)
    store = ArtifactStore(settings)
    router = ExtractionRouter(settings, store)

    class _DoclingStub:
        available = True

        def extract(self, pdf_path, profile):
            return ExtractedDocument(
                doc_id=profile.doc_id,
                doc_name=profile.doc_name,
                pages=[ExtractedPage(page_number=1, width=100, height=100, blocks=[], tables=[], figures=[])],
            )

    prov = ProvenanceRef(doc_name="x.pdf", ref_type="pdf_bbox", page_number=1, bbox=(0.0, 0.0, 10.0, 10.0), content_hash="pending")
    monkeypatch.setattr(router, "docling_adapter", _DoclingStub())
    router.layout.extract = lambda pdf_path, profile: (
        ExtractedDocument(
            doc_id=profile.doc_id,
            doc_name=profile.doc_name,
            pages=[
                ExtractedPage(
                    page_number=1,
                    width=100,
                    height=100,
                    blocks=[],
                    tables=[TableObject(bbox=(0.0, 0.0, 10.0, 10.0), headers=["h"], rows=[["r"]], reading_order=1, confidence=0.8, provenance=prov)],
                    figures=[FigureObject(bbox=(0.0, 0.0, 10.0, 10.0), caption="Figure 1", reading_order=2, confidence=0.8, provenance=prov)],
                )
            ],
        ),
        0.82,
        0.1,
        "layout",
    )

    extracted = router.extract(Path("dummy.pdf"), _profile("needs_layout_model", origin="mixed", layout="mixed"))
    assert len(extracted.pages[0].tables) == 1
    assert len(extracted.pages[0].figures) == 1
