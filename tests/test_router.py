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
    TextBlock,
)
from refinery.storage import ArtifactStore


def _profile(
    cost: Literal["fast_text_sufficient", "needs_layout_model", "needs_vision_model"],
    *,
    origin: Literal["native_digital", "scanned_image", "mixed", "form_fillable"] = "native_digital",
    layout: Literal["single_column", "multi_column", "table_heavy", "figure_heavy", "mixed"] = "single_column",
    language: str = "en",
    language_confidence: float = 0.7,
    domain: Literal["financial", "legal", "technical", "medical", "general"] = "general",
    domain_confidence: float = 0.7,
) -> DocumentProfile:
    return DocumentProfile(
        doc_id="doc_x",
        doc_name="x.pdf",
        sha256="abc",
        origin_type=origin,
        layout_complexity=layout,
        language_hint=LanguageHint(language=language, confidence=language_confidence),
        domain_hint=domain,
        domain_confidence=domain_confidence,
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
    assert ledger[-1]["strategy_used"] == "strategy_b_layout_docling"
    assert "fast_to_layout" in ledger[-1]["escalations"]
    assert ledger[-1]["policy_context"]["domain_hint"] == "general"
    assert len(ledger[-1]["routing_attempts"]) >= 2
    assert len(ledger[-1]["page_strategy_history"]) == 1
    assert ledger[-1]["page_strategy_history"][0]["final_strategy"] == "strategy_b_layout_docling"


def test_router_scanned_prefers_vision(tmp_path: Path):
    settings = Settings(workspace_root=tmp_path)
    store = ArtifactStore(settings)
    router = ExtractionRouter(settings, store)

    router.fast.extract = lambda pdf_path, profile: (_doc(), 0.95, 0.0, "fast")
    router.layout.extract = lambda pdf_path, profile: (_doc(), 0.9, 0.1, "layout")
    router.vision.extract = lambda pdf_path, profile: (_doc(), 0.7, 0.2, "vision")

    router.extract(Path("dummy.pdf"), _profile("needs_vision_model", origin="scanned_image", layout="figure_heavy"))
    ledger = store.read_jsonl(store.ledger_file)
    assert ledger[-1]["strategy_used"] == "strategy_c_local_ocr"


def test_router_fast_single_column_stays_fast_when_confident(tmp_path: Path):
    settings = Settings(workspace_root=tmp_path)
    store = ArtifactStore(settings)
    router = ExtractionRouter(settings, store)

    router.fast.extract = lambda pdf_path, profile: (_doc(), 0.92, 0.0, "fast_good")
    router.layout.extract = lambda pdf_path, profile: (_doc(), 0.8, 0.1, "layout")

    router.extract(Path("dummy.pdf"), _profile("fast_text_sufficient", origin="native_digital", layout="single_column"))
    ledger = store.read_jsonl(store.ledger_file)
    assert ledger[-1]["strategy_used"] == "strategy_a_fast_text"
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
    assert ledger[-1]["strategy_used"] == "strategy_b_layout_docling"


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


def test_router_does_not_force_language_from_ocr_lang_when_no_text(tmp_path: Path):
    settings = Settings(workspace_root=tmp_path)
    store = ArtifactStore(settings)
    router = ExtractionRouter(settings, store)

    router.vision.extract = lambda pdf_path, profile: (_doc(), 0.84, 0.12, "docling_full_page_ocr local; ocr_lang_used=eng+amh")

    router.extract(
        Path("dummy.pdf"),
        _profile("needs_vision_model", origin="scanned_image", layout="figure_heavy", language="unknown", language_confidence=0.0),
    )
    ledger = store.read_jsonl(store.ledger_file)
    assert ledger[-1]["detected_language"] == "unknown"
    assert ledger[-1]["ocr_lang_used"] == "eng+amh"


def test_router_infers_amharic_from_extracted_script_when_ocr_lang_missing(tmp_path: Path):
    settings = Settings(workspace_root=tmp_path)
    store = ArtifactStore(settings)
    router = ExtractionRouter(settings, store)

    prov = ProvenanceRef(doc_name="x.pdf", ref_type="pdf_bbox", page_number=1, bbox=(0.0, 0.0, 10.0, 10.0), content_hash="pending")
    extracted_with_amharic = ExtractedDocument(
        doc_id="doc_x",
        doc_name="x.pdf",
        pages=[
            ExtractedPage(
                page_number=1,
                width=100,
                height=100,
                blocks=[
                    # Ge'ez script sample
                    TextBlock(
                        text="የኦዲት ሪፖርት ዓመታዊ ግምገማ",
                        bbox=(0.0, 0.0, 10.0, 10.0),
                        reading_order=0,
                        confidence=0.8,
                        provenance=prov,
                    )
                ],
                tables=[],
                figures=[],
            )
        ],
    )

    router.vision.extract = lambda pdf_path, profile: (extracted_with_amharic, 0.84, 0.12, "docling_full_page_ocr local")

    router.extract(
        Path("dummy.pdf"),
        _profile("needs_vision_model", origin="scanned_image", layout="figure_heavy", language="unknown", language_confidence=0.0),
    )
    ledger = store.read_jsonl(store.ledger_file)
    assert ledger[-1]["detected_language"] == "am"


def test_router_prefers_extracted_text_language_over_profile_hint(tmp_path: Path):
    settings = Settings(workspace_root=tmp_path)
    store = ArtifactStore(settings)
    router = ExtractionRouter(settings, store)

    prov = ProvenanceRef(doc_name="x.pdf", ref_type="pdf_bbox", page_number=1, bbox=(0.0, 0.0, 10.0, 10.0), content_hash="pending")
    extracted_with_amharic = ExtractedDocument(
        doc_id="doc_x",
        doc_name="x.pdf",
        pages=[
            ExtractedPage(
                page_number=1,
                width=100,
                height=100,
                blocks=[
                    TextBlock(
                        text="የኦዲት ሪፖርት ዓመታዊ ግምገማ",
                        bbox=(0.0, 0.0, 10.0, 10.0),
                        reading_order=0,
                        confidence=0.8,
                        provenance=prov,
                    )
                ],
                tables=[],
                figures=[],
            )
        ],
    )

    router.vision.extract = lambda pdf_path, profile: (extracted_with_amharic, 0.84, 0.12, "docling_full_page_ocr local")

    router.extract(
        Path("dummy.pdf"),
        _profile("needs_vision_model", origin="scanned_image", layout="figure_heavy", language="en", language_confidence=0.9),
    )
    ledger = store.read_jsonl(store.ledger_file)
    assert ledger[-1]["detected_language"] == "am"


def test_router_uses_registered_layout_engine(tmp_path: Path):
    settings = Settings(workspace_root=tmp_path, layout_engine="custom_layout")
    store = ArtifactStore(settings)
    router = ExtractionRouter(settings, store)

    router.register_layout_engine(
        "custom_layout",
        lambda pdf_path, profile: (_doc(), 0.88, 0.05, "custom layout", "custom_layout"),
    )

    router.extract(Path("dummy.pdf"), _profile("needs_layout_model", origin="mixed", layout="mixed"))
    ledger = store.read_jsonl(store.ledger_file)
    assert ledger[-1]["strategy_used"] == "custom_layout"


def test_vision_uses_registered_vlm_provider(tmp_path: Path):
    settings = Settings(
        workspace_root=tmp_path,
        use_openrouter_vlm=True,
        openrouter_api_key="dummy",
        vlm_provider="custom_vlm",
    )
    store = ArtifactStore(settings)
    router = ExtractionRouter(settings, store)

    unresolved_doc = ExtractedDocument(
        doc_id="doc_x",
        doc_name="x.pdf",
        pages=[ExtractedPage(page_number=1, width=100, height=100, unresolved_needs_vision=True)],
    )
    router.vision._run_local_strategy_c = lambda pdf_path, profile: (unresolved_doc, 0.40, 0.0, "local low")
    router.vision.register_vlm_provider("custom_vlm", lambda pdf_path, profile: (_doc(), 0.91, 0.15, "custom vlm"))

    result_doc, result_conf, _result_cost, result_notes = router.vision.extract(
        Path("dummy.pdf"),
        _profile("needs_vision_model", origin="scanned_image", layout="figure_heavy"),
    )
    assert result_doc.doc_id == "doc_x"
    assert result_conf == 0.91
    assert result_notes == "custom vlm"


def test_router_blocks_vision_escalation_when_budget_headroom_low(tmp_path: Path):
    settings = Settings(
        workspace_root=tmp_path,
        enable_vision=True,
        max_cost_per_doc=0.12,
        router_budget_min_vision_headroom=0.05,
    )
    store = ArtifactStore(settings)
    router = ExtractionRouter(settings, store)

    router.layout.extract = lambda pdf_path, profile: (_doc(), 0.68, 0.10, "layout_low")
    router.vision.extract = lambda pdf_path, profile: (_doc(), 0.90, 0.30, "vision")

    router.extract(Path("dummy.pdf"), _profile("needs_layout_model", origin="mixed", layout="mixed"))
    ledger = store.read_jsonl(store.ledger_file)
    assert "layout_to_vision_blocked" in ledger[-1]["escalations"]
    assert ledger[-1]["policy_context"]["vision_allowed"] is False
    assert ledger[-1]["policy_context"]["vision_policy_reason"] == "insufficient_budget_headroom"


def test_router_domain_risk_boost_can_trigger_vision_escalation(tmp_path: Path):
    settings = Settings(
        workspace_root=tmp_path,
        enable_vision=True,
        max_cost_per_doc=1.0,
        router_budget_min_vision_headroom=0.05,
        router_domain_risk_boost_high=0.10,
    )
    store = ArtifactStore(settings)
    router = ExtractionRouter(settings, store)

    router.layout.extract = lambda pdf_path, profile: (_doc(), 0.76, 0.10, "layout_mid")
    router.vision.extract = lambda pdf_path, profile: (_doc(), 0.89, 0.12, "vision_good")

    router.extract(
        Path("dummy.pdf"),
        _profile("needs_layout_model", origin="mixed", layout="mixed", domain="legal", domain_confidence=0.95),
    )
    ledger = store.read_jsonl(store.ledger_file)
    assert "layout_to_vision" in ledger[-1]["escalations"]
    assert ledger[-1]["policy_context"]["domain_risk_boost"] == 0.10
    assert ledger[-1]["strategy_used"] == "vision"
