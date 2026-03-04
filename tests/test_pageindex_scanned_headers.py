from pathlib import Path

from refinery.config import Settings
from refinery.models import LogicalDocumentUnit, ProvenanceRef
from refinery.pageindex import PageIndexBuilder
from refinery.storage import ArtifactStore


def _pref(page: int, bbox: tuple[float, float, float, float]) -> ProvenanceRef:
    return ProvenanceRef(doc_name="scan.pdf", ref_type="pdf_bbox", page_number=page, bbox=bbox, content_hash="h")


def _ldu(ldu_id: str, content: str, page: int, bbox: tuple[float, float, float, float]) -> LogicalDocumentUnit:
    return LogicalDocumentUnit(
        ldu_id=ldu_id,
        chunk_type="paragraph",
        content=content,
        token_count=max(1, len(content.split())),
        parent_section=None,
        parent_section_path=[],
        page_refs=[_pref(page, bbox)],
        content_hash=f"h-{ldu_id}",
    )


def test_pageindex_scanned_headers_create_multiple_sections(tmp_path: Path):
    settings = Settings(workspace_root=tmp_path)
    store = ArtifactStore(settings)
    builder = PageIndexBuilder(store)

    ldus = [
        _ldu("h1", "EXECUTIVE SUMMARY", 1, (30.0, 24.0, 320.0, 44.0)),
        _ldu("p1", "This report summarizes operations and outcomes for the quarter.", 1, (30.0, 120.0, 540.0, 170.0)),
        _ldu("h2", "FINANCIAL PERFORMANCE", 2, (30.0, 20.0, 360.0, 42.0)),
        _ldu("p2", "Revenue increased while operational expenses remained controlled.", 2, (30.0, 130.0, 560.0, 182.0)),
    ]

    pageindex = builder.build("doc-scan-1", "scan.pdf", ldus)
    titles = [s.title for s in pageindex.root_sections]

    assert len(pageindex.root_sections) >= 2
    assert any("EXECUTIVE SUMMARY" in t for t in titles)
    assert any("FINANCIAL PERFORMANCE" in t for t in titles)


def test_pageindex_scanned_headers_fallback_to_root_when_no_headers(tmp_path: Path):
    settings = Settings(workspace_root=tmp_path)
    store = ArtifactStore(settings)
    builder = PageIndexBuilder(store)

    ldus = [
        _ldu("p1", "the the the and 11 22 33 noise tokens.", 1, (30.0, 120.0, 540.0, 170.0)),
        _ldu("p2", "another line with punctuation and no real heading.", 2, (30.0, 130.0, 560.0, 182.0)),
    ]

    pageindex = builder.build("doc-scan-2", "scan.pdf", ldus)

    assert len(pageindex.root_sections) == 1
    assert pageindex.root_sections[0].title == "Document Root"
