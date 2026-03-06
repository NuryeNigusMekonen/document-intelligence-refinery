from pathlib import Path

from refinery.models import LogicalDocumentUnit, ProvenanceRef
from refinery.pageindex import PageIndexBuilder, SENTENCE_RE, pageindex_navigate
from refinery.storage import ArtifactStore
from refinery.config import Settings


def _pref(page: int) -> ProvenanceRef:
    return ProvenanceRef(doc_name="x.pdf", ref_type="pdf_bbox", page_number=page, bbox=(0, 0, 10, 10), content_hash="h")


def _ldu(ldu_id: str, chunk_type: str, content: str, section_path: list[str], page: int) -> LogicalDocumentUnit:
    return LogicalDocumentUnit(
        ldu_id=ldu_id,
        chunk_type=chunk_type,
        content=content,
        token_count=max(1, len(content.split())),
        parent_section_path=section_path,
        parent_section=" > ".join(section_path) if section_path else None,
        page_refs=[_pref(page)],
        content_hash=f"h-{ldu_id}",
    )


def test_pageindex_builds_hierarchy_and_data_types(tmp_path: Path):
    settings = Settings(workspace_root=tmp_path)
    store = ArtifactStore(settings)
    builder = PageIndexBuilder(store)

    ldus = [
        _ldu("p1", "paragraph", "Capital expenditure projections for Q3 are increasing.", ["Financials", "CapEx"], 4),
        _ldu("t1", "table", "Year | CapEx\n2026 | 120", ["Financials", "CapEx"], 5),
        _ldu("f1", "figure", "[FIGURE] Plant investment trend.", ["Financials", "CapEx"], 5),
        _ldu("l1", "list", "1. Build\n2. Expand", ["Operations"], 8),
    ]

    idx = builder.build("doc1", "x.pdf", ldus)

    assert len(idx.root_sections) == 2
    financials = next(s for s in idx.root_sections if s.title == "Financials")
    capex = next(c for c in financials.child_sections if c.title == "CapEx")
    assert capex.page_start == 4 and capex.page_end == 5
    assert "tables" in capex.data_types_present
    assert "figures" in capex.data_types_present
    sentence_count = len([s for s in SENTENCE_RE.findall(capex.summary) if s.strip()])
    assert capex.summary.strip()
    assert 2 <= sentence_count <= 3


def test_pageindex_navigate_finds_nested_capex_section(tmp_path: Path):
    settings = Settings(workspace_root=tmp_path)
    store = ArtifactStore(settings)
    builder = PageIndexBuilder(store)

    ldus = [
        _ldu("p1", "paragraph", "Capital expenditure projections for Q3 are increasing.", ["Financials", "CapEx"], 4),
        _ldu("p2", "paragraph", "Revenue outlook remains stable.", ["Financials", "Revenue"], 3),
    ]
    idx = builder.build("doc2", "x.pdf", ldus)
    hits = pageindex_navigate(idx, "capital expenditure projections for Q3", top_k=3)
    assert any(h.title == "CapEx" for h in hits)


def test_pageindex_fallback_when_ollama_disabled(tmp_path: Path):
    settings = Settings(workspace_root=tmp_path, use_ollama_summaries=False)
    store = ArtifactStore(settings)
    builder = PageIndexBuilder(store)

    ldus = [_ldu("p1", "paragraph", "Capital expenditure projections for Q3 are increasing significantly.", ["Financials", "CapEx"], 4)]
    idx = builder.build("doc3", "x.pdf", ldus)
    capex = next(c for c in next(s for s in idx.root_sections if s.title == "Financials").child_sections if c.title == "CapEx")
    sentence_count = len([s for s in SENTENCE_RE.findall(capex.summary) if s.strip()])
    assert capex.summary.strip()
    assert 2 <= sentence_count <= 3


def test_pageindex_navigate_defaults_to_top3(tmp_path: Path):
    settings = Settings(workspace_root=tmp_path)
    store = ArtifactStore(settings)
    builder = PageIndexBuilder(store)

    ldus = [
        _ldu("p1", "paragraph", "capital expenditure projections", ["Financials", "CapEx"], 1),
        _ldu("p2", "paragraph", "revenue outlook", ["Financials", "Revenue"], 2),
        _ldu("p3", "paragraph", "operations summary", ["Operations"], 3),
        _ldu("p4", "paragraph", "risk and compliance", ["Risk"], 4),
    ]
    idx = builder.build("doc4", "x.pdf", ldus)
    hits = pageindex_navigate(idx, "financial projections")
    assert len(hits) <= 3
