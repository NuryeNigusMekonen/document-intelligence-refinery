import pytest
from pathlib import Path
from typing import Literal
from refinery.models import LogicalDocumentUnit, ProvenanceRef
from refinery.pageindex import PageIndexBuilder
from refinery.storage import ArtifactStore
from refinery.config import Settings

def _pref(page: int) -> ProvenanceRef:
    return ProvenanceRef(doc_name="x.pdf", ref_type="pdf_bbox", page_number=page, bbox=(0, 0, 10, 10), content_hash="h")

def _ldu(
    ldu_id: str,
    chunk_type: Literal["paragraph", "table", "figure", "list", "section_summary", "fact"],
    content: str,
    section_path: list[str],
    page: int,
) -> LogicalDocumentUnit:
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

def test_fragmented_titles_are_merged(tmp_path: Path):
    settings = Settings(workspace_root=tmp_path)
    store = ArtifactStore(settings)
    builder = PageIndexBuilder(store)
    ldus = [
        _ldu("h1", "paragraph", "Deferred", [], 1),
        _ldu("h2", "paragraph", "tax liability", [], 1),
        _ldu("p1", "paragraph", "This section discusses deferred tax liabilities.", ["Deferred tax liability"], 1),
    ]
    idx = builder.build("doc_frag", "x.pdf", ldus)
    titles = [s.title for s in idx.root_sections]
    assert "Deferred tax liability" in titles

def test_section_titles_not_from_table_cells(tmp_path: Path):
    settings = Settings(workspace_root=tmp_path)
    store = ArtifactStore(settings)
    builder = PageIndexBuilder(store)
    ldus = [
        _ldu("t1", "table", "Amortization | Value\n2026 | 100", ["Table Section"], 2),
        _ldu("p1", "paragraph", "Amortization", [], 2),
        _ldu("p2", "paragraph", "This section covers amortization.", ["Amortization"], 2),
    ]
    idx = builder.build("doc_table", "x.pdf", ldus)
    titles = [s.title for s in idx.root_sections]
    assert "Amortization" in titles
    assert all(not t.startswith("Amortization | Value") for t in titles)

def test_summaries_are_full_sentences(tmp_path: Path):
    settings = Settings(workspace_root=tmp_path)
    store = ArtifactStore(settings)
    builder = PageIndexBuilder(store)
    ldus = [
        _ldu("h1", "paragraph", "Revenue", [], 3),
        _ldu("p1", "paragraph", "Revenue for the year increased by 10% due to higher sales volume.", ["Revenue"], 3),
    ]
    idx = builder.build("doc_sum", "x.pdf", ldus)
    summary = idx.root_sections[0].summary
    assert summary.count(".") >= 2
    assert len(summary.split()) > 8

def test_data_types_present_includes_tables(tmp_path: Path):
    settings = Settings(workspace_root=tmp_path)
    store = ArtifactStore(settings)
    builder = PageIndexBuilder(store)
    ldus = [
        _ldu("h1", "paragraph", "Financials", [], 4),
        _ldu("t1", "table", "Year | CapEx\n2026 | 120", ["Financials"], 5),
    ]
    idx = builder.build("doc_types", "x.pdf", ldus)
    section = idx.root_sections[0]
    assert "tables" in section.data_types_present

def test_key_entities_non_empty_for_financial_sections(tmp_path: Path):
    settings = Settings(workspace_root=tmp_path)
    store = ArtifactStore(settings)
    builder = PageIndexBuilder(store)
    ldus = [
        _ldu("h1", "paragraph", "Financials", [], 4),
        _ldu("p1", "paragraph", "The company reported a net income of $1,000,000 in 2026.", ["Financials"], 4),
    ]
    idx = builder.build("doc_entities", "x.pdf", ldus)
    section = idx.root_sections[0]
    assert section.key_entities

def test_page_ranges_extend_beyond_page_1(tmp_path: Path):
    settings = Settings(workspace_root=tmp_path)
    store = ArtifactStore(settings)
    builder = PageIndexBuilder(store)
    ldus = [
        _ldu("h1", "paragraph", "Introduction", [], 1),
        _ldu("p1", "paragraph", "Intro text.", ["Introduction"], 1),
        _ldu("h2", "paragraph", "Financials", [], 4),
        _ldu("p2", "paragraph", "Financials text.", ["Financials"], 4),
    ]
    idx = builder.build("doc_range", "x.pdf", ldus)
    intro = next(s for s in idx.root_sections if s.title == "Introduction")
    financials = next(s for s in idx.root_sections if s.title == "Financials")
    assert intro.page_end == 1
    assert financials.page_start == 4
    assert financials.page_end == 4
