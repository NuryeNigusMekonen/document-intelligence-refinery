from pathlib import Path

from refinery.config import Settings
from refinery.facts import FactStore
from refinery.models import LogicalDocumentUnit, PageIndex, ProvenanceRef, SectionNode
from refinery.query import QueryAgent
from refinery.storage import ArtifactStore
from refinery.vector_store import VectorIndex


def _seed_chunk(store: ArtifactStore, doc_id: str) -> None:
    ldu = LogicalDocumentUnit(
        ldu_id="ldu1",
        chunk_type="paragraph",
        content="Capital expenditure projections for Q3 increase to 120 million.",
        token_count=10,
        parent_section_path=["Financials", "CapEx"],
        parent_section="Financials > CapEx",
        page_refs=[ProvenanceRef(doc_name="sample.pdf", ref_type="pdf_bbox", page_number=4, bbox=(10, 20, 200, 220), content_hash="h")],
        content_hash="h",
    )
    store.write_jsonl(store.chunks_dir / f"{doc_id}.jsonl", [ldu.model_dump(mode="json")])


def _seed_pageindex(store: ArtifactStore, doc_id: str) -> None:
    idx = PageIndex(
        doc_id=doc_id,
        doc_name="sample.pdf",
        root_sections=[
            SectionNode(
                title="Financials",
                page_start=1,
                page_end=10,
                child_sections=[
                    SectionNode(
                        title="CapEx",
                        page_start=4,
                        page_end=5,
                        summary="Capital expenditure trends are summarized. Q3 projections are presented.",
                    )
                ],
                summary="Financial section overview. Includes key metrics.",
            )
        ],
    )
    store.save_json(store.pageindex_dir / f"{doc_id}.json", idx.model_dump(mode="json"))


def _make_agent(tmp_path: Path, doc_id: str = "doc_q") -> QueryAgent:
    settings = Settings(workspace_root=tmp_path)
    store = ArtifactStore(settings)
    _seed_chunk(store, doc_id)
    _seed_pageindex(store, doc_id)
    vector = VectorIndex(store)
    facts = FactStore(store.db_dir / "facts.db")
    return QueryAgent(store, vector, facts)


def _assert_provenance_fields(result: dict) -> None:
    prov = result.get("provenance_chain", [])
    assert prov, "provenance must be present"
    for p in prov:
        assert p.get("doc_name")
        assert p.get("page_number") is not None
        assert p.get("bbox") is not None


def test_query_interface_semantic_includes_provenance(tmp_path: Path):
    agent = _make_agent(tmp_path)
    out = agent.query_interface("doc_q", "What are the projections for Q3?")
    assert "semantic_search" in out.get("tool_trace", [])
    _assert_provenance_fields(out)


def test_query_interface_navigate_then_semantic(tmp_path: Path):
    agent = _make_agent(tmp_path)
    out = agent.query_interface("doc_q", "What are the capital expenditure projections for Q3?")
    trace = out.get("tool_trace", [])
    assert "pageindex_navigate" in trace
    assert "semantic_search" in trace
    _assert_provenance_fields(out)


def test_query_interface_structured_query_includes_provenance(tmp_path: Path):
    agent = _make_agent(tmp_path)
    rows = agent.structured_query_tool(
        "SELECT doc_id, key, value, ref_type, page_number, bbox, content_hash FROM facts LIMIT 1",
        doc_id="doc_q",
    )
    if not rows:
        agent.fact_store.ingest_ldus("doc_q", agent._load_ldus("doc_q"))

    out = agent.query_interface(
        "doc_q",
        "SELECT doc_id, key, value, ref_type, page_number, bbox, content_hash FROM facts LIMIT 1",
    )
    assert "structured_query" in out.get("tool_trace", [])
    _assert_provenance_fields(out)
