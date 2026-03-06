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


def _make_table_agent(tmp_path: Path, doc_id: str = "doc_t") -> QueryAgent:
    settings = Settings(workspace_root=tmp_path)
    store = ArtifactStore(settings)
    table_text = (
        "| ዓመት | የተመደበ በጀት | ወጪ |\n"
        "| 2013 | 97,447 | 38,600 |\n"
        "| 2012 | 82,358 | 50,401 |\n"
    )
    ldu = LogicalDocumentUnit(
        ldu_id="ldu_table_1",
        chunk_type="table",
        content=table_text,
        token_count=20,
        parent_section_path=["Budget"],
        parent_section="Budget",
        page_refs=[ProvenanceRef(doc_name="sample.pdf", ref_type="pdf_bbox", page_number=1, bbox=(10, 20, 200, 220), content_hash="h")],
        content_hash="h",
    )
    store.write_jsonl(store.chunks_dir / f"{doc_id}.jsonl", [ldu.model_dump(mode="json")])
    _seed_pageindex(store, doc_id)
    vector = VectorIndex(store)
    facts = FactStore(store.db_dir / "facts.db")
    return QueryAgent(store, vector, facts)


def _make_multisection_agent(tmp_path: Path, doc_id: str = "doc_m") -> QueryAgent:
    settings = Settings(workspace_root=tmp_path)
    store = ArtifactStore(settings)
    ldus = [
        LogicalDocumentUnit(
            ldu_id="ldu_capex",
            chunk_type="paragraph",
            content="Capital expenditure projections for Q3 increase to 120 million.",
            token_count=10,
            parent_section_path=["Financials", "CapEx"],
            parent_section="Financials > CapEx",
            page_refs=[ProvenanceRef(doc_name="sample.pdf", ref_type="pdf_bbox", page_number=4, bbox=(10, 20, 200, 220), content_hash="h-capex")],
            content_hash="h-capex",
        ),
        LogicalDocumentUnit(
            ldu_id="ldu_revenue",
            chunk_type="paragraph",
            content="Revenue outlook for Q3 remains stable at 70 million.",
            token_count=10,
            parent_section_path=["Financials", "Revenue"],
            parent_section="Financials > Revenue",
            page_refs=[ProvenanceRef(doc_name="sample.pdf", ref_type="pdf_bbox", page_number=3, bbox=(10, 20, 200, 220), content_hash="h-revenue")],
            content_hash="h-revenue",
        ),
    ]
    store.write_jsonl(store.chunks_dir / f"{doc_id}.jsonl", [ldu.model_dump(mode="json") for ldu in ldus])
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


def test_query_interface_compacts_table_answer(tmp_path: Path):
    agent = _make_table_agent(tmp_path)
    out = agent.query_interface("doc_t", "የ3ኛው ሩብ በጀት እና ወጪ መረጃ")
    answer = str(out.get("answer") or "")
    assert "Table summary:" in answer
    assert answer.count("|") < 20
    _assert_provenance_fields(out)


def test_query_interface_ollama_synthesis_replaces_refusal_when_hits_exist(tmp_path: Path, monkeypatch):
    agent = _make_agent(tmp_path, doc_id="doc_o")
    refusal = "No high-confidence answer could be synthesized from OCR text for this question."

    noisy_hit = LogicalDocumentUnit(
        ldu_id="ldu_noisy",
        chunk_type="paragraph",
        content="foo foo foo foo",
        token_count=4,
        parent_section_path=["X"],
        parent_section="X",
        page_refs=[ProvenanceRef(doc_name="sample.pdf", ref_type="pdf_bbox", page_number=2, bbox=(1, 2, 3, 4), content_hash="h2")],
        content_hash="h2",
    )

    monkeypatch.setattr(agent, "semantic_search_tool", lambda *_args, **_kwargs: [noisy_hit])
    monkeypatch.setattr(agent, "structured_query_tool", lambda *_args, **_kwargs: [])
    monkeypatch.setattr(agent, "_synthesize_with_ollama", lambda *_args, **_kwargs: "Found 120 የተመደበ")

    out = agent.query_interface("doc_o", "What is Q3 budget?")
    assert out["answer"] != refusal
    assert out["confidence"] > 0.25
    _assert_provenance_fields(out)


def test_query_interface_ollama_empty_context_keeps_refusal(tmp_path: Path, monkeypatch):
    agent = _make_agent(tmp_path, doc_id="doc_e")
    refusal = "No high-confidence answer could be synthesized from OCR text for this question."

    empty_hit = LogicalDocumentUnit(
        ldu_id="ldu_empty",
        chunk_type="paragraph",
        content="",
        token_count=1,
        parent_section_path=["X"],
        parent_section="X",
        page_refs=[ProvenanceRef(doc_name="sample.pdf", ref_type="pdf_bbox", page_number=3, bbox=(5, 6, 7, 8), content_hash="h3")],
        content_hash="h3",
    )

    monkeypatch.setattr(agent, "semantic_search_tool", lambda *_args, **_kwargs: [empty_hit])
    monkeypatch.setattr(agent, "structured_query_tool", lambda *_args, **_kwargs: [])
    monkeypatch.setattr(agent, "_synthesize_with_ollama", lambda *_args, **_kwargs: "Not found in context")

    out = agent.query_interface("doc_e", "What is Q3 budget?")
    assert out["answer"] == refusal
    _assert_provenance_fields(out)


def test_query_interface_navigational_scopes_semantic_retrieval(tmp_path: Path, monkeypatch):
    agent = _make_multisection_agent(tmp_path)

    def _fail_global_semantic(*_args, **_kwargs):
        raise AssertionError("global semantic_search_tool should not be used when scoped candidates exist")

    monkeypatch.setattr(agent, "semantic_search_tool", _fail_global_semantic)

    out = agent.query_interface("doc_m", "What are the capital expenditure projections for Q3?")
    answer = str(out.get("answer") or "")
    assert "capital expenditure" in answer.lower()
    assert "revenue outlook" not in answer.lower()
    assert "pageindex_navigate" in out.get("tool_trace", [])
    assert "semantic_search" in out.get("tool_trace", [])
    _assert_provenance_fields(out)


def test_query_interface_navigational_falls_back_when_scoped_empty(tmp_path: Path, monkeypatch):
    agent = _make_agent(tmp_path, doc_id="doc_f")
    called = {"semantic": False}

    monkeypatch.setattr(
        agent,
        "pageindex_navigate_tool",
        lambda *_args, **_kwargs: [{"title": "UnrelatedSection", "summary": "", "child_sections": []}],
    )

    original_semantic = agent.semantic_search_tool

    def _wrapped_semantic(*args, **kwargs):
        called["semantic"] = True
        return original_semantic(*args, **kwargs)

    monkeypatch.setattr(agent, "semantic_search_tool", _wrapped_semantic)

    out = agent.query_interface("doc_f", "What are the capital expenditure projections for Q3?")
    assert called["semantic"] is True
    assert "semantic_search" in out.get("tool_trace", [])
    _assert_provenance_fields(out)


def test_query_interface_navigation_debug_opt_in(tmp_path: Path):
    agent = _make_multisection_agent(tmp_path)

    default_out = agent.query_interface("doc_m", "What are the capital expenditure projections for Q3?")
    assert "navigation_sections" not in default_out
    assert "used_section_scope" not in default_out

    debug_out = agent.query_interface(
        "doc_m",
        "What are the capital expenditure projections for Q3?",
        include_navigation_debug=True,
    )
    assert "navigation_sections" in debug_out
    assert "used_section_scope" in debug_out
    assert "CapEx" in debug_out.get("navigation_sections", [])
    assert debug_out.get("used_section_scope") is True


def test_measure_retrieval_precision_with_and_without_pageindex(tmp_path: Path):
    agent = _make_multisection_agent(tmp_path, doc_id="doc_precision")
    metrics = agent.measure_retrieval_precision(
        doc_id="doc_precision",
        topic="capital expenditure projections for Q3",
        expected_sections=["CapEx"],
        top_k=3,
    )

    assert metrics["top_k"] == 3
    assert "with_pageindex" in metrics
    assert "without_pageindex" in metrics
    assert "precision_at_k" in metrics["with_pageindex"]
    assert "precision_at_k" in metrics["without_pageindex"]
    assert metrics["with_pageindex"]["precision_at_k"] >= metrics["without_pageindex"]["precision_at_k"]
