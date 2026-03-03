from pathlib import Path

from refinery.facts import FactStore
from refinery.models import LogicalDocumentUnit, ProvenanceRef
from tests.fixtures.amharic_samples import AMHARIC_PARAGRAPH


def test_amharic_fact_extraction_with_provenance(tmp_path: Path):
    db = tmp_path / "facts.db"
    store = FactStore(db)
    ldu = LogicalDocumentUnit(
        ldu_id="ldu_am",
        chunk_type="paragraph",
        content=AMHARIC_PARAGRAPH,
        token_count=10,
        parent_section_path=["Financial"],
        page_refs=[
            ProvenanceRef(
                doc_name="am.md",
                ref_type="markdown_lines",
                line_range=(1, 1),
                section_path=["Financial"],
                content_hash="ham",
            )
        ],
        content_hash="ham",
    )
    inserted = store.ingest_ldus("doc_am", [ldu])
    assert inserted >= 1
    rows = store.query("Revenue", doc_id="doc_am")
    assert rows
    assert rows[0]["ref_type"] == "markdown_lines"
