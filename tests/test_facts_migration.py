import sqlite3
from pathlib import Path

from refinery.facts import FactStore
from refinery.models import LogicalDocumentUnit, ProvenanceRef


def test_facts_table_legacy_schema_auto_migrates(tmp_path: Path):
    db_path = tmp_path / "facts.db"

    with sqlite3.connect(db_path) as conn:
        conn.execute(
            """
            CREATE TABLE facts (
                doc_id TEXT,
                key TEXT,
                value TEXT,
                unit TEXT,
                context TEXT,
                page_number INTEGER,
                bbox TEXT,
                content_hash TEXT
            )
            """
        )
        conn.commit()

    store = FactStore(db_path)

    with sqlite3.connect(db_path) as conn:
        cols = {row[1] for row in conn.execute("PRAGMA table_info(facts)").fetchall()}

    assert "ref_type" in cols
    assert "section_path" in cols
    assert "line_range" in cols
    assert "sheet_name" in cols
    assert "cell_range" in cols

    ldu = LogicalDocumentUnit(
        ldu_id="ldu1",
        chunk_type="paragraph",
        content="Revenue was $100 in FY2024",
        token_count=6,
        parent_section_path=["Financial Results"],
        page_refs=[
            ProvenanceRef(
                doc_name="doc.md",
                ref_type="markdown_lines",
                line_range=(10, 10),
                section_path=["Financial Results"],
                content_hash="h1",
            )
        ],
        content_hash="h1",
    )

    inserted = store.ingest_ldus("doc_1", [ldu])
    assert inserted >= 1
