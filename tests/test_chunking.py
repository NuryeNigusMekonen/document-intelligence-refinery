from pathlib import Path

from refinery.chunking import ChunkValidator, Chunker
from refinery.config import Settings
from refinery.models import ExtractedDocument, ExtractedPage, LogicalDocumentUnit, ProvenanceRef, TextBlock
from refinery.storage import ArtifactStore


def test_chunk_validator_rejects_table_without_headers():
    validator = ChunkValidator(max_tokens=100)
    ldu = LogicalDocumentUnit(
        ldu_id="x",
        chunk_type="table",
        structured_payload={"headers": [], "rows": [["1"]]},
        token_count=3,
        page_refs=[ProvenanceRef(doc_name="x.pdf", ref_type="pdf_bbox", page_number=1, bbox=(0, 0, 1, 1), content_hash="h")],
        content_hash="h",
    )
    try:
        validator.validate(ldu)
        assert False, "Expected ValueError"
    except ValueError:
        assert True


def test_provenance_hash_stability(tmp_path: Path):
    settings = Settings(workspace_root=tmp_path)
    store = ArtifactStore(settings)
    chunker = Chunker(settings, store)
    refs = [ProvenanceRef(doc_name="x.pdf", ref_type="pdf_bbox", page_number=1, bbox=(1, 2, 3, 4), content_hash="x")]
    h1 = chunker._hash_with_provenance("Hello world", refs)
    h2 = chunker._hash_with_provenance("Hello world", refs)
    assert h1 == h2


def test_list_chunk_kept_by_item_boundaries(tmp_path: Path):
    settings = Settings(workspace_root=tmp_path, chunk_max_tokens=5)
    store = ArtifactStore(settings)
    chunker = Chunker(settings, store)
    doc = ExtractedDocument(
        doc_id="d1",
        doc_name="d.pdf",
        pages=[
            ExtractedPage(
                page_number=1,
                width=100,
                height=100,
                blocks=[TextBlock(text="1. one\n2. two\n3. three", bbox=(0, 0, 10, 10), reading_order=0)],
            )
        ],
    )
    ldus = chunker.run(doc)
    assert any(l.chunk_type == "list" for l in ldus)
