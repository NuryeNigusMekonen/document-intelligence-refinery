from pathlib import Path

from refinery.chunking import ChunkValidator, Chunker
from refinery.config import Settings
from refinery.models import ExtractedDocument, ExtractedPage, FigureObject, LogicalDocumentUnit, ProvenanceRef, TableObject, TextBlock
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


def test_figure_caption_is_metadata_and_bbox_present(tmp_path: Path):
    settings = Settings(workspace_root=tmp_path)
    store = ArtifactStore(settings)
    chunker = Chunker(settings, store)
    doc = ExtractedDocument(
        doc_id="d2",
        doc_name="d.pdf",
        pages=[
            ExtractedPage(
                page_number=1,
                width=100,
                height=100,
                figures=[FigureObject(bbox=(10, 10, 90, 90), caption="Figure 1: Pipeline", reading_order=0)],
            )
        ],
    )
    ldus = chunker.run(doc)
    figure_ldu = next(l for l in ldus if l.chunk_type == "figure")
    assert figure_ldu.structured_payload is not None
    assert figure_ldu.structured_payload.get("caption") == "Figure 1: Pipeline"
    assert figure_ldu.bounding_box == (10.0, 10.0, 90.0, 90.0)


def test_parent_section_and_cross_reference_resolution(tmp_path: Path):
    settings = Settings(workspace_root=tmp_path)
    store = ArtifactStore(settings)
    chunker = Chunker(settings, store)
    doc = ExtractedDocument(
        doc_id="d3",
        doc_name="d.pdf",
        pages=[
            ExtractedPage(
                page_number=1,
                width=100,
                height=100,
                blocks=[
                    TextBlock(text="1 Financial Overview", bbox=(0, 0, 80, 10), reading_order=0),
                    TextBlock(text="see Table 1 for totals", bbox=(0, 12, 90, 22), reading_order=1),
                ],
                tables=[TableObject(bbox=(0, 30, 90, 80), headers=["h1"], rows=[["v1"]], reading_order=2)],
            )
        ],
    )
    ldus = chunker.run(doc)
    paragraph = next(l for l in ldus if l.chunk_type == "paragraph")
    table = next(l for l in ldus if l.chunk_type == "table")
    assert paragraph.parent_section == "1 Financial Overview"
    assert paragraph.bounding_box == (0.0, 12.0, 90.0, 22.0)
    assert any(rel.endswith(f"->{table.ldu_id}") for rel in paragraph.relationships if rel.startswith("resolved_ref:table:1"))
