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


def test_chunk_validator_rejects_missing_required_ldu_fields():
    validator = ChunkValidator(max_tokens=100)
    ldu = LogicalDocumentUnit(
        ldu_id="x-missing",
        chunk_type="paragraph",
        content="hello",
        token_count=1,
        bounding_box=None,
        page_refs=[ProvenanceRef(doc_name="x.pdf", ref_type="pdf_bbox", page_number=1, bbox=(0, 0, 1, 1), content_hash="h")],
        content_hash="h",
    )
    try:
        validator.validate(ldu)
        assert False, "Expected ValueError"
    except ValueError:
        assert True


def test_chunk_validator_rejects_figure_without_caption_metadata_key():
    validator = ChunkValidator(max_tokens=100)
    ldu = LogicalDocumentUnit(
        ldu_id="x-figure",
        chunk_type="figure",
        content="[FIGURE]",
        structured_payload={},
        token_count=1,
        bounding_box=(0, 0, 1, 1),
        page_refs=[ProvenanceRef(doc_name="x.pdf", ref_type="pdf_bbox", page_number=1, bbox=(0, 0, 1, 1), content_hash="h")],
        content_hash="h",
    )
    try:
        validator.validate(ldu)
        assert False, "Expected ValueError"
    except ValueError:
        assert True


def test_chunk_validator_rejects_unresolved_cross_ref_relationship_marker():
    validator = ChunkValidator(max_tokens=100)
    ldu = LogicalDocumentUnit(
        ldu_id="x-ref",
        chunk_type="paragraph",
        content="see Table 1",
        token_count=3,
        bounding_box=(0, 0, 10, 10),
        page_refs=[ProvenanceRef(doc_name="x.pdf", ref_type="pdf_bbox", page_number=1, bbox=(0, 0, 10, 10), content_hash="h")],
        content_hash="h",
        relationships=["cross_ref:table"],
    )
    try:
        validator.validate_document([ldu])
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


def test_provenance_hash_stable_when_page_number_shifts(tmp_path: Path):
    settings = Settings(workspace_root=tmp_path)
    store = ArtifactStore(settings)
    chunker = Chunker(settings, store)
    refs_page_1 = [ProvenanceRef(doc_name="x.pdf", ref_type="pdf_bbox", page_number=1, bbox=(1, 2, 3, 4), content_hash="x")]
    refs_page_2 = [ProvenanceRef(doc_name="x.pdf", ref_type="pdf_bbox", page_number=2, bbox=(1, 2, 3, 4), content_hash="x")]
    h1 = chunker._hash_with_provenance("Hello world", refs_page_1)
    h2 = chunker._hash_with_provenance("Hello world", refs_page_2)
    assert h1 == h2


def test_chunk_validator_rejects_oversize_paragraph_chunk():
    validator = ChunkValidator(max_tokens=3)
    ldu = LogicalDocumentUnit(
        ldu_id="x-big",
        chunk_type="paragraph",
        content="one two three four",
        token_count=4,
        bounding_box=(0, 0, 1, 1),
        page_refs=[ProvenanceRef(doc_name="x.pdf", ref_type="pdf_bbox", page_number=1, bbox=(0, 0, 1, 1), content_hash="h")],
        content_hash="h",
    )
    try:
        validator.validate(ldu)
        assert False, "Expected ValueError"
    except ValueError:
        assert True


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


def test_numbered_list_blocks_grouped_into_single_ldu(tmp_path: Path):
    settings = Settings(workspace_root=tmp_path, chunk_max_tokens=200)
    store = ArtifactStore(settings)
    chunker = Chunker(settings, store)
    doc = ExtractedDocument(
        doc_id="d1b",
        doc_name="d.pdf",
        pages=[
            ExtractedPage(
                page_number=1,
                width=100,
                height=100,
                blocks=[
                    TextBlock(text="1. first item", bbox=(0, 0, 80, 10), reading_order=0),
                    TextBlock(text="2. second item", bbox=(0, 12, 80, 22), reading_order=1),
                    TextBlock(text="3. third item", bbox=(0, 24, 80, 34), reading_order=2),
                ],
            )
        ],
    )
    ldus = chunker.run(doc)
    list_ldus = [l for l in ldus if l.chunk_type == "list"]
    assert len(list_ldus) == 1
    assert list_ldus[0].content == "1. first item\n2. second item\n3. third item"


def test_numbered_list_split_respects_item_boundaries(tmp_path: Path):
    settings = Settings(workspace_root=tmp_path, chunk_max_tokens=6)
    store = ArtifactStore(settings)
    chunker = Chunker(settings, store)
    doc = ExtractedDocument(
        doc_id="d1c",
        doc_name="d.pdf",
        pages=[
            ExtractedPage(
                page_number=1,
                width=100,
                height=100,
                blocks=[
                    TextBlock(text="1. alpha beta gamma", bbox=(0, 0, 80, 10), reading_order=0),
                    TextBlock(text="2. delta epsilon zeta", bbox=(0, 12, 80, 22), reading_order=1),
                    TextBlock(text="3. eta theta iota", bbox=(0, 24, 80, 34), reading_order=2),
                ],
            )
        ],
    )
    ldus = chunker.run(doc)
    list_ldus = [l for l in ldus if l.chunk_type == "list"]
    assert len(list_ldus) >= 2
    for ldu in list_ldus:
        lines = [line for line in (ldu.content or "").splitlines() if line.strip()]
        assert lines
        assert all(line[:2].strip().endswith(".") for line in lines)


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


def test_parent_section_prefers_provenance_section_path(tmp_path: Path):
    settings = Settings(workspace_root=tmp_path)
    store = ArtifactStore(settings)
    chunker = Chunker(settings, store)
    section_path = ["Part I", "Revenue"]
    doc = ExtractedDocument(
        doc_id="d4",
        doc_name="d.pdf",
        pages=[
            ExtractedPage(
                page_number=1,
                width=100,
                height=100,
                blocks=[
                    TextBlock(
                        text="Total revenue increased by 10%.",
                        bbox=(0, 0, 90, 12),
                        reading_order=0,
                        provenance=ProvenanceRef(
                            doc_name="d.pdf",
                            ref_type="pdf_bbox",
                            page_number=1,
                            bbox=(0, 0, 90, 12),
                            section_path=section_path,
                            content_hash="pending",
                        ),
                    )
                ],
                tables=[
                    TableObject(
                        bbox=(0, 20, 90, 60),
                        headers=["Metric"],
                        rows=[["Revenue"]],
                        reading_order=1,
                        provenance=ProvenanceRef(
                            doc_name="d.pdf",
                            ref_type="pdf_bbox",
                            page_number=1,
                            bbox=(0, 20, 90, 60),
                            section_path=section_path,
                            content_hash="pending",
                        ),
                    )
                ],
                figures=[
                    FigureObject(
                        bbox=(0, 65, 90, 95),
                        caption="Figure 1: Revenue trend",
                        reading_order=2,
                        provenance=ProvenanceRef(
                            doc_name="d.pdf",
                            ref_type="pdf_bbox",
                            page_number=1,
                            bbox=(0, 65, 90, 95),
                            section_path=section_path,
                            content_hash="pending",
                        ),
                    )
                ],
            )
        ],
    )

    ldus = chunker.run(doc)
    paragraph = next(l for l in ldus if l.chunk_type == "paragraph")
    table = next(l for l in ldus if l.chunk_type == "table")
    figure = next(l for l in ldus if l.chunk_type == "figure")

    expected = "Part I > Revenue"
    assert paragraph.parent_section == expected
    assert table.parent_section == expected
    assert figure.parent_section == expected
