import pytest
from pydantic import ValidationError

from refinery.models import ConfidenceSignal, LogicalDocumentUnit, ProvenanceChain, ProvenanceRef, QueryAnswer, TextBlock


def test_bbox_sanity_rejects_invalid_coordinates():
    with pytest.raises(ValidationError):
        ProvenanceRef(
            doc_name="x.pdf",
            ref_type="pdf_bbox",
            page_number=1,
            bbox=(10.0, 10.0, 5.0, 20.0),
            content_hash="h1",
        )


def test_provenance_requires_ref_specific_payload():
    with pytest.raises(ValidationError):
        ProvenanceRef(
            doc_name="x.md",
            ref_type="markdown_lines",
            page_number=1,
            content_hash="h2",
        )


def test_line_range_must_be_valid_increasing_range():
    with pytest.raises(ValidationError):
        ProvenanceRef(
            doc_name="x.md",
            ref_type="markdown_lines",
            page_number=1,
            line_range=(10, 2),
            content_hash="h3",
        )


def test_ldu_auto_builds_aggregated_provenance_chain():
    refs = [
        ProvenanceRef(doc_name="x.pdf", ref_type="pdf_bbox", page_number=1, bbox=(0.0, 0.0, 5.0, 5.0), content_hash="h4"),
        ProvenanceRef(doc_name="x.pdf", ref_type="pdf_bbox", page_number=2, bbox=(5.0, 5.0, 10.0, 10.0), content_hash="h5"),
    ]
    ldu = LogicalDocumentUnit(
        ldu_id="ldu-multi",
        chunk_type="paragraph",
        content="Merged content",
        token_count=2,
        page_refs=refs,
        content_hash="hc",
    )
    assert ldu.provenance_chain is not None
    assert ldu.provenance_chain.chain_type == "aggregated"
    assert len(ldu.provenance_chain.steps) == 2


def test_query_answer_accepts_legacy_list_provenance_chain():
    answer = QueryAnswer(
        answer="ok",
        provenance_chain=[
            ProvenanceRef(doc_name="x.pdf", ref_type="pdf_bbox", page_number=1, bbox=(0.0, 0.0, 3.0, 3.0), content_hash="h6")
        ],
        confidence=0.6,
    )
    assert isinstance(answer.provenance_chain, ProvenanceChain)
    assert answer.provenance_chain.chain_type == "single_source"
    assert len(answer.provenance_chain.steps) == 1


def test_text_block_accepts_confidence_signals_payload():
    block = TextBlock(
        text="hello",
        bbox=(0.0, 0.0, 10.0, 10.0),
        reading_order=0,
        confidence=0.8,
        confidence_signals=[
            ConfidenceSignal(
                signal="ocr_word_conf",
                value=84.0,
                normalized_value=0.84,
                weight=0.8,
                threshold=60.0,
                passed=True,
            )
        ],
    )
    assert block.confidence_signals
    assert block.confidence_signals[0].signal == "ocr_word_conf"
