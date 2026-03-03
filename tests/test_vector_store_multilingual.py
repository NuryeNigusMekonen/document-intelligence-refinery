from pathlib import Path

from refinery.config import Settings
from refinery.models import LogicalDocumentUnit, ProvenanceRef
from refinery.storage import ArtifactStore
from refinery.vector_store import VectorIndex


def _ldu(ldu_id: str, text: str, content_hash: str) -> LogicalDocumentUnit:
    return LogicalDocumentUnit(
        ldu_id=ldu_id,
        chunk_type="paragraph",
        content=text,
        token_count=5,
        parent_section_path=[],
        page_refs=[ProvenanceRef(doc_name="x", ref_type="markdown_lines", line_range=(1, 1), content_hash=content_hash)],
        content_hash=content_hash,
    )


def test_multilingual_vector_retrieval_prefers_amharic(tmp_path: Path):
    settings = Settings(workspace_root=tmp_path, multilingual_embeddings=True)
    store = ArtifactStore(settings)
    index = VectorIndex(store)
    doc_id = "doc_multi"
    am = _ldu("a", "የኩባንያው ገቢ በ2024 ዓመት 100 ሚሊዮን ነው", "ha")
    en = _ldu("b", "Company revenue in 2024 is 100 million", "hb")
    index.build(doc_id, [en, am])

    hits = index.search(doc_id, "ገቢ 2024", top_k=2)
    assert hits
    assert hits[0].ldu_id == "a"
