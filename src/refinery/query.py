from __future__ import annotations

import json
from pathlib import Path

from .facts import FactStore
from .models import LogicalDocumentUnit, PageIndex, ProvenanceRef, QueryAnswer
from .pageindex import pageindex_navigate
from .storage import ArtifactStore
from .utils import normalize_text
from .vector_store import VectorIndex


class QueryAgent:
    def __init__(self, store: ArtifactStore, vector_index: VectorIndex, fact_store: FactStore):
        self.store = store
        self.vector_index = vector_index
        self.fact_store = fact_store

    def _load_ldus(self, doc_id: str) -> list[LogicalDocumentUnit]:
        rows = self.store.read_jsonl(self.store.chunks_dir / f"{doc_id}.jsonl")
        return [LogicalDocumentUnit.model_validate(r) for r in rows]

    def _load_pageindex(self, doc_id: str) -> PageIndex:
        return PageIndex.model_validate(self.store.load_json(self.store.pageindex_dir / f"{doc_id}.json"))

    def query(self, doc_id: str, question: str) -> QueryAnswer:
        ldus = self._load_ldus(doc_id)
        self.vector_index.build(doc_id, ldus)
        hits = self.vector_index.search(doc_id, question, top_k=5)
        if not hits:
            return QueryAnswer(answer="No high-confidence match found.", provenance_chain=[], confidence=0.2)

        answer_text = " ".join([(h.content or "") for h in hits[:3]]).strip()
        prov = []
        for h in hits[:3]:
            for pref in h.page_refs[:1]:
                ref = pref.model_copy()
                ref.content_hash = h.content_hash
                prov.append(ref)
        return QueryAnswer(answer=answer_text[:900], provenance_chain=prov, confidence=0.72)

    def audit_claim(self, doc_id: str, claim: str) -> dict:
        result = self.query(doc_id, claim)
        status = "VERIFIED" if result.provenance_chain else "NOT FOUND / UNVERIFIABLE"
        return {
            "status": status,
            "claim": claim,
            "answer": result.answer,
            "provenance_chain": [p.model_dump() for p in result.provenance_chain],
        }

    def navigate(self, doc_id: str, topic: str) -> list[dict]:
        idx = self._load_pageindex(doc_id)
        return [s.model_dump() for s in pageindex_navigate(idx, topic)]

    def structured_query(self, query: str, doc_id: str | None = None) -> list[dict]:
        return self.fact_store.query(query=query, doc_id=doc_id)


def open_citation_snippet(store: ArtifactStore, doc_id: str, page_number: int, bbox: tuple[float, float, float, float]) -> dict:
    rows = store.read_jsonl(store.chunks_dir / f"{doc_id}.jsonl")
    for row in rows:
        ldu = LogicalDocumentUnit.model_validate(row)
        for pref in ldu.page_refs:
            if pref.ref_type != "pdf_bbox" or pref.page_number != page_number or not pref.bbox:
                continue
            px0, py0, px1, py1 = pref.bbox
            x0, y0, x1, y1 = bbox
            overlap = not (px1 < x0 or x1 < px0 or py1 < y0 or y1 < py0)
            if overlap:
                return {
                    "doc_id": doc_id,
                    "page_number": page_number,
                    "bbox": bbox,
                    "snippet": normalize_text(ldu.content or json.dumps(ldu.structured_payload or {}))[:300],
                    "content_hash": ldu.content_hash,
                }
    return {
        "doc_id": doc_id,
        "page_number": page_number,
        "bbox": bbox,
        "snippet": "No snippet matched the requested citation bbox.",
        "content_hash": "",
    }


def format_provenance_ref(pref: ProvenanceRef) -> str:
    if pref.ref_type == "pdf_bbox":
        return f"Page {pref.page_number} | bbox {list(pref.bbox or ())}"
    if pref.ref_type == "word_section":
        return f"Section: {' > '.join(pref.section_path or [])}"
    if pref.ref_type == "markdown_lines":
        if pref.line_range:
            return f"Lines {pref.line_range[0]}-{pref.line_range[1]}"
        return "Lines: unknown"
    if pref.ref_type == "excel_cells":
        return f"Sheet: {pref.sheet_name or ''} | Cells: {pref.cell_range or ''}"
    if pref.ref_type == "image_bbox":
        return f"Image bbox: {list(pref.bbox or ())}"
    return "Unknown provenance"
