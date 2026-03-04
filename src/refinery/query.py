from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, TypedDict

from .facts import FactStore
from .models import LogicalDocumentUnit, PageIndex, ProvenanceRef, QueryAnswer
from .pageindex import pageindex_navigate
from .storage import ArtifactStore
from .utils import normalize_text
from .vector_store import VectorIndex


class QueryGraphState(TypedDict, total=False):
    doc_id: str
    question: str
    route: str
    semantic_query: str
    sections: list[dict]
    hits: list[LogicalDocumentUnit]
    rows: list[dict]
    answer: str
    provenance_chain: list[dict]
    confidence: float
    tool_trace: list[str]


class QueryAgent:
    def __init__(self, store: ArtifactStore, vector_index: VectorIndex, fact_store: FactStore):
        self.store = store
        self.vector_index = vector_index
        self.fact_store = fact_store
        self._langgraph_app: Any | None = None
        self._langgraph_checked = False

    def _load_ldus(self, doc_id: str) -> list[LogicalDocumentUnit]:
        rows = self.store.read_jsonl(self.store.chunks_dir / f"{doc_id}.jsonl")
        return [LogicalDocumentUnit.model_validate(r) for r in rows]

    def _load_pageindex(self, doc_id: str) -> PageIndex:
        return PageIndex.model_validate(self.store.load_json(self.store.pageindex_dir / f"{doc_id}.json"))

    def pageindex_navigate_tool(self, doc_id: str, topic: str) -> list[dict]:
        idx = self._load_pageindex(doc_id)
        return [s.model_dump() for s in pageindex_navigate(idx, topic)]

    def semantic_search_tool(self, doc_id: str, query: str, top_k: int = 5) -> list[LogicalDocumentUnit]:
        ldus = self._load_ldus(doc_id)
        self.vector_index.build(doc_id, ldus)
        return self.vector_index.search(doc_id, query, top_k=top_k)

    def structured_query_tool(self, query: str, doc_id: str | None = None) -> list[dict]:
        return self.fact_store.query(query=query, doc_id=doc_id)

    def _primary_doc_name(self, doc_id: str) -> str:
        ldus = self._load_ldus(doc_id)
        for ldu in ldus:
            for pref in ldu.page_refs:
                if pref.doc_name:
                    return pref.doc_name
        return doc_id

    def _normalize_pref(self, pref: ProvenanceRef, fallback_doc_name: str) -> ProvenanceRef:
        bbox = pref.bbox or (0.0, 0.0, 0.0, 0.0)
        page = pref.page_number if pref.page_number is not None else 1
        return pref.model_copy(update={"doc_name": pref.doc_name or fallback_doc_name, "page_number": page, "bbox": bbox})

    def _fallback_provenance(self, doc_id: str) -> list[ProvenanceRef]:
        ldus = self._load_ldus(doc_id)
        if ldus and ldus[0].page_refs:
            pref = ldus[0].page_refs[0]
            return [self._normalize_pref(pref, fallback_doc_name=self._primary_doc_name(doc_id))]
        return [
            ProvenanceRef(
                doc_name=doc_id,
                ref_type="pdf_bbox",
                page_number=1,
                bbox=(0.0, 0.0, 0.0, 0.0),
                content_hash="",
            )
        ]

    def _provenance_from_hits(self, doc_id: str, hits: list[LogicalDocumentUnit]) -> list[ProvenanceRef]:
        out: list[ProvenanceRef] = []
        doc_name = self._primary_doc_name(doc_id)
        for hit in hits[:3]:
            if not hit.page_refs:
                continue
            pref = self._normalize_pref(hit.page_refs[0].model_copy(update={"content_hash": hit.content_hash}), doc_name)
            out.append(pref)
        return out or self._fallback_provenance(doc_id)

    def _provenance_from_fact_rows(self, doc_id: str, rows: list[dict]) -> list[ProvenanceRef]:
        out: list[ProvenanceRef] = []
        doc_name = self._primary_doc_name(doc_id)
        for row in rows[:3]:
            bbox_text = str(row.get("bbox") or "").strip()
            bbox = (0.0, 0.0, 0.0, 0.0)
            if bbox_text:
                parts = [p.strip() for p in bbox_text.split(",")]
                if len(parts) == 4:
                    try:
                        bbox = (float(parts[0]), float(parts[1]), float(parts[2]), float(parts[3]))
                    except ValueError:
                        bbox = (0.0, 0.0, 0.0, 0.0)

            section_path = [s for s in str(row.get("section_path") or "").split(">") if s]
            line_range = None
            line_range_text = str(row.get("line_range") or "").strip()
            if line_range_text:
                parts = [p.strip() for p in line_range_text.split(",")]
                if len(parts) == 2 and all(p.isdigit() for p in parts):
                    line_range = (int(parts[0]), int(parts[1]))

            pref = ProvenanceRef(
                doc_name=doc_name,
                ref_type=row.get("ref_type") or "pdf_bbox",
                page_number=int(row.get("page_number") or 1),
                bbox=bbox,
                section_path=section_path or None,
                line_range=line_range,
                sheet_name=row.get("sheet_name") or None,
                cell_range=row.get("cell_range") or None,
                content_hash=str(row.get("content_hash") or ""),
            )
            out.append(pref)
        return out or self._fallback_provenance(doc_id)

    def _looks_structured_query(self, question: str) -> bool:
        q = question.lower().strip()
        return bool(re.search(r"\bselect\b|\bfrom\b|\bwhere\b|\bgroup\s+by\b|\bsum\b|\bavg\b", q))

    def _looks_navigational(self, question: str) -> bool:
        q = question.lower()
        return any(k in q for k in ["section", "where in", "capital expenditure", "capex", "projections", "q1", "q2", "q3", "q4", "table", "figure"])

    def _query_interface_linear(self, doc_id: str, question: str) -> dict:
        tool_trace: list[str] = []

        if self._looks_structured_query(question):
            tool_trace.append("structured_query")
            rows = self.structured_query_tool(question, doc_id=doc_id)
            answer = json.dumps(rows[:5], ensure_ascii=False)
            provenance = self._provenance_from_fact_rows(doc_id, rows)
            return {
                "answer": answer,
                "provenance_chain": [p.model_dump(mode="json") for p in provenance],
                "confidence": 0.70 if rows else 0.35,
                "tool_trace": tool_trace,
            }

        semantic_query = question
        if self._looks_navigational(question):
            tool_trace.append("pageindex_navigate")
            sections = self.pageindex_navigate_tool(doc_id, question)
            if sections:
                titles = [str(s.get("title") or "") for s in sections[:2]]
                semantic_query = f"{question} {' '.join(titles)}".strip()

        tool_trace.append("semantic_search")
        hits = self.semantic_search_tool(doc_id, semantic_query, top_k=self.store.settings.query_semantic_top_k)
        if hits:
            answer_text = " ".join([(h.content or "") for h in hits[:3]]).strip()[:900]
            provenance = self._provenance_from_hits(doc_id, hits)
            return {
                "answer": answer_text,
                "provenance_chain": [p.model_dump(mode="json") for p in provenance],
                "confidence": 0.72,
                "tool_trace": tool_trace,
            }

        tool_trace.append("structured_query")
        rows = self.structured_query_tool(question, doc_id=doc_id)
        if rows:
            answer = json.dumps(rows[:5], ensure_ascii=False)
            provenance = self._provenance_from_fact_rows(doc_id, rows)
            return {
                "answer": answer,
                "provenance_chain": [p.model_dump(mode="json") for p in provenance],
                "confidence": 0.55,
                "tool_trace": tool_trace,
            }

        fallback_prov = self._fallback_provenance(doc_id)
        return {
            "answer": "No high-confidence match found.",
            "provenance_chain": [p.model_dump(mode="json") for p in fallback_prov],
            "confidence": 0.2,
            "tool_trace": tool_trace,
        }

    def _build_langgraph_app(self):
        if self._langgraph_checked:
            return self._langgraph_app
        self._langgraph_checked = True
        try:
            from langgraph.graph import END, START, StateGraph
        except Exception:
            self._langgraph_app = None
            return None

        def route_node(state: QueryGraphState) -> QueryGraphState:
            question = str(state.get("question") or "")
            tool_trace = list(state.get("tool_trace") or [])
            if self._looks_structured_query(question):
                tool_trace.append("structured_query")
                return {"route": "structured", "tool_trace": tool_trace, "semantic_query": question}

            if self._looks_navigational(question):
                sections = self.pageindex_navigate_tool(str(state.get("doc_id") or ""), question)
                tool_trace.append("pageindex_navigate")
                semantic_query = question
                if sections:
                    titles = [str(s.get("title") or "") for s in sections[:2]]
                    semantic_query = f"{question} {' '.join(titles)}".strip()
                tool_trace.append("semantic_search")
                return {
                    "route": "semantic",
                    "sections": sections,
                    "semantic_query": semantic_query,
                    "tool_trace": tool_trace,
                }

            tool_trace.append("semantic_search")
            return {"route": "semantic", "semantic_query": question, "tool_trace": tool_trace}

        def execute_node(state: QueryGraphState) -> QueryGraphState:
            route = state.get("route")
            doc_id = str(state.get("doc_id") or "")
            question = str(state.get("question") or "")
            tool_trace = list(state.get("tool_trace") or [])

            if route == "structured":
                rows = self.structured_query_tool(question, doc_id=doc_id)
                return {"rows": rows, "tool_trace": tool_trace}

            semantic_query = str(state.get("semantic_query") or question)
            hits = self.semantic_search_tool(doc_id, semantic_query, top_k=self.store.settings.query_semantic_top_k)
            if hits:
                return {"hits": hits, "tool_trace": tool_trace}

            tool_trace.append("structured_query")
            rows = self.structured_query_tool(question, doc_id=doc_id)
            return {"rows": rows, "tool_trace": tool_trace}

        def finalize_node(state: QueryGraphState) -> QueryGraphState:
            doc_id = str(state.get("doc_id") or "")
            tool_trace = list(state.get("tool_trace") or [])
            hits = state.get("hits") or []
            if hits:
                answer_text = " ".join([(h.content or "") for h in hits[:3]]).strip()[:900]
                provenance = self._provenance_from_hits(doc_id, hits)
                return {
                    "answer": answer_text,
                    "provenance_chain": [p.model_dump(mode="json") for p in provenance],
                    "confidence": 0.72,
                    "tool_trace": tool_trace,
                }

            rows = state.get("rows") or []
            if rows:
                answer = json.dumps(rows[:5], ensure_ascii=False)
                provenance = self._provenance_from_fact_rows(doc_id, rows)
                conf = 0.70 if "structured_query" in tool_trace and len(tool_trace) == 1 else 0.55
                return {
                    "answer": answer,
                    "provenance_chain": [p.model_dump(mode="json") for p in provenance],
                    "confidence": conf,
                    "tool_trace": tool_trace,
                }

            fallback_prov = self._fallback_provenance(doc_id)
            return {
                "answer": "No high-confidence match found.",
                "provenance_chain": [p.model_dump(mode="json") for p in fallback_prov],
                "confidence": 0.2,
                "tool_trace": tool_trace,
            }

        graph = StateGraph(QueryGraphState)
        graph.add_node("route", route_node)
        graph.add_node("execute", execute_node)
        graph.add_node("finalize", finalize_node)
        graph.add_edge(START, "route")
        graph.add_edge("route", "execute")
        graph.add_edge("execute", "finalize")
        graph.add_edge("finalize", END)
        self._langgraph_app = graph.compile()
        return self._langgraph_app

    def query_interface(self, doc_id: str, question: str) -> dict:
        if not self.store.settings.query_use_langgraph:
            return self._query_interface_linear(doc_id, question)

        app = self._build_langgraph_app()
        if app is None:
            return self._query_interface_linear(doc_id, question)

        try:
            result = app.invoke({"doc_id": doc_id, "question": question, "tool_trace": []})
            return {
                "answer": result.get("answer", "No high-confidence match found."),
                "provenance_chain": result.get("provenance_chain", [p.model_dump(mode="json") for p in self._fallback_provenance(doc_id)]),
                "confidence": float(result.get("confidence", 0.2)),
                "tool_trace": result.get("tool_trace", []),
            }
        except Exception:
            return self._query_interface_linear(doc_id, question)

    def query(self, doc_id: str, question: str) -> QueryAnswer:
        out = self.query_interface(doc_id, question)
        prov = [ProvenanceRef.model_validate(p) for p in out.get("provenance_chain", [])]
        return QueryAnswer(answer=out["answer"], provenance_chain=prov, confidence=float(out.get("confidence", 0.2)))

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
        return self.pageindex_navigate_tool(doc_id, topic)

    def structured_query(self, query: str, doc_id: str | None = None) -> list[dict]:
        return self.structured_query_tool(query=query, doc_id=doc_id)


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
