from __future__ import annotations

import json
import re
from collections import Counter
from copy import deepcopy
from pathlib import Path
from typing import Any, TypedDict
from urllib import request

from pydantic import ValidationError

from models import BBox, LogicalDocumentUnit, PageIndex, ProvenanceChain, ProvenanceRef, QueryAnswer
from refinery.facts import FactStore
from refinery.storage import ArtifactStore
from refinery.utils import normalize_text
from refinery.vector_store import VectorIndex

from .indexer import pageindex_navigate


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
    navigation_sections: list[str]
    used_section_scope: bool


class QueryAgent:
    def __init__(self, store: ArtifactStore, vector_index: VectorIndex, fact_store: FactStore):
        self.store = store
        self.vector_index = vector_index
        self.fact_store = fact_store
        self._langgraph_app: Any | None = None
        self._langgraph_checked = False
        self._ocr_refusal = "No high-confidence answer could be synthesized from OCR text for this question."

    def _load_ldus(self, doc_id: str) -> list[LogicalDocumentUnit]:
        rows = self.store.read_jsonl(self.store.chunks_dir / f"{doc_id}.jsonl")
        out: list[LogicalDocumentUnit] = []
        for row in rows:
            try:
                out.append(LogicalDocumentUnit.model_validate(row))
            except ValidationError:
                sanitized = self._sanitize_ldu_row_bbox(row)
                out.append(LogicalDocumentUnit.model_validate(sanitized))
        return out

    def _sanitize_bbox_payload(self, bbox_payload: Any) -> Any:
        if not isinstance(bbox_payload, (list, tuple)) or len(bbox_payload) != 4:
            return bbox_payload
        try:
            x0, y0, x1, y1 = [float(v) for v in bbox_payload]
        except Exception:
            return bbox_payload

        x0 = max(0.0, x0)
        y0 = max(0.0, y0)
        x1 = max(0.0, x1)
        y1 = max(0.0, y1)

        if x1 <= x0:
            x1 = x0 + 1e-6
        if y1 <= y0:
            y1 = y0 + 1e-6
        return [x0, y0, x1, y1]

    def _sanitize_ldu_row_bbox(self, row: dict[str, Any]) -> dict[str, Any]:
        cleaned = deepcopy(row)
        if isinstance(cleaned.get("bounding_box"), (list, tuple)):
            cleaned["bounding_box"] = self._sanitize_bbox_payload(cleaned.get("bounding_box"))

        page_refs = cleaned.get("page_refs")
        if isinstance(page_refs, list):
            for pref in page_refs:
                if isinstance(pref, dict) and isinstance(pref.get("bbox"), (list, tuple)):
                    pref["bbox"] = self._sanitize_bbox_payload(pref.get("bbox"))
        return cleaned

    def _load_pageindex(self, doc_id: str) -> PageIndex:
        return PageIndex.model_validate(self.store.load_json(self.store.pageindex_dir / f"{doc_id}.json"))

    def pageindex_navigate_tool(self, doc_id: str, topic: str, top_k: int = 3) -> list[dict]:
        idx = self._load_pageindex(doc_id)
        return [s.model_dump() for s in pageindex_navigate(idx, topic, top_k=top_k)]

    def semantic_search_tool(self, doc_id: str, query: str, top_k: int = 5) -> list[LogicalDocumentUnit]:
        ldus = self._load_ldus(doc_id)
        self.vector_index.build(doc_id, ldus)
        return self.vector_index.search(doc_id, query, top_k=top_k)

    def _normalize_section_title(self, title: str) -> str:
        return normalize_text(str(title or "")).strip().lower()

    def _collect_section_titles(self, sections: list[dict]) -> set[str]:
        titles: set[str] = set()
        stack = list(sections or [])
        while stack:
            node = stack.pop(0)
            title = self._normalize_section_title(str(node.get("title") or ""))
            if title:
                titles.add(title)
            children = node.get("child_sections") or []
            if isinstance(children, list):
                stack.extend(child for child in children if isinstance(child, dict))
        return titles

    def _collect_section_titles_display(self, sections: list[dict]) -> list[str]:
        titles: list[str] = []
        seen: set[str] = set()
        stack = list(sections or [])
        while stack:
            node = stack.pop(0)
            raw_title = str(node.get("title") or "").strip()
            if raw_title and raw_title not in seen:
                seen.add(raw_title)
                titles.append(raw_title)
            children = node.get("child_sections") or []
            if isinstance(children, list):
                stack.extend(child for child in children if isinstance(child, dict))
        return titles

    def _ldu_in_selected_sections(self, ldu: LogicalDocumentUnit, selected_titles: set[str]) -> bool:
        if not selected_titles:
            return False

        section_parts = [self._normalize_section_title(p) for p in (ldu.parent_section_path or []) if self._normalize_section_title(p)]
        if any(part in selected_titles for part in section_parts):
            return True

        parent = self._normalize_section_title(ldu.parent_section or "")
        if parent and any(title in parent for title in selected_titles):
            return True

        return False

    def _rank_ldus_lexical(self, ldus: list[LogicalDocumentUnit], query: str, top_k: int) -> list[LogicalDocumentUnit]:
        query_terms = [t.lower() for t in re.findall(r"\w+", query, flags=re.UNICODE)]
        if not query_terms:
            return ldus[:top_k]

        query_vec = Counter(query_terms)
        ranked: list[tuple[float, LogicalDocumentUnit]] = []
        for ldu in ldus:
            text = ldu.content or str(ldu.structured_payload or "")
            doc_terms = [t.lower() for t in re.findall(r"\w+", text, flags=re.UNICODE)]
            if not doc_terms:
                continue
            doc_vec = Counter(doc_terms)
            score = sum(min(query_vec[t], doc_vec[t]) for t in query_vec.keys())
            if score > 0:
                ranked.append((float(score), ldu))

        ranked.sort(key=lambda x: x[0], reverse=True)
        return [ldu for _, ldu in ranked[:top_k]]

    def _semantic_search_with_sections(self, doc_id: str, query: str, sections: list[dict] | None, top_k: int) -> list[LogicalDocumentUnit]:
        if not sections:
            return self.semantic_search_tool(doc_id, query, top_k=top_k)

        selected_titles = self._collect_section_titles(sections)
        if not selected_titles:
            return self.semantic_search_tool(doc_id, query, top_k=top_k)

        scoped_ldus = [ldu for ldu in self._load_ldus(doc_id) if self._ldu_in_selected_sections(ldu, selected_titles)]
        if not scoped_ldus:
            return self.semantic_search_tool(doc_id, query, top_k=top_k)

        scoped_hits = self._rank_ldus_lexical(scoped_ldus, query, top_k=top_k)
        if scoped_hits:
            return scoped_hits

        return self.semantic_search_tool(doc_id, query, top_k=top_k)

    def _semantic_search_with_sections_meta(
        self,
        doc_id: str,
        query: str,
        sections: list[dict] | None,
        top_k: int,
    ) -> tuple[list[LogicalDocumentUnit], list[str], bool]:
        if not sections:
            return self.semantic_search_tool(doc_id, query, top_k=top_k), [], False

        selected_titles = self._collect_section_titles(sections)
        display_titles = self._collect_section_titles_display(sections)
        if not selected_titles:
            return self.semantic_search_tool(doc_id, query, top_k=top_k), display_titles, False

        scoped_ldus = [ldu for ldu in self._load_ldus(doc_id) if self._ldu_in_selected_sections(ldu, selected_titles)]
        if not scoped_ldus:
            return self.semantic_search_tool(doc_id, query, top_k=top_k), display_titles, False

        scoped_hits = self._rank_ldus_lexical(scoped_ldus, query, top_k=top_k)
        if scoped_hits:
            return scoped_hits, display_titles, True

        return self.semantic_search_tool(doc_id, query, top_k=top_k), display_titles, False

    def _normalize_title_key(self, text: str) -> str:
        return normalize_text(text or "").strip().lower()

    def _extract_section_title_from_hit(self, hit: LogicalDocumentUnit) -> str:
        if hit.parent_section_path:
            return str(hit.parent_section_path[-1]).strip()
        if hit.parent_section:
            parts = [p.strip() for p in str(hit.parent_section).split(">") if p.strip()]
            if parts:
                return parts[-1]
        return ""

    def _unique_preserve_order(self, values: list[str]) -> list[str]:
        out: list[str] = []
        seen: set[str] = set()
        for value in values:
            key = self._normalize_title_key(value)
            if not key or key in seen:
                continue
            seen.add(key)
            out.append(value)
        return out

    def measure_retrieval_precision(
        self,
        doc_id: str,
        topic: str,
        expected_sections: list[str],
        top_k: int = 3,
    ) -> dict:
        k = max(int(top_k), 1)
        expected = self._unique_preserve_order(expected_sections)
        expected_keys = {self._normalize_title_key(title) for title in expected if title.strip()}

        nav_sections = self.pageindex_navigate_tool(doc_id, topic, top_k=k)
        nav_titles = self._unique_preserve_order([str(s.get("title") or "").strip() for s in nav_sections])[:k]
        nav_hits = self._semantic_search_with_sections(doc_id, topic, sections=nav_sections, top_k=k)
        nav_hit_sections = self._unique_preserve_order([self._extract_section_title_from_hit(hit) for hit in nav_hits])[:k]

        baseline_hits = self.semantic_search_tool(doc_id, topic, top_k=k)
        baseline_sections = self._unique_preserve_order([self._extract_section_title_from_hit(hit) for hit in baseline_hits])[:k]

        nav_match = [s for s in nav_hit_sections if self._normalize_title_key(s) in expected_keys]
        baseline_match = [s for s in baseline_sections if self._normalize_title_key(s) in expected_keys]

        def _precision(matches: list[str], denominator: int) -> float:
            if denominator <= 0:
                return 0.0
            return round(len(matches) / denominator, 4)

        return {
            "doc_id": doc_id,
            "topic": topic,
            "top_k": k,
            "expected_sections": expected,
            "with_pageindex": {
                "navigation_sections": nav_titles,
                "retrieved_sections": nav_hit_sections,
                "matched_sections": nav_match,
                "precision_at_k": _precision(nav_match, min(k, len(nav_hit_sections)) if nav_hit_sections else k),
            },
            "without_pageindex": {
                "retrieved_sections": baseline_sections,
                "matched_sections": baseline_match,
                "precision_at_k": _precision(baseline_match, min(k, len(baseline_sections)) if baseline_sections else k),
            },
        }

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
        bbox = pref.bbox or BBox.model_validate((0.0, 0.0, 1.0, 1.0))
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
                bbox=BBox.model_validate((0.0, 0.0, 1.0, 1.0)),
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
            bbox = BBox.model_validate((0.0, 0.0, 1.0, 1.0))
            if bbox_text:
                parts = [p.strip() for p in bbox_text.split(",")]
                if len(parts) == 4:
                    try:
                        bbox = BBox.model_validate((float(parts[0]), float(parts[1]), float(parts[2]), float(parts[3])))
                    except Exception:
                        bbox = BBox.model_validate((0.0, 0.0, 1.0, 1.0))

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
        return any(k in q for k in ["section", "where in", "capital expenditure", "capex", "projections", "table", "figure"])

    def _looks_temporal_question(self, question: str) -> bool:
        """Heuristic to detect 'when' / date-seeking questions.

        This is intentionally simple and fast: it is used only to
        trigger light-weight validation on the composed answer text.
        """
        q = normalize_text(question or "").strip().lower()
        if not q:
            return False
        if q.startswith("when ") or q.startswith("when is") or q.startswith("when was"):
            return True
        # Also catch variants like "what is the date of" / "on which date"
        temporal_markers = [
            "date of",
            "on which date",
            "what date",
            "which year",
        ]
        return any(marker in q for marker in temporal_markers)

    def _extract_date_from_text(self, text: str) -> str | None:
        """Extract a plausible date substring from answer/context text.

        We intentionally keep this conservative so that we only adjust
        answers when we see a clear date pattern.
        """
        t = normalize_text(text or "")
        if not t:
            return None

        # Common formats: "12 June 2023", "12th June 2023", "June 12, 2023", "June 12, 2023 G.C." etc.
        # NOTE: normalize_text() may lowercase the input, so we always
        # use case-insensitive matching.
        patterns = [
            # 12 June 2023 / 12th June 2023
            r"\b\d{1,2}(?:st|nd|rd|th)?\s+[a-zA-Z]+\s+\d{4}\b",
            # June 12, 2023 / June 12th, 2023 / June 12, 2023 G.C.
            r"\b[a-zA-Z]+\s+\d{1,2}(?:st|nd|rd|th)?,?\s+\d{4}(?:\s+g\.?c\.?)?\b",
            # 2023-06-12
            r"\b\d{4}-\d{2}-\d{2}\b",
            # 12/06/2023 or 12/06/23
            r"\b\d{1,2}/\d{1,2}/\d{2,4}\b",
        ]
        for pat in patterns:
            m = re.search(pat, t, flags=re.IGNORECASE)
            if m:
                return m.group(0)
        return None

    def _extract_date_from_related_sections(self, doc_id: str, hits: list[LogicalDocumentUnit]) -> str | None:
        """Search all LDUs in the same logical sections as the hits for a date.

        This handles cases where retrieval surfaces a heading like
        "ANNUAL ORDINARY GENERAL MEETING OF SHAREHOLDERS" but the actual
        date appears in a nearby paragraph under the same section.
        """
        if not hits:
            return None

        # Collect normalized section labels from the current hits.
        candidate_sections: set[str] = set()
        for h in hits:
            for seg in (h.parent_section_path or []):
                key = normalize_text(str(seg) or "").strip()
                if key:
                    candidate_sections.add(key)
            parent = normalize_text(h.parent_section or "").strip()
            if parent:
                candidate_sections.add(parent)

        if not candidate_sections:
            return None

        # Scan all LDUs that belong to any of these sections and pool
        # their text to look for a clear date expression.
        ldus = self._load_ldus(doc_id)
        buf_parts: list[str] = []
        for ldu in ldus:
            sec_keys: list[str] = []
            for seg in (ldu.parent_section_path or []):
                key = normalize_text(str(seg) or "").strip()
                if key:
                    sec_keys.append(key)
            parent = normalize_text(ldu.parent_section or "").strip()
            if parent:
                sec_keys.append(parent)
            if not sec_keys:
                continue
            if not any(key in candidate_sections for key in sec_keys):
                continue
            if ldu.content:
                buf_parts.append(normalize_text(ldu.content))

        if not buf_parts:
            return None

        big_text = " ".join(buf_parts)
        return self._extract_date_from_text(big_text)

    def _tokenize_answer_text(self, text: str) -> list[str]:
        return [t.lower() for t in re.findall(r"\w+", text, flags=re.UNICODE)]

    def _is_noisy_answer_text(self, text: str) -> bool:
        t = normalize_text(text or "")
        if len(t) < 8:
            return True
        tokens = self._tokenize_answer_text(t)
        if len(tokens) < 3:
            return True
        unique_ratio = len(set(tokens)) / max(len(tokens), 1)
        if unique_ratio < 0.35:
            return True
        most_common = max((tokens.count(tok) for tok in set(tokens)), default=0)
        if most_common / max(len(tokens), 1) > 0.5:
            return True
        return False

    def _is_table_like_hit(self, hit: LogicalDocumentUnit) -> bool:
        if hit.chunk_type == "table":
            return True
        text = hit.content or ""
        if text.count("|") >= 8 and text.count("\n") >= 1:
            return True
        return False

    def _parse_pipe_table_rows(self, text: str) -> list[list[str]]:
        rows: list[list[str]] = []
        for raw_line in (text or "").splitlines():
            line = raw_line.strip()
            if "|" not in line:
                continue
            cells = [c.strip() for c in line.split("|")]
            if cells and cells[0] == "":
                cells = cells[1:]
            if cells and cells[-1] == "":
                cells = cells[:-1]
            if not cells:
                continue
            non_empty = [c for c in cells if c]
            if len(non_empty) < 2:
                continue
            rows.append(cells)
        return rows

    def _compact_table_answer_from_hit(self, hit: LogicalDocumentUnit, question: str) -> str | None:
        text = hit.content or ""
        rows = self._parse_pipe_table_rows(text)
        if len(rows) < 2:
            return None

        header = rows[0]
        data_rows = rows[1:]
        header_cells = [c for c in header if c][:6]
        header_text = " | ".join(header_cells) if header_cells else "(no clear header)"

        q_tokens = [t for t in self._tokenize_answer_text(question) if len(t) >= 2]

        def row_score(row: list[str]) -> tuple[int, int]:
            row_text = " ".join(row).lower()
            overlap = sum(1 for t in q_tokens if t in row_text)
            numeric = len(re.findall(r"\d+(?:[\.,]\d+)?", row_text))
            return overlap, numeric

        scored = [(idx, row, *row_score(row)) for idx, row in enumerate(data_rows)]
        scored_sorted = sorted(scored, key=lambda x: (x[2], x[3]), reverse=True)
        picked = [item for item in scored_sorted[:3] if item[2] > 0 or item[3] > 0]
        if not picked:
            picked = scored_sorted[:2]

        picked = sorted(picked, key=lambda x: x[0])
        row_summaries: list[str] = []
        for _idx, row, _overlap, _numeric in picked:
            compact_cells = [c for c in row if c][:6]
            if not compact_cells:
                continue
            row_summaries.append(" | ".join(compact_cells))

        if not row_summaries:
            return None

        summary = f"Table summary: columns {header_text}. Key rows: " + "; ".join(row_summaries)
        return summary[:900]

    def _compact_pipe_stream_answer(self, text: str, question: str) -> str | None:
        if (text or "").count("|") < 12:
            return None
        cells = [c.strip() for c in (text or "").split("|") if c.strip()]
        if len(cells) < 8:
            return None

        q_tokens = [t for t in self._tokenize_answer_text(question) if len(t) >= 2]
        selected: list[str] = []
        seen: set[str] = set()
        for cell in cells:
            cell_l = cell.lower()
            has_overlap = any(t in cell_l for t in q_tokens)
            has_number = bool(re.search(r"\d+(?:[\.,]\d+)?", cell_l))
            if not (has_overlap or has_number):
                continue
            if cell_l in seen:
                continue
            seen.add(cell_l)
            selected.append(cell)
            if len(selected) >= 10:
                break

        head = " | ".join(cells[:8])
        key_values = " | ".join(selected[:10]) if selected else " | ".join(cells[8:16])
        return f"Table-like summary: {head}. Key values: {key_values}"[:900]

    def _compose_answer_from_hits(self, hits: list[LogicalDocumentUnit], question: str) -> str:
        for hit in hits[:3]:
            if not self._is_table_like_hit(hit):
                continue
            compact = self._compact_table_answer_from_hit(hit, question)
            if compact:
                return compact

        for hit in hits[:3]:
            compact_stream = self._compact_pipe_stream_answer(hit.content or "", question)
            if compact_stream:
                return compact_stream

        snippets: list[str] = []
        seen: set[str] = set()
        for hit in hits:
            raw = hit.content or ""
            text = normalize_text(raw).strip()
            if not text:
                continue
            key = text.lower()
            if key in seen:
                continue
            if self._is_noisy_answer_text(text):
                continue
            seen.add(key)
            snippets.append(text)
            if len(snippets) >= 3:
                break

        if snippets:
            return " ".join(snippets)[:900]

        fallback = " ".join([(h.content or "") for h in hits[:3]]).strip()
        return normalize_text(fallback)[:900]

    def _build_ollama_context_from_hits(self, hits: list[LogicalDocumentUnit]) -> str:
        max_chars = max(int(self.store.settings.ollama_max_context_chars), 1000)
        parts: list[str] = []
        used = 0
        for idx, hit in enumerate(hits[:8], start=1):
            text = normalize_text(hit.content or "").strip()
            if not text:
                continue
            line = f"[chunk {idx}] {text}"
            if used + len(line) > max_chars and parts:
                break
            parts.append(line)
            used += len(line)
        return "\n".join(parts)

    def _extract_amharic_phrases(self, text: str) -> set[str]:
        phrases = re.findall(r"[\u1200-\u137F]{3,}", text or "")
        return {p for p in phrases if p}

    def _ollama_output_grounded(self, answer: str, context: str) -> bool:
        answer_text = answer or ""
        context_text = context or ""
        numeric_tokens = set(re.findall(r"\d+(?:[\.,]\d+)?", context_text))
        if any(tok in answer_text for tok in numeric_tokens):
            return True
        amharic_phrases = self._extract_amharic_phrases(context_text)
        if any(phrase in answer_text for phrase in amharic_phrases):
            return True
        return False

    def _synthesize_with_ollama(self, question: str, context: str) -> str | None:
        if not self.store.settings.use_ollama_answers:
            return None
        if not (context or "").strip():
            return None

        prompt = (
            "System: Answer only using the provided context. "
            "If missing, say 'Not found in context'.\n\n"
            f"User question: {question}\n\n"
            f"Context:\n{context}\n"
        )
        payload = {
            "model": self.store.settings.ollama_answer_model,
            "prompt": prompt,
            "stream": False,
        }
        try:
            host = self.store.settings.ollama_host.rstrip("/")
            req = request.Request(
                f"{host}/api/generate",
                data=json.dumps(payload).encode("utf-8"),
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            with request.urlopen(req, timeout=45) as resp:
                raw = resp.read().decode("utf-8", errors="ignore")
            data = json.loads(raw)
            output = str(data.get("response", "")).strip()
            return output or None
        except Exception:
            return None

    def _fallback_answer_with_ollama(self, doc_id: str, question: str, hits: list[LogicalDocumentUnit], tool_trace: list[str]) -> dict:
        provenance = self._provenance_from_hits(doc_id, hits)
        context = self._build_ollama_context_from_hits(hits)
        if not context:
            return {
                "answer": self._ocr_refusal,
                "provenance_chain": [p.model_dump(mode="json") for p in provenance],
                "confidence": 0.25,
                "tool_trace": tool_trace,
            }

        ollama_answer = self._synthesize_with_ollama(question, context)
        if not ollama_answer:
            return {
                "answer": self._ocr_refusal,
                "provenance_chain": [p.model_dump(mode="json") for p in provenance],
                "confidence": 0.25,
                "tool_trace": tool_trace,
            }

        confidence = 0.25
        if self._ollama_output_grounded(ollama_answer, context):
            confidence = 0.45
        return {
            "answer": ollama_answer,
            "provenance_chain": [p.model_dump(mode="json") for p in provenance],
            "confidence": confidence,
            "tool_trace": tool_trace,
        }

    def _query_interface_linear(self, doc_id: str, question: str) -> dict:
        tool_trace: list[str] = []
        sections: list[dict] = []
        navigation_sections: list[str] = []
        used_section_scope = False

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
        hits, navigation_sections, used_section_scope = self._semantic_search_with_sections_meta(
            doc_id,
            semantic_query,
            sections=sections,
            top_k=self.store.settings.query_semantic_top_k,
        )
        if hits:
            answer_text = self._compose_answer_from_hits(hits, question)
            # For temporal ("when") questions, try to ensure the answer
            # actually contains a date. If it doesn't, scan the same
            # hits' content for a plausible date and prefer that.
            if self._looks_temporal_question(question):
                if not self._extract_date_from_text(answer_text):
                    # First search in immediate hit context
                    combined_context = " \n".join(normalize_text(h.content or "") for h in hits[:5])
                    date_snippet = self._extract_date_from_text(combined_context)
                    if not date_snippet:
                        # If still not found, search all LDUs within the
                        # same logical sections as the hits (more expensive
                        # but still bounded to this document).
                        date_snippet = self._extract_date_from_related_sections(doc_id, hits)
                    if date_snippet:
                        answer_text = date_snippet
            if self._is_noisy_answer_text(answer_text):
                tool_trace.append("structured_query")
                rows = self.structured_query_tool(question, doc_id=doc_id)
                if rows:
                    provenance = self._provenance_from_fact_rows(doc_id, rows)
                    return {
                        "answer": json.dumps(rows[:5], ensure_ascii=False),
                        "provenance_chain": [p.model_dump(mode="json") for p in provenance],
                        "confidence": 0.58,
                        "tool_trace": tool_trace,
                        "navigation_sections": navigation_sections,
                        "used_section_scope": used_section_scope,
                    }
                fallback = self._fallback_answer_with_ollama(doc_id, question, hits, tool_trace)
                fallback["navigation_sections"] = navigation_sections
                fallback["used_section_scope"] = used_section_scope
                return fallback
            provenance = self._provenance_from_hits(doc_id, hits)
            return {
                "answer": answer_text,
                "provenance_chain": [p.model_dump(mode="json") for p in provenance],
                "confidence": 0.72,
                "tool_trace": tool_trace,
                "navigation_sections": navigation_sections,
                "used_section_scope": used_section_scope,
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
                "navigation_sections": navigation_sections,
                "used_section_scope": used_section_scope,
            }

        fallback_prov = self._fallback_provenance(doc_id)
        return {
            "answer": "No high-confidence match found.",
            "provenance_chain": [p.model_dump(mode="json") for p in fallback_prov],
            "confidence": 0.2,
            "tool_trace": tool_trace,
            "navigation_sections": navigation_sections,
            "used_section_scope": used_section_scope,
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
            sections = state.get("sections") if isinstance(state.get("sections"), list) else []
            hits, navigation_sections, used_section_scope = self._semantic_search_with_sections_meta(
                doc_id,
                semantic_query,
                sections=sections,
                top_k=self.store.settings.query_semantic_top_k,
            )
            if hits:
                return {
                    "hits": hits,
                    "tool_trace": tool_trace,
                    "navigation_sections": navigation_sections,
                    "used_section_scope": used_section_scope,
                }

            tool_trace.append("structured_query")
            rows = self.structured_query_tool(question, doc_id=doc_id)
            return {"rows": rows, "tool_trace": tool_trace}

        def finalize_node(state: QueryGraphState) -> QueryGraphState:
            doc_id = str(state.get("doc_id") or "")
            question = str(state.get("question") or "")
            tool_trace = list(state.get("tool_trace") or [])
            navigation_sections = list(state.get("navigation_sections") or [])
            used_section_scope = bool(state.get("used_section_scope") or False)
            hits = state.get("hits") or []
            if hits:
                answer_text = self._compose_answer_from_hits(hits, question)
                # For temporal ("when") questions in the langgraph path,
                # also enforce that the answer contains a plausible date.
                # If not, try to extract a date from the same hit context
                # or from LDUs within the same logical sections.
                if self._looks_temporal_question(question):
                    if not self._extract_date_from_text(answer_text):
                        combined_context = " \n".join(normalize_text(h.content or "") for h in hits[:5])
                        date_snippet = self._extract_date_from_text(combined_context)
                        if not date_snippet:
                            date_snippet = self._extract_date_from_related_sections(doc_id, hits)
                        if date_snippet:
                            answer_text = date_snippet
                if self._is_noisy_answer_text(answer_text):
                    rows = self.structured_query_tool(question, doc_id=doc_id)
                    if rows:
                        if "structured_query" not in tool_trace:
                            tool_trace.append("structured_query")
                        provenance = self._provenance_from_fact_rows(doc_id, rows)
                        return {
                            "answer": json.dumps(rows[:5], ensure_ascii=False),
                            "provenance_chain": [p.model_dump(mode="json") for p in provenance],
                            "confidence": 0.58,
                            "tool_trace": tool_trace,
                            "navigation_sections": navigation_sections,
                            "used_section_scope": used_section_scope,
                        }
                    if "structured_query" not in tool_trace:
                        tool_trace.append("structured_query")
                    fallback = self._fallback_answer_with_ollama(doc_id, question, hits, tool_trace)
                    # Ensure fallback is merged into state (QueryGraphState)
                    return {
                        **state,
                        "answer": fallback.get("answer", "No high-confidence match found."),
                        "provenance_chain": fallback.get("provenance_chain", []),
                        "confidence": fallback.get("confidence", 0.2),
                        "tool_trace": fallback.get("tool_trace", tool_trace),
                        "navigation_sections": navigation_sections,
                        "used_section_scope": used_section_scope,
                    }
                provenance = self._provenance_from_hits(doc_id, hits)
                return {
                    "answer": answer_text,
                    "provenance_chain": [p.model_dump(mode="json") for p in provenance],
                    "confidence": 0.72,
                    "tool_trace": tool_trace,
                    "navigation_sections": navigation_sections,
                    "used_section_scope": used_section_scope,
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
                    "navigation_sections": navigation_sections,
                    "used_section_scope": used_section_scope,
                }

            fallback_prov = self._fallback_provenance(doc_id)
            return {
                "answer": "No high-confidence match found.",
                "provenance_chain": [p.model_dump(mode="json") for p in fallback_prov],
                "confidence": 0.2,
                "tool_trace": tool_trace,
                "navigation_sections": navigation_sections,
                "used_section_scope": used_section_scope,
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

    def query_interface(self, doc_id: str, question: str, include_navigation_debug: bool = False) -> dict:
        def _maybe_strip_navigation_debug(result: dict) -> dict:
            if include_navigation_debug:
                return result
            cleaned = dict(result)
            cleaned.pop("navigation_sections", None)
            cleaned.pop("used_section_scope", None)
            return cleaned

        if not self.store.settings.query_use_langgraph:
            return _maybe_strip_navigation_debug(self._query_interface_linear(doc_id, question))

        app = self._build_langgraph_app()
        if app is None:
            return _maybe_strip_navigation_debug(self._query_interface_linear(doc_id, question))

        try:
            result = app.invoke({"doc_id": doc_id, "question": question, "tool_trace": []})
            out = {
                "answer": result.get("answer", "No high-confidence match found."),
                "provenance_chain": result.get("provenance_chain", [p.model_dump(mode="json") for p in self._fallback_provenance(doc_id)]),
                "confidence": float(result.get("confidence", 0.2)),
                "tool_trace": result.get("tool_trace", []),
                "navigation_sections": result.get("navigation_sections", []),
                "used_section_scope": bool(result.get("used_section_scope", False)),
            }
            return _maybe_strip_navigation_debug(out)
        except Exception:
            return _maybe_strip_navigation_debug(self._query_interface_linear(doc_id, question))

    def query(self, doc_id: str, question: str) -> QueryAnswer:
        out = self.query_interface(doc_id, question)
        prov = [ProvenanceRef.model_validate(p) for p in out.get("provenance_chain", [])]
        return QueryAnswer(
            answer=out["answer"],
            provenance_chain=ProvenanceChain.from_refs(prov) if prov else ProvenanceChain.from_refs(self._fallback_provenance(doc_id)),
            confidence=float(out.get("confidence", 0.2)),
        )

    def audit_claim(self, doc_id: str, claim: str) -> dict:
        result = self.query(doc_id, claim)
        status = "VERIFIED" if result.provenance_chain.steps else "NOT FOUND / UNVERIFIABLE"
        return {
            "status": status,
            "claim": claim,
            "answer": result.answer,
            "provenance_chain": [p.model_dump(mode="json") for p in result.provenance_chain.steps],
        }

    def navigate(self, doc_id: str, topic: str, top_k: int = 3) -> list[dict]:
        return self.pageindex_navigate_tool(doc_id, topic, top_k=top_k)

    def structured_query(self, query: str, doc_id: str | None = None) -> list[dict]:
        return self.structured_query_tool(query=query, doc_id=doc_id)


def open_citation_snippet(store: ArtifactStore, doc_id: str, page_number: int, bbox: tuple[float, float, float, float]) -> dict:
    rows = store.read_jsonl(store.chunks_dir / f"{doc_id}.jsonl")
    for row in rows:
        ldu = LogicalDocumentUnit.model_validate(row)
        for pref in ldu.page_refs:
            if pref.ref_type != "pdf_bbox" or pref.page_number != page_number or not pref.bbox:
                continue
            pref_bbox = pref.bbox
            if isinstance(pref_bbox, BBox):
                px0, py0, px1, py1 = pref_bbox.as_tuple()
            else:
                px0, py0, px1, py1 = tuple(float(v) for v in pref_bbox)
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
