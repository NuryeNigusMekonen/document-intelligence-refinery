from __future__ import annotations

import logging
import re
import time
from dataclasses import dataclass
from typing import Sequence

from models import BBox, ExtractedDocument, LogicalDocumentUnit, ProvenanceChain, ProvenanceRef, TableObject, TextBlock
from refinery.config import Settings
from refinery.runtime_rules import load_runtime_rules
from refinery.storage import ArtifactStore
from refinery.utils import deterministic_id, normalize_text, sha256_text, token_count

logger = logging.getLogger(__name__)


ETHIOPIC_RE = re.compile(r"[\u1200-\u137F]")


@dataclass
class _ResolvableRef:
    ldu_index: int
    ref_type: str
    ordinal: int


class ChunkValidator:
    def __init__(
        self,
        max_tokens: int,
        list_item_re: re.Pattern[str] | None = None,
        list_split_item_boundary_only: bool = True,
    ):
        self.max_tokens = max_tokens
        self.list_item_re = list_item_re or re.compile(r"^\s*(\d+\.|[-*])\s+")
        self.list_split_item_boundary_only = list_split_item_boundary_only

    def validate(self, ldu: LogicalDocumentUnit) -> None:
        content = (ldu.content or "").strip()
        if not content:
            raise ValueError("LDU content is required")
        if not ldu.page_refs:
            raise ValueError("LDU page_refs are required")
        if ldu.bounding_box is None:
            raise ValueError("LDU bounding_box is required")
        if not ldu.content_hash:
            raise ValueError("LDU content_hash is required")
        if token_count(content) != ldu.token_count:
            raise ValueError("LDU token_count must match content token count")

        if ldu.chunk_type in {"paragraph", "list"} and ldu.token_count > self.max_tokens:
            raise ValueError(f"{ldu.chunk_type} chunk exceeds max_tokens={self.max_tokens}")

        for ref in ldu.page_refs:
            if ref.content_hash != ldu.content_hash:
                raise ValueError("LDU/page_refs content_hash mismatch")

        if ldu.chunk_type == "table":
            payload = ldu.structured_payload or {}
            if not payload.get("headers"):
                raise ValueError("Table chunk must contain headers")
            headers = payload.get("headers", [])
            rows = payload.get("rows", [])
            if not isinstance(headers, list) or not isinstance(rows, list):
                raise ValueError("Table chunk payload must contain list headers and rows")
            for row in rows:
                if not isinstance(row, list):
                    raise ValueError("Table chunk rows must be a list of row lists")
                if len(row) > len(headers):
                    raise ValueError("Table row has more cells than headers")

        if ldu.chunk_type == "figure":
            payload = ldu.structured_payload or {}
            if "caption" not in payload:
                raise ValueError("Figure chunk must include caption metadata key")

        if ldu.chunk_type == "list" and ldu.token_count > self.max_tokens and self.list_split_item_boundary_only:
            lines = [ln for ln in content.splitlines() if ln.strip()]
            if any(not self.list_item_re.match(ln) for ln in lines):
                raise ValueError("List split violated item boundary rule")

        if ldu.chunk_type == "list":
            lines = [ln for ln in content.splitlines() if ln.strip()]
            if lines and any(not self.list_item_re.match(ln) for ln in lines):
                raise ValueError("List chunk contains non-list item line")

    def validate_document(self, ldus: list[LogicalDocumentUnit]) -> None:
        for ldu in ldus:
            self.validate(ldu)

        for ldu in ldus:
            refs = [r for r in ldu.relationships if r.startswith("cross_ref:")]
            for rel in refs:
                ref_type = rel.split(":", 1)[1].strip().lower()
                has_resolution = any(
                    r.startswith(f"resolved_ref:{ref_type}:") or r.startswith(f"unresolved_ref:{ref_type}:")
                    for r in ldu.relationships
                )
                if not has_resolution:
                    raise ValueError(f"Cross-reference '{ref_type}' missing resolution relationship")


class ChunkingEngine:
    def __init__(self, settings: Settings, store: ArtifactStore):
        self.settings = settings
        self.store = store
        self.rules = load_runtime_rules(settings)
        chunking_rules = self.rules.get("chunking", {}) if isinstance(self.rules.get("chunking", {}), dict) else {}
        configured_max_tokens = int(chunking_rules.get("max_tokens", settings.chunk_max_tokens))
        self.max_tokens = settings.chunk_max_tokens if settings.chunk_max_tokens != 350 else configured_max_tokens
        self.list_split_item_boundary_only = bool(chunking_rules.get("list_split_item_boundary_only", True))
        heading_pattern = str(chunking_rules.get("heading_regex", r"^(\d+(\.\d+)*)\s+.+|^[A-Z][A-Za-z\s]{3,60}$"))
        list_item_pattern = str(chunking_rules.get("list_item_regex", r"^\s*(\d+\.|[-*])\s+"))
        cross_ref_pattern = str(chunking_rules.get("cross_ref_regex", r"see\s+(table|figure|section)\s+\d+"))
        resolvable_ref_pattern = str(chunking_rules.get("resolvable_ref_regex", r"see\s+(table|figure|section)\s+(\d+)"))
        self.heading_re = re.compile(heading_pattern)
        self.list_item_re = re.compile(list_item_pattern)
        self.cross_ref_re = re.compile(cross_ref_pattern, re.IGNORECASE)
        self.resolvable_ref_re = re.compile(resolvable_ref_pattern, re.IGNORECASE)
        self.table_block_overlap_suppress_ratio = float(chunking_rules.get("table_block_overlap_suppress_ratio", 0.60))
        self.table_block_suppress_ethiopic_min_chars = int(chunking_rules.get("table_block_suppress_ethiopic_min_chars", 3))
        self.validator = ChunkValidator(
            max_tokens=self.max_tokens,
            list_item_re=self.list_item_re,
            list_split_item_boundary_only=self.list_split_item_boundary_only,
        )

    def _section_label(self, section_path: list[str]) -> str | None:
        return " > ".join(section_path) if section_path else None

    def _effective_section_path(self, fallback: list[str], provenance: ProvenanceRef | None) -> list[str]:
        if provenance and provenance.section_path:
            cleaned = [segment.strip() for segment in provenance.section_path if segment and segment.strip()]
            if cleaned:
                return cleaned
        return fallback.copy()

    def _is_list_item(self, text: str) -> bool:
        normalized = text.strip()
        if normalized.lower().startswith("[list]"):
            return True
        return bool(self.list_item_re.match(normalized))

    def _strip_list_prefix(self, text: str) -> str:
        normalized = text.strip()
        if normalized.lower().startswith("[list]"):
            return normalized[6:].strip()
        return normalized

    def _is_header_fragment_text(self, text: str) -> bool:
        """
        Heuristic for short header fragments that often appear split across lines in scanned PDFs,
        e.g., "Deferred" on one line and "tax liability" on the next.

        Accepts 1-6 tokens, rejects lines that end with sentence punctuation, numeric-only, or look like list items.
        """
        normalized = normalize_text(text or "")
        if not normalized:
            return False
        # Reject obvious list markers
        if self._is_list_item(normalized):
            return False
        # Reject if it looks like a complete sentence end
        if normalized.endswith((".", "?", "!", ";", ":")):
            return False
        # Tokenize on words incl. Ethiopic range
        tokens = re.findall(r"[\w\u1200-\u137F]+", normalized, flags=re.UNICODE)
        n_tokens = len(tokens)
        if n_tokens < 1 or n_tokens > 6:
            return False
        # Reject if majority numeric
        digits = sum(ch.isdigit() for ch in normalized)
        letters = sum(ch.isalpha() for ch in normalized)
        if digits > 0 and digits >= letters:
            return False
        # Avoid obvious table header lines
        if "|" in normalized:
            return False
        return True

    def _split_paragraph_if_needed(self, text: str) -> list[str]:
        """
        Split oversized paragraph text into multiple parts that each respect the
        max token budget. Prefer sentence boundaries; if a single sentence still
        exceeds the budget, fall back to whitespace-based hard splits.
        """
        if token_count(text) <= self.max_tokens:
            return [text]

        # First try sentence-aware splitting (supports Latin and Ethiopic full stop ።)
        # Keep delimiters by splitting on boundary whitespace after punctuation.
        sentences = [s.strip() for s in re.split(r"(?<=[\.\?\!።])\s+", text) if s and s.strip()]
        if not sentences:
            sentences = [text]

        parts: list[str] = []
        current: list[str] = []
        for sent in sentences:
            candidate = (" ".join(current + [sent])).strip()
            if token_count(candidate) > self.max_tokens and current:
                parts.append(" ".join(current).strip())
                current = [sent]
            else:
                current.append(sent)
        if current:
            parts.append(" ".join(current).strip())

        # Ensure each part fits by falling back to hard token-based splitting
        ensured: list[str] = []
        for p in parts:
            if token_count(p) <= self.max_tokens:
                ensured.append(p)
                continue
            # Hard split by whitespace chunks while respecting token budget
            chunks: list[str] = []
            acc: list[str] = []
            for piece in re.findall(r"\S+\s*", p):
                candidate = ("".join(acc + [piece])).strip()
                if token_count(candidate) > self.max_tokens and acc:
                    chunks.append("".join(acc).strip())
                    acc = [piece]
                else:
                    acc.append(piece)
            if acc:
                chunks.append("".join(acc).strip())
            ensured.extend([c for c in chunks if c])

        return ensured or [text]

    def _to_bbox(self, bbox: BBox | tuple[float, float, float, float]) -> BBox:
        if isinstance(bbox, BBox):
            return bbox
        return BBox.model_validate(bbox)

    def _merge_bboxes(self, bboxes: Sequence[BBox | tuple[float, float, float, float]]) -> BBox | None:
        normalized = [self._to_bbox(b) for b in bboxes]
        if not normalized:
            return None
        return BBox(
            x0=min(b.x0 for b in normalized),
            y0=min(b.y0 for b in normalized),
            x1=max(b.x1 for b in normalized),
            y1=max(b.y1 for b in normalized),
        )

    def _ref_for_payload(
        self,
        extracted: ExtractedDocument,
        page_number: int,
        bbox: BBox | tuple[float, float, float, float],
        section_path: list[str],
        provenance: ProvenanceRef | None,
    ) -> ProvenanceRef:
        if provenance is not None:
            cloned = provenance.model_copy(deep=True)
            if section_path and not cloned.section_path:
                cloned.section_path = section_path.copy()
            return cloned
        return ProvenanceRef(
            doc_name=extracted.doc_name,
            ref_type="pdf_bbox",
            page_number=page_number,
            bbox=bbox,
            section_path=section_path.copy() if section_path else None,
            content_hash="pending",
        )

    def _bbox_from_refs(self, page_refs: list[ProvenanceRef]) -> BBox | None:
        for ref in page_refs:
            if ref.bbox is not None:
                return self._to_bbox(ref.bbox)
        return None

    def _hash_with_provenance(self, content: str, page_refs: list[ProvenanceRef]) -> str:
        def _q(value: float) -> int:
            return int(round(float(value), 1) * 10)

        def _bbox_signature(pref: ProvenanceRef) -> str:
            if pref.bbox is None:
                return "none"
            bb = self._to_bbox(pref.bbox)
            return f"{_q(bb.x0)}:{_q(bb.y0)}:{_q(bb.x1)}:{_q(bb.y1)}"

        signatures: list[str] = []
        for pref in page_refs:
            signatures.append(
                "::".join(
                    [
                        str(pref.ref_type or ""),
                        _bbox_signature(pref),
                        ">".join(pref.section_path or []),
                        f"{pref.line_range[0]}-{pref.line_range[1]}" if pref.line_range else "",
                        str(pref.sheet_name or ""),
                        str(pref.cell_range or ""),
                    ]
                )
            )
        prov = "|".join(signatures)
        return sha256_text(normalize_text(content) + "||" + prov)

    def _make_table_ldu(
        self,
        extracted: ExtractedDocument,
        page_number: int,
        table: TableObject,
        section_path: list[str],
    ) -> LogicalDocumentUnit | None:
        """
        Build a table LDU. Returns None when the table has no meaningful textual content
        (e.g., OCR returned empty headers/cells), so the caller can safely skip it.
        """
        # Guard against empty/garbage OCR results for tables
        headers = list(table.headers or [])
        rows = list(table.rows or [])

        has_any_header_text = any((h or "").strip() for h in headers)
        has_any_cell_text = any(any((c or "").strip() for c in (row or [])) for row in rows)

        # The validator requires headers to exist. If there's no header text at all, skip.
        if not has_any_header_text and not has_any_cell_text:
            return None
        if not has_any_header_text:
            return None

        # Build a stable textual representation, ignoring fully-empty rows
        header_line = " | ".join(headers)
        row_lines = [
            " | ".join(row)
            for row in rows
            if any((c or "").strip() for c in row)
        ]
        content = header_line if not row_lines else header_line + "\n" + "\n".join(row_lines)

        if not (content or "").strip():
            return None

        payload = {"headers": headers, "rows": rows}
        effective_section = self._effective_section_path(section_path, table.provenance)
        page_refs = [self._ref_for_payload(extracted, page_number, table.bbox, effective_section, table.provenance)]
        content_hash = self._hash_with_provenance(content, page_refs)
        for ref in page_refs:
            ref.content_hash = content_hash
        ldu = LogicalDocumentUnit(
            ldu_id=deterministic_id(
                "ldu",
                {"doc_id": extracted.doc_id, "type": "table", "page": page_number, "hash": content_hash},
            ),
            chunk_type="table",
            content=content,
            structured_payload=payload,
            token_count=token_count(content),
            bounding_box=table.bbox,
            parent_section=self._section_label(effective_section),
            parent_section_path=effective_section,
            page_refs=page_refs,
            provenance_chain=ProvenanceChain.from_refs(page_refs),
            content_hash=content_hash,
        )
        self.validator.validate(ldu)
        return ldu

    def _resolve_cross_references(self, ldus: list[LogicalDocumentUnit], pending: list[_ResolvableRef]) -> None:
        table_ids = [ldu.ldu_id for ldu in ldus if ldu.chunk_type == "table"]
        figure_ids = [ldu.ldu_id for ldu in ldus if ldu.chunk_type == "figure"]
        section_ids = [ldu.ldu_id for ldu in ldus if ldu.chunk_type in {"paragraph", "list", "section_summary"}]

        for ref in pending:
            source = ldus[ref.ldu_index]
            targets: list[str]
            if ref.ref_type == "table":
                targets = table_ids
            elif ref.ref_type == "figure":
                targets = figure_ids
            else:
                targets = section_ids
            target_idx = ref.ordinal - 1
            if 0 <= target_idx < len(targets):
                source.relationships.append(f"resolved_ref:{ref.ref_type}:{ref.ordinal}->{targets[target_idx]}")
            else:
                source.relationships.append(f"unresolved_ref:{ref.ref_type}:{ref.ordinal}")

    def _split_list_if_needed(self, text: str) -> list[str]:
        if token_count(text) <= self.max_tokens:
            return [text]
        if not self.list_split_item_boundary_only:
            chunks: list[str] = []
            current: list[str] = []
            for ln in [ln for ln in text.splitlines() if ln.strip()]:
                candidate = "\n".join(current + [ln])
                if token_count(candidate) > self.max_tokens and current:
                    chunks.append("\n".join(current))
                    current = [ln]
                else:
                    current.append(ln)
            if current:
                chunks.append("\n".join(current))
            return chunks or [text]

        items = [ln for ln in text.splitlines() if self.list_item_re.match(ln)]
        inline_item_re = re.compile(r"(?:^|\s)(?:\d+\.|[-*])\s+")
        inline_matches = list(inline_item_re.finditer(text))
        if (not items) or (len(items) == 1 and len(inline_matches) >= 2):
            if len(inline_matches) >= 2:
                extracted_items: list[str] = []
                for idx, match in enumerate(inline_matches):
                    start = match.start()
                    end = inline_matches[idx + 1].start() if idx + 1 < len(inline_matches) else len(text)
                    item_text = text[start:end].strip()
                    if item_text:
                        extracted_items.append(item_text)
                items = extracted_items
        if not items:
            return [text]
        chunks: list[str] = []
        current: list[str] = []
        for item in items:
            candidate = "\n".join(current + [item])
            if token_count(candidate) > self.max_tokens and current:
                chunks.append("\n".join(current))
                current = [item]
            else:
                current.append(item)
        if current:
            chunks.append("\n".join(current))
        return chunks

    def _bbox_overlap_ratio(self, a: BBox, b: BBox) -> float:
        ax0, ay0, ax1, ay1 = a.as_tuple()
        bx0, by0, bx1, by1 = b.as_tuple()
        inter_w = max(0.0, min(ax1, bx1) - max(ax0, bx0))
        inter_h = max(0.0, min(ay1, by1) - max(ay0, by0))
        inter_area = inter_w * inter_h
        area_a = max((ax1 - ax0) * (ay1 - ay0), 1e-6)
        return inter_area / area_a

    def _should_suppress_block_for_table(self, block_text: str, block_bbox: BBox, table_bboxes: Sequence[BBox]) -> bool:
        if not table_bboxes:
            return False
        max_overlap = max((self._bbox_overlap_ratio(block_bbox, tb) for tb in table_bboxes), default=0.0)
        if max_overlap < self.table_block_overlap_suppress_ratio:
            return False
        ethiopic_chars = len(ETHIOPIC_RE.findall(block_text or ""))
        if ethiopic_chars >= self.table_block_suppress_ethiopic_min_chars:
            return False
        return True

    def run(self, extracted: ExtractedDocument) -> list[LogicalDocumentUnit]:
        started = time.perf_counter()
        logger.info("stage=chunking start doc=%s", extracted.doc_name)
        section_path: list[str] = []
        ldus: list[LogicalDocumentUnit] = []
        pending_resolved_refs: list[_ResolvableRef] = []

        for page in extracted.pages:
            table_bboxes = [self._to_bbox(t.bbox) for t in page.tables if t.bbox is not None]
            sorted_blocks = sorted(page.blocks, key=lambda b: b.reading_order)
            idx = 0
            while idx < len(sorted_blocks):
                block = sorted_blocks[idx]
                text = normalize_text(block.text)
                if not text:
                    idx += 1
                    continue
                if self._should_suppress_block_for_table(text, self._to_bbox(block.bbox), table_bboxes):
                    idx += 1
                    continue

                section_path = self._effective_section_path(section_path, block.provenance)

                if self.heading_re.match(text) and not self._is_list_item(text):
                    # Stitch adjacent short header fragments into a single title when present.
                    title_parts: list[str] = [text]
                    j = idx + 1
                    while j < len(sorted_blocks):
                        cand_block = sorted_blocks[j]
                        cand_text = normalize_text(cand_block.text)
                        if not cand_text:
                            j += 1
                            continue
                        # Skip blocks that should be suppressed due to table overlap
                        if self._should_suppress_block_for_table(cand_text, self._to_bbox(cand_block.bbox), table_bboxes):
                            j += 1
                            continue
                        # Accept next line if it looks like a short header fragment
                        if self._is_header_fragment_text(cand_text):
                            # Tentatively append and check overall token budget for titles
                            tentative = re.sub(r"\s+", " ", (" ".join(title_parts + [cand_text])).strip())
                            tokens = re.findall(r"[\w\u1200-\u137F]+", tentative, flags=re.UNICODE)
                            if 2 <= len(tokens) <= 12:
                                title_parts.append(cand_text)
                                j += 1
                                continue
                        break
                    combined_title = re.sub(r"\s+", " ", (" ".join(title_parts)).strip())
                    section_path = [combined_title]
                    # Consume stitched header lines
                    idx = j
                    continue

                chunk_type = "list" if self._is_list_item(text) else "paragraph"
                if chunk_type == "list":
                    list_text_lines: list[str] = []
                    list_blocks: list[TextBlock] = []
                    list_section_path = section_path.copy()
                    while idx < len(sorted_blocks):
                        candidate_block = sorted_blocks[idx]
                        candidate_text = normalize_text(candidate_block.text)
                        if not candidate_text:
                            idx += 1
                            continue
                        if self._should_suppress_block_for_table(candidate_text, self._to_bbox(candidate_block.bbox), table_bboxes):
                            idx += 1
                            continue
                        candidate_section = self._effective_section_path(section_path, candidate_block.provenance)
                        if self.heading_re.match(candidate_text) and not self._is_list_item(candidate_text):
                            break
                        if not self._is_list_item(candidate_text):
                            break
                        section_path = candidate_section
                        list_section_path = candidate_section
                        list_blocks.append(candidate_block)
                        list_text_lines.append(self._strip_list_prefix(candidate_text))
                        idx += 1

                    if not list_blocks:
                        continue

                    full_list_text = "\n".join(list_text_lines)
                    list_parts = self._split_list_if_needed(full_list_text)
                    merged_bbox = self._merge_bboxes([b.bbox for b in list_blocks])
                    page_refs = [
                        self._ref_for_payload(extracted, page.page_number, b.bbox, list_section_path, b.provenance)
                        for b in list_blocks
                    ]

                    for part_index, part in enumerate(list_parts):
                        content_hash = self._hash_with_provenance(part, page_refs)
                        for ref in page_refs:
                            ref.content_hash = content_hash
                        rel = self.cross_ref_re.findall(part)
                        ldu = LogicalDocumentUnit(
                            ldu_id=deterministic_id(
                                "ldu",
                                {
                                    "doc": extracted.doc_id,
                                    "page": page.page_number,
                                    "order": list_blocks[0].reading_order,
                                    "part": part,
                                    "split": part_index,
                                },
                            ),
                            chunk_type="list",
                            content=part,
                            token_count=token_count(part),
                            bounding_box=merged_bbox,
                            parent_section=self._section_label(list_section_path),
                            parent_section_path=list_section_path.copy(),
                            page_refs=[ref.model_copy(deep=True) for ref in page_refs],
                            provenance_chain=ProvenanceChain.from_refs(page_refs),
                            content_hash=content_hash,
                            relationships=[f"cross_ref:{t}" for t in rel],
                        )
                        for m in self.resolvable_ref_re.finditer(part):
                            pending_resolved_refs.append(
                                _ResolvableRef(ldu_index=len(ldus), ref_type=m.group(1).lower(), ordinal=int(m.group(2)))
                            )
                        self.validator.validate(ldu)
                        ldus.append(ldu)
                    continue

                idx += 1
                page_refs = [
                    self._ref_for_payload(extracted, page.page_number, block.bbox, section_path, block.provenance)
                ]
                content_hash = self._hash_with_provenance(text, page_refs)
                for ref in page_refs:
                    ref.content_hash = content_hash
                rel = self.cross_ref_re.findall(text)
                # If the paragraph is within budget, emit a single LDU; otherwise split before validation
                if token_count(text) <= self.max_tokens:
                    ldu = LogicalDocumentUnit(
                        ldu_id=deterministic_id(
                            "ldu",
                            {
                                "doc": extracted.doc_id,
                                "page": page.page_number,
                                "order": block.reading_order,
                                "part": text,
                            },
                        ),
                        chunk_type="paragraph",
                        content=text,
                        token_count=token_count(text),
                        bounding_box=block.bbox,
                        parent_section=self._section_label(section_path),
                        parent_section_path=section_path.copy(),
                        page_refs=page_refs,
                        provenance_chain=ProvenanceChain.from_refs(page_refs),
                        content_hash=content_hash,
                        relationships=[f"cross_ref:{t}" for t in rel],
                    )
                    for m in self.resolvable_ref_re.finditer(text):
                        pending_resolved_refs.append(
                            _ResolvableRef(ldu_index=len(ldus), ref_type=m.group(1).lower(), ordinal=int(m.group(2)))
                        )
                    self.validator.validate(ldu)
                    ldus.append(ldu)
                else:
                    parts = self._split_paragraph_if_needed(text)
                    for part_index, part in enumerate(parts):
                        part_refs = [ref.model_copy(deep=True) for ref in page_refs]
                        part_hash = self._hash_with_provenance(part, part_refs)
                        for ref in part_refs:
                            ref.content_hash = part_hash
                        rel_local = self.cross_ref_re.findall(part)
                        part_ldu = LogicalDocumentUnit(
                            ldu_id=deterministic_id(
                                "ldu",
                                {
                                    "doc": extracted.doc_id,
                                    "page": page.page_number,
                                    "order": block.reading_order,
                                    "part": part,
                                    "split": part_index,
                                },
                            ),
                            chunk_type="paragraph",
                            content=part,
                            token_count=token_count(part),
                            bounding_box=block.bbox,
                            parent_section=self._section_label(section_path),
                            parent_section_path=section_path.copy(),
                            page_refs=part_refs,
                            provenance_chain=ProvenanceChain.from_refs(part_refs),
                            content_hash=part_hash,
                            relationships=[f"cross_ref:{t}" for t in rel_local],
                        )
                        for m in self.resolvable_ref_re.finditer(part):
                            pending_resolved_refs.append(
                                _ResolvableRef(ldu_index=len(ldus), ref_type=m.group(1).lower(), ordinal=int(m.group(2)))
                            )
                        self.validator.validate(part_ldu)
                        ldus.append(part_ldu)

            for table in page.tables:
                table_section = self._effective_section_path(section_path, table.provenance)
                table_ldu = self._make_table_ldu(extracted, page.page_number, table, table_section)
                if table_ldu is not None:
                    ldus.append(table_ldu)

            for figure in page.figures:
                caption = normalize_text(figure.caption or "")
                figure_section = self._effective_section_path(section_path, figure.provenance)
                page_refs = [
                    self._ref_for_payload(extracted, page.page_number, figure.bbox, figure_section, figure.provenance)
                ]
                figure_content = "[FIGURE]"
                if caption:
                    figure_content = f"[FIGURE] {caption}"
                content_hash = self._hash_with_provenance(figure_content, page_refs)
                for ref in page_refs:
                    ref.content_hash = content_hash
                ldu = LogicalDocumentUnit(
                    ldu_id=deterministic_id("ldu", {"doc": extracted.doc_id, "page": page.page_number, "figure": figure_content}),
                    chunk_type="figure",
                    content=figure_content,
                    structured_payload={"caption": caption or None},
                    token_count=token_count(figure_content),
                    bounding_box=figure.bbox,
                    parent_section=self._section_label(figure_section),
                    parent_section_path=figure_section.copy(),
                    page_refs=page_refs,
                    provenance_chain=ProvenanceChain.from_refs(page_refs),
                    content_hash=content_hash,
                )
                self.validator.validate(ldu)
                ldus.append(ldu)

        self._resolve_cross_references(ldus, pending_resolved_refs)
        self.validator.validate_document(ldus)

        rows = [ldu.model_dump(mode="json") for ldu in ldus]
        out = self.store.chunks_dir / f"{extracted.doc_id}.jsonl"
        self.store.write_jsonl(out, rows)
        elapsed = int((time.perf_counter() - started) * 1000)
        logger.info("stage=chunking end doc=%s chunks=%s duration_ms=%s", extracted.doc_name, len(ldus), elapsed)
        return ldus


class Chunker(ChunkingEngine):
    pass
