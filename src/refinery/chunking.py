from __future__ import annotations

import logging
import re
import time
from dataclasses import dataclass
from typing import Sequence

from .config import Settings
from .models import BBox, ExtractedDocument, LogicalDocumentUnit, ProvenanceChain, ProvenanceRef, TableObject
from .runtime_rules import load_runtime_rules
from .storage import ArtifactStore
from .utils import deterministic_id, normalize_text, sha256_text, token_count

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
        if ldu.chunk_type == "table":
            payload = ldu.structured_payload or {}
            if not payload.get("headers"):
                raise ValueError("Table chunk must contain headers")
        if ldu.chunk_type == "list" and ldu.token_count > self.max_tokens and self.list_split_item_boundary_only:
            content = ldu.content or ""
            lines = [ln for ln in content.splitlines() if ln.strip()]
            if any(not self.list_item_re.match(ln) for ln in lines):
                raise ValueError("List split violated item boundary rule")


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

    def _to_bbox(self, bbox: BBox | tuple[float, float, float, float]) -> BBox:
        if isinstance(bbox, BBox):
            return bbox
        return BBox.model_validate(bbox)

    def _bbox_from_refs(self, page_refs: list[ProvenanceRef]) -> BBox | None:
        for ref in page_refs:
            if ref.bbox is not None:
                return self._to_bbox(ref.bbox)
        return None

    def _hash_with_provenance(self, content: str, page_refs: list[ProvenanceRef]) -> str:
        prov = "|".join(
            [
                f"{p.ref_type}:{p.page_number}:{self._to_bbox(p.bbox).as_tuple() if p.bbox else None}:{p.section_path}:{p.line_range}:{p.sheet_name}:{p.cell_range}"
                for p in page_refs
            ]
        )
        return sha256_text(normalize_text(content) + "||" + prov)

    def _make_table_ldu(self, extracted: ExtractedDocument, page_number: int, table: TableObject, section_path: list[str]) -> LogicalDocumentUnit:
        payload = {"headers": table.headers, "rows": table.rows}
        content = " | ".join(table.headers) + "\n" + "\n".join([" | ".join(row) for row in table.rows])
        page_refs = [
            table.provenance
            or ProvenanceRef(
                doc_name=extracted.doc_name,
                ref_type="pdf_bbox",
                page_number=page_number,
                bbox=table.bbox,
                section_path=section_path.copy() if section_path else None,
                content_hash="pending",
            )
        ]
        content_hash = self._hash_with_provenance(content, page_refs)
        for ref in page_refs:
            ref.content_hash = content_hash
        ldu = LogicalDocumentUnit(
            ldu_id=deterministic_id("ldu", {"doc_id": extracted.doc_id, "type": "table", "page": page_number, "hash": content_hash}),
            chunk_type="table",
            content=content,
            structured_payload=payload,
            token_count=token_count(content),
            bounding_box=table.bbox,
            parent_section=self._section_label(section_path),
            parent_section_path=section_path,
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
        ax0, ay0, ax1, ay1 = a
        bx0, by0, bx1, by1 = b
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
            for block in sorted(page.blocks, key=lambda b: b.reading_order):
                text = normalize_text(block.text)
                if not text:
                    continue
                if self._should_suppress_block_for_table(text, self._to_bbox(block.bbox), table_bboxes):
                    continue

                if self.heading_re.match(text):
                    section_path = [text]
                    continue

                chunk_type = "list" if self.list_item_re.match(text) else "paragraph"
                parts = self._split_list_if_needed(text) if chunk_type == "list" else [text]
                for part in parts:
                    page_refs = [
                        block.provenance
                        or ProvenanceRef(
                            doc_name=extracted.doc_name,
                            ref_type="pdf_bbox",
                            page_number=page.page_number,
                            bbox=block.bbox,
                            section_path=section_path.copy() if section_path else None,
                            content_hash="pending",
                        )
                    ]
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
                                "order": block.reading_order,
                                "part": part,
                            },
                        ),
                        chunk_type=chunk_type,
                        content=part,
                        token_count=token_count(part),
                        bounding_box=block.bbox,
                        parent_section=self._section_label(section_path),
                        parent_section_path=section_path.copy(),
                        page_refs=page_refs,
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

            for table in page.tables:
                ldus.append(self._make_table_ldu(extracted, page.page_number, table, section_path.copy()))

            for figure in page.figures:
                caption = normalize_text(figure.caption or "")
                page_refs = [
                    figure.provenance
                    or ProvenanceRef(
                        doc_name=extracted.doc_name,
                        ref_type="pdf_bbox",
                        page_number=page.page_number,
                        bbox=figure.bbox,
                        section_path=section_path.copy() if section_path else None,
                        content_hash="pending",
                    )
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
                    parent_section=self._section_label(section_path),
                    parent_section_path=section_path.copy(),
                    page_refs=page_refs,
                    provenance_chain=ProvenanceChain.from_refs(page_refs),
                    content_hash=content_hash,
                )
                self.validator.validate(ldu)
                ldus.append(ldu)

        self._resolve_cross_references(ldus, pending_resolved_refs)

        rows = [ldu.model_dump(mode="json") for ldu in ldus]
        out = self.store.chunks_dir / f"{extracted.doc_id}.jsonl"
        self.store.write_jsonl(out, rows)
        elapsed = int((time.perf_counter() - started) * 1000)
        logger.info("stage=chunking end doc=%s chunks=%s duration_ms=%s", extracted.doc_name, len(ldus), elapsed)
        return ldus


class Chunker(ChunkingEngine):
    pass
