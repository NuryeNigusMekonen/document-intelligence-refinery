from __future__ import annotations

from pathlib import Path

from ..models import ExtractedDocument, ExtractedPage, ProvenanceRef, TableObject, TextBlock


class MarkdownAdapter:
    name = "markdown_adapter"

    def extract(self, file_path: Path, doc_id: str) -> ExtractedDocument:
        lines = file_path.read_text(encoding="utf-8").splitlines()
        section_stack: list[str] = []
        blocks: list[TextBlock] = []
        tables: list[TableObject] = []
        reading_order = 0
        i = 0

        while i < len(lines):
            line = lines[i]
            stripped = line.strip()
            line_no = i + 1

            if stripped.startswith("#"):
                level = len(stripped) - len(stripped.lstrip("#"))
                title = stripped[level:].strip()
                section_stack = section_stack[: max(level - 1, 0)] + [title]
                i += 1
                continue

            if stripped.startswith("```"):
                start = line_no
                code = []
                i += 1
                while i < len(lines) and not lines[i].strip().startswith("```"):
                    code.append(lines[i])
                    i += 1
                end = i + 1 if i < len(lines) else len(lines)
                prov = ProvenanceRef(
                    doc_name=file_path.name,
                    ref_type="markdown_lines",
                    line_range=(start, end),
                    section_path=section_stack.copy() if section_stack else ["Document Root"],
                    content_hash="pending",
                )
                blocks.append(
                    TextBlock(
                        text="[CODE] " + "\n".join(code),
                        bbox=(0.0, float(reading_order) * 10.0, 1000.0, float(reading_order + 1) * 10.0),
                        reading_order=reading_order,
                        confidence=0.95,
                        provenance=prov,
                    )
                )
                reading_order += 1
                i += 1
                continue

            if "|" in stripped and i + 1 < len(lines) and set(lines[i + 1].strip()) <= {"|", "-", ":", " "}:
                start = line_no
                raw = [stripped]
                i += 2
                while i < len(lines) and "|" in lines[i]:
                    raw.append(lines[i].strip())
                    i += 1
                headers = [c.strip() for c in raw[0].strip("|").split("|")]
                rows = [[c.strip() for c in r.strip("|").split("|")] for r in raw[1:]]
                prov = ProvenanceRef(
                    doc_name=file_path.name,
                    ref_type="markdown_lines",
                    line_range=(start, start + len(raw)),
                    section_path=section_stack.copy() if section_stack else ["Document Root"],
                    content_hash="pending",
                )
                tables.append(
                    TableObject(
                        bbox=(0.0, 0.0, 1000.0, 1000.0),
                        headers=headers,
                        rows=rows,
                        reading_order=10000 + len(tables),
                        confidence=0.9,
                        provenance=prov,
                    )
                )
                continue

            if stripped:
                prefix = "[LIST] " if stripped.startswith(("- ", "* ")) or stripped[:2].strip().endswith(".") else ""
                prov = ProvenanceRef(
                    doc_name=file_path.name,
                    ref_type="markdown_lines",
                    line_range=(line_no, line_no),
                    section_path=section_stack.copy() if section_stack else ["Document Root"],
                    content_hash="pending",
                )
                blocks.append(
                    TextBlock(
                        text=prefix + stripped,
                        bbox=(0.0, float(reading_order) * 10.0, 1000.0, float(reading_order + 1) * 10.0),
                        reading_order=reading_order,
                        confidence=0.92,
                        provenance=prov,
                    )
                )
                reading_order += 1
            i += 1

        page = ExtractedPage(page_number=1, width=1000.0, height=2000.0, blocks=blocks, tables=tables, figures=[])
        return ExtractedDocument(doc_id=doc_id, doc_name=file_path.name, pages=[page])
