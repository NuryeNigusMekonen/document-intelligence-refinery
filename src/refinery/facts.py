from __future__ import annotations

import logging
import re
import sqlite3
from pathlib import Path

import yaml

from .models import LogicalDocumentUnit

logger = logging.getLogger(__name__)

FACT_PATTERNS = [
    re.compile(r"\b(Revenue|Net profit|Total assets|Operating income|EPS)\b[^\n]{0,60}?([\$€£]?\d[\d,]*(?:\.\d+)?%?)", re.IGNORECASE),
    re.compile(r"\b(FY\s?\d{2,4}|\d{4})\b"),
]


def _load_amharic_keywords() -> dict[str, list[str]]:
    resource = Path(__file__).parent / "resources" / "amharic_keywords.yaml"
    if not resource.exists():
        return {}
    data = yaml.safe_load(resource.read_text(encoding="utf-8")) or {}
    return data.get("keywords", {})


AMHARIC_KEYWORDS = _load_amharic_keywords()
NUM_RE = re.compile(r"([\$€£]?\d[\d,]*(?:\.\d+)?%?)")


class FactStore:
    def __init__(self, db_path: Path):
        self.db_path = db_path
        self._init_db()

    def _conn(self) -> sqlite3.Connection:
        return sqlite3.connect(self.db_path)

    def _init_db(self) -> None:
        with self._conn() as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS facts (
                    doc_id TEXT,
                    key TEXT,
                    value TEXT,
                    unit TEXT,
                    context TEXT,
                    ref_type TEXT,
                    page_number INTEGER,
                    section_path TEXT,
                    line_range TEXT,
                    sheet_name TEXT,
                    cell_range TEXT,
                    bbox TEXT,
                    content_hash TEXT
                )
                """
            )
            existing_cols = {row[1] for row in conn.execute("PRAGMA table_info(facts)").fetchall()}
            required_cols = {
                "ref_type": "TEXT",
                "section_path": "TEXT",
                "line_range": "TEXT",
                "sheet_name": "TEXT",
                "cell_range": "TEXT",
            }
            for col, col_type in required_cols.items():
                if col not in existing_cols:
                    conn.execute(f"ALTER TABLE facts ADD COLUMN {col} {col_type}")
            conn.commit()

    def ingest_ldus(self, doc_id: str, ldus: list[LogicalDocumentUnit]) -> int:
        rows: list[tuple] = []
        for ldu in ldus:
            if ldu.chunk_type not in {"paragraph", "table", "fact"}:
                continue
            text = ldu.content or ""
            if not text.strip():
                continue
            for pattern in FACT_PATTERNS:
                for match in pattern.finditer(text):
                    if match.lastindex and match.lastindex >= 2:
                        key = match.group(1)
                        value = match.group(2)
                    else:
                        key = "Year"
                        value = match.group(1)
                    unit = "%" if value.endswith("%") else ("currency" if any(c in value for c in "$€£") else "")
                    pref = ldu.page_refs[0]
                    rows.append(
                        (
                            doc_id,
                            key,
                            value,
                            unit,
                            text[:300],
                            pref.ref_type,
                            pref.page_number,
                            ">".join(pref.section_path or []),
                            ",".join(str(v) for v in pref.line_range) if pref.line_range else "",
                            pref.sheet_name or "",
                            pref.cell_range or "",
                            ",".join(str(v) for v in pref.bbox) if pref.bbox else "",
                            ldu.content_hash,
                        )
                    )

            for key, am_terms in AMHARIC_KEYWORDS.items():
                if not any(term in text for term in am_terms):
                    continue
                number_match = NUM_RE.search(text)
                value = number_match.group(1) if number_match else ""
                unit = "%" if value.endswith("%") else ("currency" if any(c in value for c in "$€£") else "")
                pref = ldu.page_refs[0]
                rows.append(
                    (
                        doc_id,
                        key.capitalize(),
                        value,
                        unit,
                        text[:300],
                        pref.ref_type,
                        pref.page_number,
                        ">".join(pref.section_path or []),
                        ",".join(str(v) for v in pref.line_range) if pref.line_range else "",
                        pref.sheet_name or "",
                        pref.cell_range or "",
                        ",".join(str(v) for v in pref.bbox) if pref.bbox else "",
                        ldu.content_hash,
                    )
                )
        with self._conn() as conn:
            conn.executemany(
                "INSERT INTO facts (doc_id, key, value, unit, context, ref_type, page_number, section_path, line_range, sheet_name, cell_range, bbox, content_hash) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                rows,
            )
            conn.commit()
        logger.info("facts_ingested=%s", len(rows))
        return len(rows)

    def query(self, query: str, doc_id: str | None = None) -> list[dict]:
        sql = query.strip()
        if not sql.lower().startswith("select"):
            sql = "SELECT doc_id, key, value, unit, context, ref_type, page_number, section_path, line_range, sheet_name, cell_range, bbox, content_hash FROM facts WHERE key LIKE ? OR context LIKE ? LIMIT 20"
            term = f"%{query}%"
            params = [term, term]
            if doc_id:
                sql = "SELECT doc_id, key, value, unit, context, ref_type, page_number, section_path, line_range, sheet_name, cell_range, bbox, content_hash FROM facts WHERE doc_id = ? AND (key LIKE ? OR context LIKE ?) LIMIT 20"
                params = [doc_id, term, term]
        else:
            params = []
        with self._conn() as conn:
            cur = conn.execute(sql, params)
            cols = [d[0] for d in cur.description]
            return [dict(zip(cols, row)) for row in cur.fetchall()]
