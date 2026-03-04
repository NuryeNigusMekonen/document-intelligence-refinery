#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import sqlite3
import subprocess
import sys
from pathlib import Path
from typing import Any, NoReturn, cast


def _parse_json_from_output(stdout: str) -> dict:
    decoder = json.JSONDecoder()
    idx = 0
    last_obj: dict | None = None
    text = stdout.strip()
    while idx < len(text):
        while idx < len(text) and text[idx].isspace():
            idx += 1
        if idx >= len(text):
            break
        try:
            obj, end = decoder.raw_decode(text, idx)
            if isinstance(obj, dict):
                last_obj = obj
            idx = end
        except json.JSONDecodeError:
            idx += 1
    if last_obj is None:
        raise ValueError(f"No JSON object found in output:\n{stdout}")
    return last_obj


def _run_cli(args: list[str]) -> dict:
    cmd = [sys.executable, "-m", "refinery.cli", *args]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        raise RuntimeError(
            f"Command failed: {' '.join(cmd)}\nSTDOUT:\n{proc.stdout}\nSTDERR:\n{proc.stderr}"
        )
    return _parse_json_from_output(proc.stdout)


def _fail(stage_name: str, reason: str) -> NoReturn:
    print(f"FAILURE: {stage_name}")
    print(reason)
    raise SystemExit(1)


def _assert_file(stage_name: str, file_path: Path) -> None:
    if not file_path.exists() or not file_path.is_file():
        _fail(stage_name, f"Missing required file: {file_path}")


def _validate_ledger_entry(stage_name: str, ledger_path: Path, doc_id: str) -> dict[str, Any]:
    if not ledger_path.exists():
        _fail(stage_name, f"Missing ledger file: {ledger_path}")

    last_match: dict | None = None
    for line in ledger_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue
        if row.get("doc_id") == doc_id:
            last_match = row

    if last_match is None:
        _fail(stage_name, f"No ledger entry found for doc_id={doc_id}")

    return cast(dict[str, Any], last_match)


def _validate_stage2_extraction_metadata(stage_name: str, ledger_row: dict[str, Any]) -> None:
    if ledger_row.get("origin_type") in (None, ""):
        _fail(stage_name, "ledger row missing required field: origin_type")
    if ledger_row.get("layout_complexity") in (None, ""):
        _fail(stage_name, "ledger row missing required field: layout_complexity")

    pages_processed = ledger_row.get("pages_processed")
    if not isinstance(pages_processed, int) or pages_processed <= 0:
        _fail(stage_name, f"ledger row has invalid pages_processed={pages_processed}")

    strategy_used = str(ledger_row.get("strategy_used") or "").strip()
    accepted_prefixes = (
        "strategy_a_fast_text",
        "strategy_b_layout_docling",
        "strategy_c_local_ocr",
    )
    accepted_legacy = {
        "fast_text",
        "layout_lite",
        "docling",
        "vision",
        "vision_disabled",
        "docling+vision_disabled",
    }
    if not strategy_used:
        _fail(stage_name, "ledger row missing required field: strategy_used")
    if not (strategy_used.startswith(accepted_prefixes) or strategy_used in accepted_legacy):
        _fail(stage_name, f"ledger row has unsupported strategy_used={strategy_used}")


def _bbox_within_page_bounds(bbox: list[Any] | tuple[Any, ...], page_w: float, page_h: float, tolerance_ratio: float = 0.05) -> bool:
    if len(bbox) != 4:
        return False
    try:
        x0, y0, x1, y1 = [float(v) for v in bbox]
    except Exception:
        return False
    tol_x = page_w * tolerance_ratio
    tol_y = page_h * tolerance_ratio
    min_x, max_x = -tol_x, page_w + tol_x
    min_y, max_y = -tol_y, page_h + tol_y
    return (min_x <= x0 <= max_x and min_x <= x1 <= max_x and min_y <= y0 <= max_y and min_y <= y1 <= max_y and x1 >= x0 and y1 >= y0)


def _validate_extracted_bbox_bounds(stage_name: str, extracted_path: Path) -> None:
    _assert_file(stage_name, extracted_path)
    data = json.loads(extracted_path.read_text(encoding="utf-8"))
    pages = data.get("pages", [])
    if not isinstance(pages, list):
        _fail(stage_name, "extracted JSON missing pages array")

    for page_idx, page in enumerate(pages, start=1):
        page_w = float(page.get("width") or 0.0)
        page_h = float(page.get("height") or 0.0)
        if page_w <= 0 or page_h <= 0:
            _fail(stage_name, f"invalid page dimensions at page {page_idx}: width={page_w} height={page_h}")

        for bucket in ("blocks", "tables", "figures"):
            items = page.get(bucket, []) or []
            if not isinstance(items, list):
                continue
            for item_idx, item in enumerate(items, start=1):
                bbox = item.get("bbox") if isinstance(item, dict) else None
                if bbox is None:
                    continue
                if not _bbox_within_page_bounds(bbox, page_w, page_h, tolerance_ratio=0.05):
                    _fail(
                        stage_name,
                        f"bbox out of page bounds (>5%) at page {page_idx} {bucket}[{item_idx}] bbox={bbox} page=({page_w},{page_h})",
                    )

                if isinstance(item, dict):
                    pref = item.get("provenance")
                    pref_bbox = None
                    if isinstance(pref, dict):
                        pref_bbox = pref.get("bbox")
                    if pref_bbox is not None:
                        if not _bbox_within_page_bounds(pref_bbox, page_w, page_h, tolerance_ratio=0.05):
                            _fail(
                                stage_name,
                                f"provenance bbox out of page bounds (>5%) at page {page_idx} {bucket}[{item_idx}] bbox={pref_bbox} page=({page_w},{page_h})",
                            )


def _validate_provenance_bbox_bounds(stage_name: str, first_prov: dict[str, Any], extracted_path: Path) -> None:
    if not first_prov:
        return
    bbox = first_prov.get("bbox")
    page_number = first_prov.get("page")
    if bbox is None or page_number is None:
        return
    try:
        page_number_int = int(page_number)
    except Exception:
        return

    data = json.loads(extracted_path.read_text(encoding="utf-8"))
    pages = data.get("pages", [])
    if not isinstance(pages, list) or page_number_int < 1 or page_number_int > len(pages):
        return
    page = pages[page_number_int - 1]
    page_w = float(page.get("width") or 0.0)
    page_h = float(page.get("height") or 0.0)
    if page_w <= 0 or page_h <= 0:
        return
    if not _bbox_within_page_bounds(bbox, page_w, page_h, tolerance_ratio=0.05):
        _fail(stage_name, f"first provenance bbox out of page bounds (>5%): bbox={bbox} page=({page_w},{page_h})")


def _validate_facts_table(stage_name: str, db_path: Path) -> None:
    _assert_file(stage_name, db_path)
    with sqlite3.connect(db_path) as conn:
        cur = conn.cursor()
        cur.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='facts'")
        row = cur.fetchone()
        if row is None:
            _fail(stage_name, "facts table not found in facts.db")


def _load_jsonl(stage_name: str, jsonl_path: Path) -> list[dict[str, Any]]:
    _assert_file(stage_name, jsonl_path)
    rows: list[dict[str, Any]] = []
    for idx, line in enumerate(jsonl_path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError as exc:
            _fail(stage_name, f"Invalid JSONL at line {idx}: {exc}")
        if not isinstance(row, dict):
            _fail(stage_name, f"JSONL line {idx} is not an object")
        rows.append(cast(dict[str, Any], row))
    return rows


def _validate_chunks_schema(stage_name: str, chunks_path: Path, doc_id: str) -> None:
    rows = _load_jsonl(stage_name, chunks_path)
    if not rows:
        _fail(stage_name, f"No chunks generated for doc_id={doc_id}")

    required_fields = {"ldu_id", "chunk_type", "token_count", "page_refs", "content_hash"}
    allowed_chunk_types = {"paragraph", "table", "figure", "list", "section_summary", "fact"}

    for idx, row in enumerate(rows, start=1):
        missing = [field for field in required_fields if field not in row]
        if missing:
            _fail(stage_name, f"Chunk row {idx} missing fields: {missing}")

        if not isinstance(row.get("ldu_id"), str) or not str(row.get("ldu_id") or "").strip():
            _fail(stage_name, f"Chunk row {idx} has invalid ldu_id")

        chunk_type = row.get("chunk_type")
        if chunk_type not in allowed_chunk_types:
            _fail(stage_name, f"Chunk row {idx} has invalid chunk_type={chunk_type}")

        token_count = row.get("token_count")
        if not isinstance(token_count, int) or token_count < 0:
            _fail(stage_name, f"Chunk row {idx} has invalid token_count={token_count}")

        if not isinstance(row.get("content_hash"), str) or not str(row.get("content_hash") or "").strip():
            _fail(stage_name, f"Chunk row {idx} has invalid content_hash")

        page_refs = row.get("page_refs")
        if not isinstance(page_refs, list) or not page_refs:
            _fail(stage_name, f"Chunk row {idx} has empty/invalid page_refs")

        for pref_idx, pref in enumerate(page_refs, start=1):
            if not isinstance(pref, dict):
                _fail(stage_name, f"Chunk row {idx} page_ref {pref_idx} is not an object")
            if not pref.get("doc_name"):
                _fail(stage_name, f"Chunk row {idx} page_ref {pref_idx} missing doc_name")
            if not pref.get("content_hash"):
                _fail(stage_name, f"Chunk row {idx} page_ref {pref_idx} missing content_hash")


def _chunk_counts_by_type(rows: list[dict[str, Any]]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for row in rows:
        chunk_type = str(row.get("chunk_type") or "unknown")
        counts[chunk_type] = counts.get(chunk_type, 0) + 1
    return counts


def _count_pageindex_sections(pageindex_path: Path) -> int:
    data = json.loads(pageindex_path.read_text(encoding="utf-8"))
    roots = data.get("root_sections", [])
    if not isinstance(roots, list):
        return 0

    total = 0
    stack: list[dict[str, Any]] = [node for node in roots if isinstance(node, dict)]
    while stack:
        node = stack.pop()
        total += 1
        children = node.get("child_sections", [])
        if isinstance(children, list):
            stack.extend(child for child in children if isinstance(child, dict))
    return total


def _validate_query_output(stage_name: str, query_out: dict) -> None:
    if "answer" not in query_out:
        _fail(stage_name, "query-interface output missing 'answer'")
    if "provenance_chain" not in query_out:
        _fail(stage_name, "query-interface output missing 'provenance_chain'")
    if "tool_trace" not in query_out:
        _fail(stage_name, "query-interface output missing 'tool_trace'")

    answer = query_out.get("answer")
    if not isinstance(answer, str) or not answer.strip():
        _fail(stage_name, "query-interface answer is empty or not a string")

    tool_trace = query_out.get("tool_trace")
    if not isinstance(tool_trace, list):
        _fail(stage_name, "tool_trace is not a list")
    if not any(t in {"pageindex_navigate", "semantic_search", "structured_query"} for t in tool_trace):
        _fail(stage_name, "tool_trace does not include expected retrieval tools")

    provenance_raw = query_out.get("provenance_chain")
    if not isinstance(provenance_raw, list):
        _fail(stage_name, "provenance_chain is not a list")
    if not provenance_raw:
        _fail(stage_name, "provenance_chain is empty")

    for idx, pref_raw in enumerate(provenance_raw, start=1):
        if not isinstance(pref_raw, dict):
            _fail(stage_name, f"provenance entry {idx} is not an object")
        pref = cast(dict[str, object], pref_raw)
        if pref.get("doc_name") in (None, ""):
            _fail(stage_name, f"provenance entry {idx} missing doc_name")
        if pref.get("page_number") is None:
            _fail(stage_name, f"provenance entry {idx} missing page_number")
        if pref.get("bbox") is None:
            _fail(stage_name, f"provenance entry {idx} missing bbox")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Validate end-to-end refinery pipeline and fail fast per stage"
    )
    parser.add_argument("doc", help="Path to a supported document file")
    parser.add_argument(
        "--question",
        default="What are the capital expenditure projections for Q3?",
        help="Question for query-interface validation",
    )
    args = parser.parse_args()

    stage = "INPUT_VALIDATION"
    repo_root = Path(__file__).resolve().parents[1]
    artifacts_dir = os.environ.get("REFINERY_ARTIFACTS_DIR", ".refinery")
    artifacts_root = repo_root / artifacts_dir
    doc_path = Path(args.doc).resolve()
    if not doc_path.exists():
        _fail(stage, f"Document not found: {doc_path}")

    stage = "INGEST"
    ingest_out: dict | None = None
    try:
        ingest_out = _run_cli(["ingest", str(doc_path)])
    except Exception as exc:
        _fail(stage, str(exc))

    if ingest_out is None:
        _fail(stage, "ingest returned no output")

    doc_id = str(ingest_out.get("doc_id") or "")
    if not doc_id:
        _fail(stage, f"ingest did not return doc_id: {ingest_out}")

    stage = "INGEST_ARTIFACTS"
    _assert_file(stage, artifacts_root / "profiles" / f"{doc_id}.json")
    _assert_file(stage, artifacts_root / "extracted" / f"{doc_id}.json")
    ledger_row = _validate_ledger_entry(stage, artifacts_root / "extraction_ledger.jsonl", doc_id)

    stage = "BBOX_SANITY"
    _validate_extracted_bbox_bounds(stage, artifacts_root / "extracted" / f"{doc_id}.json")

    stage = "STAGE2_EXTRACTION_VALIDATION"
    _validate_stage2_extraction_metadata(stage, ledger_row)
    strategy_used = str(ledger_row.get("strategy_used") or "unknown")
    confidence_score = float(ledger_row.get("confidence_score") or 0.0)
    print(f"Stage2 EXTRACTION PASS strategy={strategy_used} confidence={confidence_score:.2f}")

    stage = "BUILD_INDEX"
    build_out: dict | None = None
    try:
        build_out = _run_cli(["build-index", str(doc_path)])
    except Exception as exc:
        _fail(stage, str(exc))

    if build_out is None:
        _fail(stage, "build-index returned no output")

    build_doc_id = str(build_out.get("doc_id") or "")
    if build_doc_id != doc_id:
        _fail(stage, f"build-index doc_id mismatch: ingest={doc_id} build-index={build_doc_id}")

    stage = "INDEX_ARTIFACTS"
    chunks_path = artifacts_root / "chunks" / f"{doc_id}.jsonl"
    _validate_chunks_schema(stage, chunks_path, doc_id)
    pageindex_path = artifacts_root / "pageindex" / f"{doc_id}.json"
    _assert_file(stage, pageindex_path)
    _validate_facts_table(stage, artifacts_root / "db" / "facts.db")

    stage = "QUERY_INTERFACE"
    query_out: dict | None = None
    try:
        query_out = _run_cli(["query-interface", "--doc", doc_id, args.question])
    except Exception as exc:
        _fail(stage, str(exc))

    if query_out is None:
        _fail(stage, "query-interface returned no output")

    stage = "QUERY_OUTPUT"
    _validate_query_output(stage, query_out)

    chunks_rows = _load_jsonl(stage, chunks_path)
    by_type = _chunk_counts_by_type(chunks_rows)
    section_count = _count_pageindex_sections(pageindex_path)

    provenance_chain = query_out.get("provenance_chain")
    first_prov: dict[str, Any] = {}
    if isinstance(provenance_chain, list) and provenance_chain and isinstance(provenance_chain[0], dict):
        first_prov = cast(dict[str, Any], provenance_chain[0])

    strategy_used = str(ledger_row.get("strategy_used") or "unknown")
    confidence_score = float(ledger_row.get("confidence_score") or 0.0)
    origin_type = ledger_row.get("origin_type")
    layout_complexity = ledger_row.get("layout_complexity")
    pages_processed = ledger_row.get("pages_processed")
    blocks_extracted = ledger_row.get("blocks_extracted")
    tables_extracted = ledger_row.get("tables_extracted")
    escalations = ledger_row.get("escalations")
    if not isinstance(escalations, list):
        escalations = []

    print("SUCCESS: END_TO_END_VALIDATION")
    print(f"doc_id={doc_id}")
    print(f"strategy_used={strategy_used}")
    print(f"confidence_score={confidence_score:.4f}")
    print(f"origin_type={origin_type}")
    print(f"layout_complexity={layout_complexity}")
    print(f"pages_processed={pages_processed}")
    print(f"blocks_extracted={blocks_extracted}")
    print(f"tables_extracted={tables_extracted}")
    print(f"escalations={json.dumps(escalations, ensure_ascii=False)}")
    print(f"chunk_count_total={len(chunks_rows)}")
    print(f"chunk_count_by_type={json.dumps(by_type, ensure_ascii=False, sort_keys=True)}")
    print(f"pageindex_section_count={section_count}")
    print(
        "first_provenance="
        + json.dumps(
            {
                "page": first_prov.get("page_number"),
                "bbox": first_prov.get("bbox"),
            },
            ensure_ascii=False,
        )
    )

    stage = "PROVENANCE_BBOX_SANITY"
    _validate_provenance_bbox_bounds(
        stage,
        {
            "page": first_prov.get("page_number"),
            "bbox": first_prov.get("bbox"),
        },
        artifacts_root / "extracted" / f"{doc_id}.json",
    )


if __name__ == "__main__":
    main()
