#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path


def _run_cli(args: list[str]) -> dict:
    cmd = [sys.executable, "-m", "refinery.cli", *args]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        raise RuntimeError(
            f"Command failed: {' '.join(cmd)}\nSTDOUT:\n{proc.stdout}\nSTDERR:\n{proc.stderr}"
        )
    return _parse_json_from_output(proc.stdout)


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


def _read_last_ledger_entry_for_doc(ledger_path: Path, doc_id: str) -> dict | None:
    if not ledger_path.exists():
        return None
    last: dict | None = None
    for line in ledger_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue
        if row.get("doc_id") == doc_id:
            last = row
    return last


def _top_sections(pageindex_path: Path, top_k: int = 5) -> list[str]:
    if not pageindex_path.exists():
        return []
    data = json.loads(pageindex_path.read_text(encoding="utf-8"))
    roots = data.get("root_sections", [])
    out: list[str] = []
    for sec in roots[:top_k]:
        title = sec.get("title")
        if title:
            out.append(str(title))
    return out


def _chunk_ids_from_provenance(chunks_path: Path, provenance: list[dict], top_k: int = 5) -> list[str]:
    if not chunks_path.exists():
        return []
    hashes = [str(p.get("content_hash") or "") for p in provenance if p.get("content_hash")]
    if not hashes:
        return []
    wanted = set(hashes)
    ids: list[str] = []
    for line in chunks_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue
        if str(row.get("content_hash") or "") in wanted:
            ldu_id = str(row.get("ldu_id") or "")
            if ldu_id and ldu_id not in ids:
                ids.append(ldu_id)
        if len(ids) >= top_k:
            break
    return ids


def main() -> None:
    parser = argparse.ArgumentParser(description="Run refinery smoke E2E: ingest -> build-index -> query-interface")
    parser.add_argument("doc", help="Path to a supported document file")
    parser.add_argument(
        "--question",
        default="What are the capital expenditure projections for Q3?",
        help="Question for query-interface",
    )
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[1]
    doc_path = Path(args.doc).resolve()
    if not doc_path.exists():
        raise FileNotFoundError(f"Document not found: {doc_path}")

    ingest_out = _run_cli(["ingest", str(doc_path)])
    doc_id = str(ingest_out.get("doc_id") or "")
    if not doc_id:
        raise RuntimeError(f"Could not read doc_id from ingest output: {ingest_out}")

    _run_cli(["build-index", str(doc_path)])
    query_out = _run_cli(["query-interface", "--doc", str(doc_path), args.question])

    ledger_path = repo_root / ".refinery" / "extraction_ledger.jsonl"
    last_ledger = _read_last_ledger_entry_for_doc(ledger_path, doc_id) or {}
    strategy = str(last_ledger.get("strategy_used") or "unknown")

    pageindex_path = repo_root / ".refinery" / "pageindex" / f"{doc_id}.json"
    sections = _top_sections(pageindex_path)

    provenance = list(query_out.get("provenance_chain") or [])
    chunks_path = repo_root / ".refinery" / "chunks" / f"{doc_id}.jsonl"
    chunk_ids = _chunk_ids_from_provenance(chunks_path, provenance)

    print("=== Smoke E2E ===")
    print(f"doc_id: {doc_id}")
    print(f"selected strategy: {strategy}")
    print(f"ledger entry path: {ledger_path}")
    print("top PageIndex sections:")
    if sections:
        for s in sections:
            print(f"- {s}")
    else:
        print("- (none)")

    print("top retrieved chunk ids:")
    if chunk_ids:
        for cid in chunk_ids:
            print(f"- {cid}")
    else:
        print("- (none)")

    print("final answer:")
    print(query_out.get("answer", ""))
    print("provenance:")
    if provenance:
        for p in provenance:
            doc_name = p.get("doc_name")
            page_number = p.get("page_number")
            bbox = p.get("bbox")
            print(f"- doc={doc_name} page={page_number} bbox={bbox}")
    else:
        print("- (none)")


if __name__ == "__main__":
    main()
