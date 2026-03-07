#!/usr/bin/env bash
set -euo pipefail

# Five-minute demo runner aligned to the required protocol:
# 1) Triage  2) Extraction  3) PageIndex  4) Query+Provenance

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

# Allow overriding command via REFINERY_CMD env; fallback to module if binary not found
if command -v refinery >/dev/null 2>&1; then
  REFINERY_CMD="refinery"
else
  REFINERY_CMD="python -m refinery.cli"
fi

PDF="${1:-}"
if [[ -z "${PDF}" ]]; then
  PDF="$(ls data/*.pdf 2>/dev/null | head -n 1 || true)"
fi

if [[ -z "$PDF" ]]; then
  echo "No PDF found in data/. Drop a PDF into ./data and rerun:"
  echo "  cp /path/to/your.pdf data/"
  echo "  scripts/video_demo_5min.sh"
  exit 1
fi

QUESTION=${QUESTION:-"Summarize the key facts and figures in this document."}
TOPIC=${TOPIC:-"revenue"}

echo "[DEMO] Using PDF: $PDF"

mkdir -p reports/demo

echo "[1/5] Triage -- compute and show DocumentProfile"
python - <<'PY'
from pathlib import Path
import json
from refinery.config import Settings
from refinery.pipeline import RefineryPipeline, doc_id_from_pdf

pdf = Path("$PDF")
pipe = RefineryPipeline(Settings(workspace_root=Path.cwd()))
profile = pipe.triage_only(pdf)
out = json.loads(profile.model_dump_json())
print(json.dumps(out, indent=2, ensure_ascii=False))
(Path("reports/demo/profile.json")).write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")
(Path("reports/demo/_doc_id.txt")).write_text(profile.doc_id, encoding="utf-8")
PY

DOC_ID=$(cat reports/demo/_doc_id.txt)

echo "[2/5] Extraction -- ingest pipeline (chunks, vectors, facts)"
${REFINERY_CMD} ingest "$PDF" >/dev/null || true

echo "[3/5] Extraction artifacts -- first extracted table (if available) + ledger tail"
python - <<'PY'
from pathlib import Path
import json

doc_id = Path("reports/demo/_doc_id.txt").read_text(encoding="utf-8").strip()
ext_path = Path(".refinery/extracted")/f"{doc_id}.json"
if ext_path.exists():
    data = json.loads(ext_path.read_text(encoding="utf-8"))
    example = None
    for page in data.get("pages", []):
        tables = page.get("tables", [])
        if tables:
            t = tables[0]
            example = {
                "page_number": page.get("page_number"),
                "bbox": t.get("bbox"),
                "headers": t.get("headers"),
                "sample_rows": t.get("rows", [])[:5],
            }
            break
    if example:
        Path("reports/demo/first_table.json").write_text(json.dumps(example, indent=2, ensure_ascii=False), encoding="utf-8")
        print(json.dumps(example, indent=2, ensure_ascii=False))
    else:
        print("No tables detected in extracted document")
else:
    print("No extracted artifact found at", str(ext_path))

# Pretty-print the last ledger entry
ledger = Path(".refinery/extraction_ledger.jsonl")
if ledger.exists():
    last = None
    for line in ledger.read_text(encoding="utf-8").splitlines():
        line=line.strip()
        if not line:
            continue
        last = line
    if last:
        row = json.loads(last)
        keep = {
            k: row.get(k)
            for k in [
                "doc_id","origin_type","layout_complexity","strategy_used","confidence_score",
                "pages_processed","blocks_extracted","tables_extracted","table_pages_count",
                "detected_language","ocr_lang_used","escalations","notes","policy_context"
            ]
        }
        Path("reports/demo/ledger_tail.json").write_text(json.dumps(keep, indent=2, ensure_ascii=False), encoding="utf-8")
        print("\n[Ledger tail]", json.dumps(keep, indent=2, ensure_ascii=False))
PY

echo "[4/5] PageIndex -- build and navigate tree without vector search"
${REFINERY_CMD} build-index "$PDF" >/dev/null || true
${REFINERY_CMD} show-pageindex --doc "$PDF" --topic "$TOPIC" --top-k 3 | tee reports/demo/pageindex_nodes.json || true

echo "[5/5] Query with Provenance -- ask question and store output"
${REFINERY_CMD} query-interface --doc "$PDF" "$QUESTION" --output my_query_output.json >/dev/null || true

python - <<'PY'
import json, sys
from pathlib import Path
out = json.loads(Path("my_query_output.json").read_text(encoding="utf-8"))
chain = out.get("provenance_chain", [])
if chain:
    pref = chain[0]
    page = int(pref.get("page_number", 1))
    bbox = pref.get("bbox") or [0.0,0.0,1.0,1.0]
    print(f"Primary citation -> page {page}, bbox {bbox}")
    Path("reports/demo/primary_citation.json").write_text(json.dumps({"page": page, "bbox": bbox}, indent=2), encoding="utf-8")
else:
    print("No provenance_chain returned; see my_query_output.json")
PY

# Also return a snippet overlapping the cited bbox for quick verification
PAGE=$(python - <<'PY'
import json
from pathlib import Path
try:
    chain = json.loads(Path("my_query_output.json").read_text(encoding="utf-8")).get("provenance_chain", [])
    print(int(chain[0].get("page_number", 1)) if chain else 1)
except Exception:
    print(1)
PY
)

BBOX=$(python - <<'PY'
import json
from pathlib import Path
try:
    chain = json.loads(Path("my_query_output.json").read_text(encoding="utf-8")).get("provenance_chain", [])
    if chain and chain[0].get("bbox"):
        b = chain[0]["bbox"]
        print(f"{float(b[0])},{float(b[1])},{float(b[2])},{float(b[3])}")
    else:
        print("0,0,1,1")
except Exception:
    print("0,0,1,1")
PY
)

${REFINERY_CMD} open-citation --doc "$PDF" --page "$PAGE" --bbox "$BBOX" | tee reports/demo/citation_snippet.json || true

echo "\n[DONE] Artifacts saved under: reports/demo and .refinery"
echo "- Profile: reports/demo/profile.json"
echo "- First table (if any): reports/demo/first_table.json"
echo "- Ledger tail: reports/demo/ledger_tail.json"
echo "- PageIndex nodes: reports/demo/pageindex_nodes.json"
echo "- Query result: my_query_output.json"
echo "- Citation snippet: reports/demo/citation_snippet.json"

