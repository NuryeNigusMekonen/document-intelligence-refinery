#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

PDF="$(ls data/*.pdf 2>/dev/null | head -n 1 || true)"
if [[ -z "$PDF" ]]; then
  echo "No PDF found in data/. Drop a PDF and rerun."
  exit 0
fi

echo "[1/5] Triage"
python - <<'PY'
from pathlib import Path
from refinery.config import Settings
from refinery.pipeline import RefineryPipeline

pdf = sorted(Path("data").glob("*.pdf"))[0]
pipe = RefineryPipeline(Settings(workspace_root=Path.cwd()))
profile = pipe.triage_only(pdf)
print(profile.model_dump_json(indent=2))
PY

echo "[2/5] Extraction + ingest"
refinery ingest "$PDF"

echo "[3/5] Print one extracted table JSON example (if exists)"
python - <<'PY'
from pathlib import Path
import json
from refinery.pipeline import doc_id_from_pdf

pdf = sorted(Path("data").glob("*.pdf"))[0]
doc_id = doc_id_from_pdf(pdf)
ext_path = Path(".refinery/extracted") / f"{doc_id}.json"
if not ext_path.exists():
    print("No extracted document artifact found")
else:
    data = json.loads(ext_path.read_text())
    for page in data.get("pages", []):
        tables = page.get("tables", [])
        if tables:
            print(json.dumps(tables[0], indent=2))
            break
    else:
        print("No tables detected in document")
PY

echo "[4/5] Build page index + print tree"
refinery build-index "$PDF"
refinery show-pageindex --doc "$PDF"

echo "[5/5] Query + provenance chain"
refinery query --doc "$PDF" "Summarize key financial facts"

echo "Ledger tail:"
tail -n 1 .refinery/extraction_ledger.jsonl || true
