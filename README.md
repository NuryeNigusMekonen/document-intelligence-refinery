# The Document Intelligence Refinery

Local-first document extraction pipeline (PDF, DOCX, Markdown, images, XLSX) that preserves structure and provenance and supports confidence-gated escalation.

English behavior remains unchanged, with added multilingual support for English + Amharic.

## Install

```bash
cd document-intelligence-refinery
python3.12 -m venv .venv
source .venv/bin/activate
pip install -e .[dev]
```

## Commands

```bash
refinery ingest data/*
refinery build-index data/*
refinery query --doc data/file.pdf "What is net profit?"
refinery audit --doc data/file.pdf "Revenue is $10M"
refinery show-pageindex --doc data/file.pdf
refinery open-citation --doc data/file.pdf --page 2 --bbox "50,100,300,180"
```

Supported file types:
- `.pdf`
- `.docx`
- `.md`
- `.png`, `.jpg`, `.jpeg`
- `.xlsx`

Language support matrix:
- PDF (digital): English + Amharic script detection
- PDF (scanned): OCR language routing (`eng`, `amh+eng`, fallback)
- DOCX: English + Amharic text preserved
- Markdown: English + Amharic text preserved
- Images: OCR-based, Amharic when language pack available
- XLSX: Unicode-safe cell extraction

## Provenance

Every chunk/fact carries:
- document name/id
- provenance `ref_type` and typed location fields
- deterministic `content_hash`

Provenance examples by type:
- PDF: `Page 3 | bbox [x0,y0,x1,y1]`
- Word: `Section: Financial Results > Revenue`
- Markdown: `Lines 42-57`
- Excel: `Sheet: Summary | Cells: B2:E10`
- Image: `Image bbox: [x0,y0,x1,y1]`

Extraction attempts are logged append-only in `.refinery/extraction_ledger.jsonl`.

Ledger rows also include:
- `detected_language`
- `ocr_lang_used` (when OCR paths are used)

## OCR setup (Ubuntu)

```bash
sudo apt-get update
sudo apt-get install -y tesseract-ocr tesseract-ocr-eng tesseract-ocr-amh
```

If Amharic pack is missing, the pipeline falls back to English OCR and records a fallback note.

## Environment examples

```bash
export REFINERY_OCR_ENABLED=true
export REFINERY_OCR_ENGINE=tesseract
export REFINERY_OCR_LANG_DEFAULT=eng
export REFINERY_OCR_LANG_FALLBACK=eng+amh
export REFINERY_OCR_AMHARIC_ENABLED=true
export REFINERY_LANGUAGE_DETECTION_MODE=script
export REFINERY_MULTILINGUAL_EMBEDDINGS=true
export REFINERY_EMBEDDING_MODEL=multilingual-lexical
```

## Common failure modes

- Missing `tesseract-ocr-amh` language pack → fallback to `eng`
- Very low OCR confidence on noisy scans
- Mixed-script lines can produce `unknown` language hint when sparse text

## Demo protocol (steps 1-4)

```bash
bash scripts/demo_protocol.sh
```

The demo performs triage, extraction, chunking/indexing, query, and prints provenance chains.
