# Acceptance Matrix (Non-Video)

This matrix records challenge-style execution evidence for one representative document in each required class (A/B/C/D).

## Legend
- **PASS**: Pipeline stage executed and produced expected artifact/output.
- **PARTIAL**: Stage executed but quality/provenance outcome is limited.
- **GAP**: Requirement not fully met yet.

---

## Class A — Annual Financial Report (native digital)
- **File:** `data/CBE ANNUAL REPORT 2023-24.pdf`
- **doc_id:** `doc_028529e16612ffee`

### Triage
- origin_type: `native_digital`
- layout_complexity: `table_heavy`
- language: `en`
- estimated_extraction_cost: `needs_layout_model`
- **Status:** PASS

### Extraction / Ledger
- strategy_used: `layout_lite`
- confidence_score: `0.8336`
- escalations: `fast_to_layout`
- cost_estimate: `0.1`
- **Status:** PASS

### Query + Provenance
- question: “What does the report say about revenue?”
- confidence: `0.72`
- provenance_count: `3`
- first provenance: page `5`, `pdf_bbox` present, `content_hash` present
- **Status:** PASS

---

## Class B — Scanned Government/Legal (image-based)
- **File:** `data/Audit Report - 2023.pdf`
- **doc_id:** `doc_7bc9d75ac27daec6`

### Triage
- origin_type: `scanned_image`
- layout_complexity: `figure_heavy`
- language: `en`
- estimated_extraction_cost: `needs_vision_model`
- **Status:** PASS

### Extraction / Ledger
- strategy_used: `layout_lite+vision_disabled`
- confidence_score: `0.3`
- escalations: `fast_to_layout`, `layout_to_vision`
- ocr_lang_used: `eng`
- **Status:** PARTIAL (vision stage disabled in this run)

### Query + Provenance
- question: “What is the title of this report?”
- confidence: `0.72`
- answer preview: “INDEPENDENT AUDITOR'S REPORT AND FINANCIAL STATEMENTS 30 JUNE 2023 ...”
- provenance_count: `1`
- first provenance: page `1`, `pdf_bbox` present, `content_hash` present
- **Status:** PASS (provenance-backed retrieval verified)

---

## Class C — Technical Assessment Report (mixed)
- **File:** `data/fta_performance_survey_final_report_2022.pdf`
- **doc_id:** `doc_fa10bcfe0272bb3c`

### Triage
- origin_type: `native_digital`
- layout_complexity: `table_heavy`
- language: `en`
- estimated_extraction_cost: `needs_layout_model`
- **Status:** PASS

### Extraction / Ledger
- strategy_used: `layout_lite`
- confidence_score: `0.8787`
- escalations: `fast_to_layout`
- cost_estimate: `0.1`
- **Status:** PASS

### Query + Provenance
- question: “What are the key findings in this assessment report?”
- confidence: `0.72`
- provenance_count: `3`
- first provenance: page `100`, `pdf_bbox` present, `content_hash` present
- **Status:** PASS

---

## Class D — Structured Data Report (table-heavy)
- **File:** `data/tax_expenditure_ethiopia_2021_22.pdf`
- **doc_id:** `doc_6bc2a669b953a7e3`

### Triage
- origin_type: `native_digital`
- layout_complexity: `table_heavy`
- language: `en`
- estimated_extraction_cost: `needs_layout_model`
- **Status:** PASS

### Extraction / Ledger
- strategy_used: `layout_lite`
- confidence_score: `0.9196`
- escalations: `fast_to_layout`
- cost_estimate: `0.1`
- **Status:** PASS

### Query + Provenance
- question: “What tax expenditure figures are reported?”
- confidence: `0.72`
- provenance_count: `3`
- first provenance: page `30`, `pdf_bbox` present, `content_hash` present
- **Status:** PASS

---

## Overall Non-Video Verdict

- **A:** PASS
- **B:** PASS (title-page retrieval proven; deeper scanned extraction still quality-limited)
- **C:** PASS
- **D:** PASS

### Remaining items to claim strict full completion (non-video)
1. Address rubric-level gaps already tracked in `CHALLENGE_STATUS.md`:
   - LangGraph-based query agent integration
   - LLM-generated PageIndex summaries

