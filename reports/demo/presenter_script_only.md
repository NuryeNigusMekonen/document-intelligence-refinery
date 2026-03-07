Five-minute Presenter Script — Artifacts-only Walkthrough (no live commands)

Purpose
- Show only what is already in the repo/artifacts. No terminal, no running code. Open files under .refinery and explain what they prove, mapped to the rubric. Keep strictly within 5 minutes.

Chosen example
- Use the existing artifacts for: doc_8aebaf799267bc81
- Files you will open during the recording:
  - Profile: [.refinery/profiles/doc_8aebaf799267bc81.json](.refinery/profiles/doc_8aebaf799267bc81.json)
  - Extracted document: [.refinery/extracted/doc_8aebaf799267bc81.json](.refinery/extracted/doc_8aebaf799267bc81.json)
  - Chunks (for doc name/provenance): [.refinery/chunks/doc_8aebaf799267bc81.jsonl](.refinery/chunks/doc_8aebaf799267bc81.jsonl)
  - PageIndex: [.refinery/pageindex/doc_8aebaf799267bc81.json](.refinery/pageindex/doc_8aebaf799267bc81.json)
  - Ledger (scroll to last line): [.refinery/extraction_ledger.jsonl](.refinery/extraction_ledger.jsonl)
  - Query output (example): [my_query_output.json](my_query_output.json)
  - Source PDF: open the PDF whose name is shown in doc_name (from profile/extracted/chunks). If it’s in data/, open it side-by-side.

Timing and exact narration

0:00 – 0:25 | Opening: protocol overview
- On screen: VS Code with the Explorer showing the .refinery/ folders. Optionally keep the source PDF visible in a viewer.
- Say: “I’ll present a 5-minute, artifacts-only walkthrough of our Document Intelligence Refinery. We’ll follow the rubric sequence: Triage, Extraction, PageIndex, and Query with provenance. All evidence is from prior runs stored under .refinery/.”

0:25 – 1:25 | Triage — DocumentProfile and routing strategy
- Show: [.refinery/profiles/doc_8aebaf799267bc81.json](.refinery/profiles/doc_8aebaf799267bc81.json)
- Point to these fields in the JSON: doc_name, origin_type, layout_complexity, domain_hint, estimated_extraction_cost, and the confidences (origin_confidence, layout_confidence, domain_confidence, extraction_cost_confidence). Also highlight page_count, avg_char_density, image_area_ratio, whitespace_ratio.
- Say: “This is the DocumentProfile produced during triage. The system classifies origin_type (e.g., native_digital, mixed, scanned_image), layout_complexity (e.g., single_column, table_heavy), domain_hint, and the estimated extraction cost. Signals like avg_char_density, image_area_ratio, and whitespace_ratio drive the classification.”
- Say: “Based on these signals, our router selects strategy A/B/C. Fast-Text for clean digital, Layout/Docling for table/multi-column, Vision/OCR for scanned or handwriting-like.”
- Optional code proof (brief): show [TriageAgent._compute_metrics()](src/agents/triage.py:190) for signal computation and [classify_profile()](src/agents/triage.py:68) for mapping signals to origin/layout/cost.

1:25 – 2:25 | Extraction — fidelity, structured tables, ledger and escalation guard
- Show: [.refinery/extracted/doc_8aebaf799267bc81.json](.refinery/extracted/doc_8aebaf799267bc81.json)
- Find and expand: pages[N].tables[0]. Show headers and a few rows.
- Side-by-side: open the source PDF to the corresponding page number (from pages[N].page_number) to visually compare that the headers/values match. Keep the JSON and PDF visible together.
- Say: “Here is a structured table (headers + rows) next to the original page. This enables reliable downstream use.”
- Show: [.refinery/extraction_ledger.jsonl](.refinery/extraction_ledger.jsonl) and scroll to the last line. Read out strategy_used, confidence_score, pages_processed, tables_extracted, detected_language/ocr_lang_used, and escalations.
- Say: “The extraction ledger captures strategy, confidence, cost estimate, and escalations. If confidence is below thresholds, the router escalates (e.g., fast → layout → vision) and records that.”
- Optional code proof: [ExtractionRouter.extract()](src/agents/extractor.py:1246) for stage logic; escalation thresholds and floors; label mapping in [ExtractionRouter._normalize_strategy_label()](src/agents/extractor.py:1174); ledger writing in [ExtractionRouter._write_ledger()](src/agents/extractor.py:1084).

2:25 – 3:15 | PageIndex — hierarchical navigation without vectors
- Show: [.refinery/pageindex/doc_8aebaf799267bc81.json](.refinery/pageindex/doc_8aebaf799267bc81.json)
- Expand: root_sections[0], then child_sections[0] to demonstrate multiple levels. Point out title, page_start/page_end, summary, key_entities, data_types_present.
- Say: “This is the PageIndex tree built from logical units. It provides hierarchical narrowing of scope. No embedding/vector search is required for this navigation.”
- Optional code proof: [PageIndexBuilder.build()](src/agents/indexer.py:313) constructs the tree; navigation scoring uses lexical BM25-like logic in [pageindex_navigate()](src/agents/indexer.py:368), not embeddings.

3:15 – 4:35 | Query with provenance — answer + citation + source verification
- Show: [my_query_output.json](my_query_output.json)
- Point to: answer, confidence, provenance_chain[0]. Show doc_name, page_number, bbox.
- Say: “We answer in natural language and include verifiable provenance: document name, page number, and a bounding box. This lets us independently validate citations.”
- Verify against source: Open the source PDF at the cited page_number. Zoom to the bbox region (use your viewer’s rectangle/zoom tools). Read the value shown in the PDF and confirm it matches the answer context.
- Optional proof of snippet logic: [open_citation_snippet()](src/agents/query_agent.py:890) shows how we overlap bbox with chunks to surface a representative snippet.

4:35 – 5:00 | Close — rubric alignment and boundaries
- Say: “We’ve shown the Triage profile with signals and why that routes to a specific strategy; Extraction quality with a structured table and the ledger for confidence/cost/escalation; PageIndex hierarchical navigation without vectors; and Query with provenance validated against the PDF. All evidence came from artifacts under .refinery/, no live runs.”

Notes to keep it within 5 minutes
- Move briskly; show only one table and one PageIndex branch.
- If no tables exist for this doc, say: “This document is text-dominant; the process still captures blocks and figures and records quality in the ledger.”
- If the PageIndex has shallow depth, expand two different root_sections to show coverage and entities/summaries.

