# Extraction Quality Analysis (Table Precision/Recall)

Method: Page-level proxy benchmark across corpus PDFs with extracted artifacts.
- Weak ground truth: pages where `pdfplumber.extract_tables()` returns non-empty table structures.
- Prediction: pages with at least one extracted table in `.refinery/extracted/{doc_id}.json`.

## Aggregate (Micro)
- Documents evaluated: **18**
- TP pages: **150**
- FP pages: **4**
- FN pages: **489**
- Precision: **0.974**
- Recall: **0.2347**
- F1: **0.3783**

## Per-document snapshot

| doc_id | doc_name | precision | recall | f1 |
|---|---|---:|---:|---:|
| doc_028529e16612ffee | CBE ANNUAL REPORT 2023-24.pdf | 1.0 | 1.0 | 1.0 |
| doc_05cd116c8d148f33 | Annual_Report_JUNE-2021.pdf | 0.0 | 0.0 | 0.0 |
| doc_07f73c06d1765488 | 2013-E.C-Procurement-information.pdf | 0.0 | 0.0 | 0.0 |
| doc_23a0f7144db9947e | 2022_Audited_Financial_Statement_Report.pdf | 0.0 | 0.0 | 0.0 |
| doc_262084398a402524 | Annual_Report_JUNE-2020.pdf | 0.0 | 0.0 | 0.0 |
| doc_415a646a5077b30f | 2020_Audited_Financial_Statement_Report.pdf | 0.0 | 0.0 | 0.0 |
| doc_4510cef7a67e5476 | 20191010_Pharmaceutical-Manufacturing-Opportunites-in-Ethiopia_VF.pdf | 0.0 | 0.0 | 0.0 |
| doc_75587d41e7c4a9b8 | Annual_Report_JUNE-2017.pdf | 0.0 | 0.0 | 0.0 |
| doc_8aebaf799267bc81 | Annual_Report_JUNE-2023.pdf | 0.0 | 0.0 | 0.0 |
| doc_a75daf179ba13c93 | Annual_Report_JUNE-2022.pdf | 0.0 | 0.0 | 0.0 |
| doc_c716d4a64e11695a | 2013-E.C-Audit-finding-information.pdf | 0.0 | 0.0 | 0.0 |
| doc_d32e1f3eb5f512fc | Annual_Report_JUNE-2019.pdf | 0.0 | 0.0 | 0.0 |
| doc_e62bde522e9c764d | 2021_Audited_Financial_Statement_Report.pdf | 0.0 | 0.0 | 0.0 |
| doc_ee54d564c8eb6029 | 2013-E.C-Assigned-regular-budget-and-expense.pdf | 0.0 | 0.0 | 0.0 |
| doc_ee7e72b6068431cc | 2019_Audited_Financial_Statement_Report.pdf | 0.0 | 0.0 | 0.0 |
| doc_f2ee3f86adaa892f | Annual_Report_JUNE-2018.pdf | 0.0 | 0.0 | 0.0 |
| doc_f789d3443d8ca4ca | 2018_Audited_Financial_Statement_Report.pdf | 0.0 | 0.0 | 0.0 |
| doc_fa10bcfe0272bb3c | fta_performance_survey_final_report_2022.pdf | 0.9818 | 1.0 | 0.9908 |

## Lessons Learned (Failures and Fixes)
- Case 1: Scanned/noisy pages produce OCR drift and table row fragmentation; fix applied by escalating to vision strategy with confidence gates and preserving provenance for audit.
- Case 2: Dense fiscal tables with weak borders caused row-header ambiguity; fix applied through table-preserving chunk rules and section-first navigation before retrieval.