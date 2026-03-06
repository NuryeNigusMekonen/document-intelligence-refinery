from __future__ import annotations

import json
from pathlib import Path

import pdfplumber

from refinery.config import Settings
from refinery.facts import FactStore
from refinery.query import QueryAgent
from refinery.storage import ArtifactStore
from refinery.vector_store import VectorIndex


def generate_qa_examples(store: ArtifactStore, reports_dir: Path) -> tuple[Path, Path, int]:
    agent = QueryAgent(store, VectorIndex(store), FactStore(store.db_dir / "facts.db"))

    class_docs = {
        "Class A (Native Financial)": (
            "doc_028529e16612ffee",
            [
                "What are the key capital and reserve related highlights?",
                "What does the report mention about net profit before tax?",
                "What does the report show about total assets in 2024?",
            ],
        ),
        "Class B (Scanned Government/Legal)": (
            "doc_c716d4a64e11695a",
            [
                "What is the main subject of this document?",
                "Summarize one major audit finding reported.",
                "What budget or expense context is described?",
            ],
        ),
        "Class C (Technical Assessment)": (
            "doc_fa10bcfe0272bb3c",
            [
                "What are the key findings in this assessment report?",
                "What is the objective of the survey/report?",
                "What recommendations are highlighted?",
            ],
        ),
        "Class D (Table-Heavy Fiscal)": (
            "doc_ee54d564c8eb6029",
            [
                "What are the assigned budget values reported?",
                "What are the main expense figures reported?",
                "Which sections discuss budget versus expense allocation?",
            ],
        ),
    }

    qa_rows: list[dict] = []
    md_lines = [
        "# Final Submission Evidence: 12 Q&A with ProvenanceChain",
        "",
        "Generated from repository artifacts and query agent outputs.",
        "",
    ]

    for class_name, (doc_id, questions) in class_docs.items():
        profile_path = store.profiles_dir / f"{doc_id}.json"
        doc_name = doc_id
        if profile_path.exists():
            doc_name = json.loads(profile_path.read_text(encoding="utf-8")).get("doc_name", doc_id)

        md_lines.append(f"## {class_name}")
        md_lines.append(f"- `doc_id`: `{doc_id}`")
        md_lines.append(f"- `doc_name`: `{doc_name}`")
        md_lines.append("")

        for i, question in enumerate(questions, start=1):
            out = agent.query_interface(doc_id, question, include_navigation_debug=True)
            provenance_chain = out.get("provenance_chain", [])

            qa_rows.append(
                {
                    "class": class_name,
                    "doc_id": doc_id,
                    "doc_name": doc_name,
                    "question_index": i,
                    "question": question,
                    "answer": out.get("answer", ""),
                    "confidence": out.get("confidence", 0.0),
                    "tool_trace": out.get("tool_trace", []),
                    "provenance_chain": provenance_chain,
                }
            )

            md_lines.append(f"### Q{i}")
            md_lines.append(f"**Question:** {question}")
            md_lines.append(f"**Answer:** {out.get('answer', '')}")
            md_lines.append(f"**Confidence:** {out.get('confidence', 0.0)}")
            md_lines.append(f"**Tool Trace:** {', '.join(out.get('tool_trace', []))}")
            md_lines.append("**ProvenanceChain:**")
            if not provenance_chain:
                md_lines.append("- *(none returned)*")
            else:
                for step_idx, ref in enumerate(provenance_chain, start=1):
                    md_lines.append(
                        "- step "
                        f"{step_idx}: "
                        f"doc_name=`{ref.get('doc_name')}`, "
                        f"ref_type=`{ref.get('ref_type')}`, "
                        f"page_number=`{ref.get('page_number')}`, "
                        f"bbox=`{ref.get('bbox')}`, "
                        f"content_hash=`{ref.get('content_hash')}`"
                    )
            md_lines.append("")

    md_path = reports_dir / "final_qa_provenance_examples.md"
    json_path = reports_dir / "final_qa_provenance_examples.json"
    md_path.write_text("\n".join(md_lines), encoding="utf-8")
    json_path.write_text(json.dumps(qa_rows, ensure_ascii=False, indent=2), encoding="utf-8")
    return md_path, json_path, len(qa_rows)


def generate_table_quality_analysis(store: ArtifactStore, reports_dir: Path, workspace_root: Path) -> tuple[Path, Path, int]:
    metrics: list[dict] = []

    for profile_path in sorted(store.profiles_dir.glob("*.json")):
        profile = json.loads(profile_path.read_text(encoding="utf-8"))
        doc_id = profile.get("doc_id")
        doc_name = profile.get("doc_name")
        if not doc_id:
            continue

        extracted_path = store.extracted_dir / f"{doc_id}.json"
        if not extracted_path.exists():
            continue

        src_pdf = workspace_root / "data" / str(doc_name)
        if src_pdf.suffix.lower() != ".pdf" or not src_pdf.exists():
            continue

        extracted_doc = json.loads(extracted_path.read_text(encoding="utf-8"))
        predicted_pages = {
            int(page.get("page_number", 0))
            for page in extracted_doc.get("pages", [])
            if page.get("tables")
        }

        gt_pages: set[int] = set()
        try:
            with pdfplumber.open(src_pdf) as pdf:
                for idx, page in enumerate(pdf.pages, start=1):
                    tables = page.extract_tables() or []
                    has_table = False
                    for table in tables:
                        if table and any(any((cell or "").strip() for cell in row or []) for row in table):
                            has_table = True
                            break
                    if has_table:
                        gt_pages.add(idx)
        except Exception:
            continue

        tp = len(predicted_pages & gt_pages)
        fp = len(predicted_pages - gt_pages)
        fn = len(gt_pages - predicted_pages)

        precision = (tp / (tp + fp)) if (tp + fp) else 0.0
        recall = (tp / (tp + fn)) if (tp + fn) else 0.0
        f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0

        metrics.append(
            {
                "doc_id": doc_id,
                "doc_name": doc_name,
                "origin_type": profile.get("origin_type"),
                "layout_complexity": profile.get("layout_complexity"),
                "pred_table_pages": len(predicted_pages),
                "gt_table_pages": len(gt_pages),
                "tp_pages": tp,
                "fp_pages": fp,
                "fn_pages": fn,
                "precision": round(precision, 4),
                "recall": round(recall, 4),
                "f1": round(f1, 4),
            }
        )

    sum_tp = sum(m["tp_pages"] for m in metrics)
    sum_fp = sum(m["fp_pages"] for m in metrics)
    sum_fn = sum(m["fn_pages"] for m in metrics)

    micro_precision = (sum_tp / (sum_tp + sum_fp)) if (sum_tp + sum_fp) else 0.0
    micro_recall = (sum_tp / (sum_tp + sum_fn)) if (sum_tp + sum_fn) else 0.0
    micro_f1 = (2 * micro_precision * micro_recall / (micro_precision + micro_recall)) if (micro_precision + micro_recall) else 0.0

    summary = {
        "documents_evaluated": len(metrics),
        "page_level_proxy": {
            "tp": sum_tp,
            "fp": sum_fp,
            "fn": sum_fn,
            "micro_precision": round(micro_precision, 4),
            "micro_recall": round(micro_recall, 4),
            "micro_f1": round(micro_f1, 4),
        },
        "method_note": "Proxy benchmark: pdfplumber table-bearing pages are treated as weak ground truth; extraction output table-bearing pages are predictions.",
        "per_document": metrics,
    }

    metrics_json_path = reports_dir / "table_extraction_metrics.json"
    metrics_md_path = reports_dir / "table_extraction_quality.md"

    metrics_json_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    lines = [
        "# Extraction Quality Analysis (Table Precision/Recall)",
        "",
        "Method: Page-level proxy benchmark across corpus PDFs with extracted artifacts.",
        "- Weak ground truth: pages where `pdfplumber.extract_tables()` returns non-empty table structures.",
        "- Prediction: pages with at least one extracted table in `.refinery/extracted/{doc_id}.json`.",
        "",
        "## Aggregate (Micro)",
        f"- Documents evaluated: **{summary['documents_evaluated']}**",
        f"- TP pages: **{sum_tp}**",
        f"- FP pages: **{sum_fp}**",
        f"- FN pages: **{sum_fn}**",
        f"- Precision: **{summary['page_level_proxy']['micro_precision']}**",
        f"- Recall: **{summary['page_level_proxy']['micro_recall']}**",
        f"- F1: **{summary['page_level_proxy']['micro_f1']}**",
        "",
        "## Per-document snapshot",
        "",
        "| doc_id | doc_name | precision | recall | f1 |",
        "|---|---|---:|---:|---:|",
    ]

    for item in metrics[:30]:
        lines.append(f"| {item['doc_id']} | {item['doc_name']} | {item['precision']} | {item['recall']} | {item['f1']} |")

    lines += [
        "",
        "## Lessons Learned (Failures and Fixes)",
        "- Case 1: Scanned/noisy pages produce OCR drift and table row fragmentation; fix applied by escalating to vision strategy with confidence gates and preserving provenance for audit.",
        "- Case 2: Dense fiscal tables with weak borders caused row-header ambiguity; fix applied through table-preserving chunk rules and section-first navigation before retrieval.",
    ]

    metrics_md_path.write_text("\n".join(lines), encoding="utf-8")
    return metrics_md_path, metrics_json_path, len(metrics)


def main() -> None:
    workspace_root = Path.cwd()
    reports_dir = workspace_root / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)

    settings = Settings(workspace_root=workspace_root)
    store = ArtifactStore(settings)

    qa_md, qa_json, qa_count = generate_qa_examples(store, reports_dir)
    quality_md, quality_json, docs_eval = generate_table_quality_analysis(store, reports_dir, workspace_root)

    print(f"wrote {qa_md}")
    print(f"wrote {qa_json}")
    print(f"wrote {quality_md}")
    print(f"wrote {quality_json}")
    print(f"qa_count={qa_count}")
    print(f"docs_eval={docs_eval}")


if __name__ == "__main__":
    main()
