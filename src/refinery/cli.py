from __future__ import annotations

import argparse
import glob
import json
import logging
from pathlib import Path
from datetime import datetime

from .config import Settings
from .facts import FactStore
from .file_types import doc_id_from_file
from .logging_utils import configure_logging
from .pipeline import RefineryPipeline
from .query import QueryAgent, open_citation_snippet
from .storage import ArtifactStore
from .vector_store import VectorIndex

SUPPORTED_EXTENSIONS = {".pdf", ".docx", ".md", ".png", ".jpg", ".jpeg", ".xlsx"}


def _expand_inputs(patterns: list[str]) -> list[Path]:
    files: list[Path] = []
    for pattern in patterns:
        matches = glob.glob(pattern)
        if matches:
            files.extend(Path(m) for m in matches if Path(m).suffix.lower() in SUPPORTED_EXTENSIONS)
        elif Path(pattern).suffix.lower() in SUPPORTED_EXTENSIONS:
            files.append(Path(pattern))
    return sorted(set(files))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="refinery", description="Document Intelligence Refinery")
    sub = parser.add_subparsers(dest="command", required=True)

    ingest = sub.add_parser("ingest")
    ingest.add_argument("pdfs", nargs="+")

    bidx = sub.add_parser("build-index")
    bidx.add_argument("pdfs", nargs="+")

    query = sub.add_parser("query")
    query.add_argument("--doc", required=True)
    query.add_argument("question")

    qiface = sub.add_parser("query-interface")
    qiface.add_argument("--doc", required=True)
    qiface.add_argument("--debug-sections", action="store_true", help="Include PageIndex section selection/debug metadata")
    qiface.add_argument("question")
    qiface.add_argument("--output", type=str, default=None, help="Optional path to save the output JSON result.")
    qiface.add_argument(
        "--append-output",
        action="store_true",
        help=(
            "If set together with --output, append the current question/answer to the existing JSON file "
            "instead of overwriting it. The file will contain a JSON array of entries."
        ),
    )

    audit = sub.add_parser("audit")
    audit.add_argument("--doc", required=True)
    audit.add_argument("claim")

    show = sub.add_parser("show-pageindex")
    show.add_argument("--doc", required=True)
    show.add_argument("--topic", default="")
    show.add_argument("--top-k", type=int, default=3)

    eval_pageindex = sub.add_parser("eval-pageindex")
    eval_pageindex.add_argument("--doc", required=True)
    eval_pageindex.add_argument("--topic", required=True)
    eval_pageindex.add_argument("--expected", required=True, help="Comma-separated expected section titles")
    eval_pageindex.add_argument("--top-k", type=int, default=3)

    citation = sub.add_parser("open-citation")
    citation.add_argument("--doc", required=True)
    citation.add_argument("--page", required=True, type=int)
    citation.add_argument("--bbox", required=True, help="x0,y0,x1,y1")

    return parser


def main() -> None:
    configure_logging(logging.INFO)
    args = build_parser().parse_args()
    settings = Settings(workspace_root=Path.cwd())
    pipeline = RefineryPipeline(settings)

    if args.command == "ingest":
        for pdf in _expand_inputs(args.pdfs):
            result = pipeline.ingest(pdf)
            print(json.dumps({"pdf": str(pdf), **result}, indent=2, ensure_ascii=False))
        return

    if args.command == "build-index":
        for pdf in _expand_inputs(args.pdfs):
            result = pipeline.build_index(pdf)
            print(json.dumps({"pdf": str(pdf), **result}, indent=2, ensure_ascii=False))
        return

    store = ArtifactStore(settings)
    vector_index = VectorIndex(store)
    fact_store = FactStore(store.db_dir / "facts.db")
    agent = QueryAgent(store, vector_index, fact_store)

    doc_path = Path(args.doc)
    doc_id = doc_id_from_file(doc_path) if doc_path.exists() else args.doc

    if args.command == "query":
        ans = agent.query(doc_id, args.question)
        # Append to query history
        try:
            record = {
                "timestamp": datetime.utcnow().isoformat() + "Z",
                "command": "query",
                "doc_id": doc_id,
                "question": args.question,
                "result": ans.model_dump(mode="json"),
            }
            store.append_jsonl(store.query_history_file, record)
        except Exception:
            pass
        print(json.dumps(ans.model_dump(mode="json"), indent=2, ensure_ascii=False))
        return

    if args.command == "query-interface":
        out = agent.query_interface(doc_id, args.question, include_navigation_debug=bool(args.debug_sections))
        # Always include the question in the output
        out_with_question = {"question": args.question, **out}
        # Append to query history (artifacts/query_history.jsonl)
        try:
            # Attempt to include doc_name from profile if available
            doc_name: str | None = None
            prof_path = store.profiles_dir / f"{doc_id}.json"
            if prof_path.exists():
                try:
                    prof_data = json.loads(prof_path.read_text(encoding="utf-8"))
                    doc_name = str(prof_data.get("doc_name")) if prof_data.get("doc_name") else None
                except Exception:
                    doc_name = None
            record = {
                "timestamp": datetime.utcnow().isoformat() + "Z",
                "command": "query-interface",
                "doc_id": doc_id,
                "doc_name": doc_name,
                "question": args.question,
                "result": out_with_question,
            }
            store.append_jsonl(store.query_history_file, record)
        except Exception:
            pass

        # Decide how to save the output when --output is provided.
        if getattr(args, "output", None):
            out_path = Path(args.output)
            data_to_write: object

            if getattr(args, "append_output", False):
                # Append semantics: keep a JSON array of entries in the output file.
                if out_path.exists():
                    try:
                        existing = json.loads(out_path.read_text(encoding="utf-8"))
                    except Exception:
                        existing = None

                    if isinstance(existing, list):
                        existing.append(out_with_question)
                        data_to_write = existing
                    elif isinstance(existing, dict):
                        # Existing single object -> promote to array
                        data_to_write = [existing, out_with_question]
                    else:
                        # Unknown/invalid content -> start a new array
                        data_to_write = [out_with_question]
                else:
                    # New file with append semantics: start with a single-entry array
                    data_to_write = [out_with_question]
            else:
                # Default behavior: overwrite with a single object (backwards compatible)
                data_to_write = out_with_question

            output_json = json.dumps(data_to_write, indent=2, ensure_ascii=False)
            out_path.write_text(output_json, encoding="utf-8")
            print(f"Output saved to {args.output}")
        else:
            # No output path -> print a single object
            print(json.dumps(out_with_question, indent=2, ensure_ascii=False))
        return

    if args.command == "audit":
        out = agent.audit_claim(doc_id, args.claim)
        print(json.dumps(out, indent=2, ensure_ascii=False))
        return

    if args.command == "show-pageindex":
        nodes = agent.navigate(doc_id, args.topic, top_k=max(int(args.top_k), 1))
        print(json.dumps(nodes, indent=2, ensure_ascii=False))
        return

    if args.command == "eval-pageindex":
        expected_sections = [s.strip() for s in str(args.expected).split(",") if s.strip()]
        out = agent.measure_retrieval_precision(
            doc_id=doc_id,
            topic=str(args.topic),
            expected_sections=expected_sections,
            top_k=max(int(args.top_k), 1),
        )
        print(json.dumps(out, indent=2, ensure_ascii=False))
        return

    if args.command == "open-citation":
        bbox = tuple(float(x.strip()) for x in args.bbox.split(","))
        if len(bbox) != 4:
            raise ValueError("bbox must be x0,y0,x1,y1")
        out = open_citation_snippet(store, doc_id, args.page, bbox)
        print(json.dumps(out, indent=2, ensure_ascii=False))
        return


if __name__ == "__main__":
    main()
