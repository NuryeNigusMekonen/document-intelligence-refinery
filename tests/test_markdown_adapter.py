from pathlib import Path

from refinery.adapters.markdown_adapter import MarkdownAdapter
from refinery.chunking import Chunker
from refinery.config import Settings
from refinery.storage import ArtifactStore


def test_markdown_adapter_and_chunking(tmp_path: Path):
    md = tmp_path / "doc.md"
    md.write_text(
        "# Title\n\n## Financial Results\nRevenue grew 10%.\n\n- item one\n- item two\n\n| A | B |\n|---|---|\n| 1 | 2 |\n",
        encoding="utf-8",
    )
    adapter = MarkdownAdapter()
    extracted = adapter.extract(md, "doc_md")
    assert extracted.pages
    assert extracted.pages[0].blocks or extracted.pages[0].tables

    settings = Settings(workspace_root=tmp_path)
    store = ArtifactStore(settings)
    chunker = Chunker(settings, store)
    ldus = chunker.run(extracted)
    assert ldus
    assert all(ref.ref_type in {"markdown_lines", "pdf_bbox"} for ldu in ldus for ref in ldu.page_refs)
