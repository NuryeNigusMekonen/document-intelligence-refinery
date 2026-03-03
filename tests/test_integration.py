from pathlib import Path

import pytest

from refinery.config import Settings
from refinery.pipeline import RefineryPipeline


def test_pipeline_integration_if_pdf_present():
    data_dir = Path("data")
    pdfs = sorted(data_dir.glob("*.pdf"))
    if not pdfs:
        pytest.skip("No PDF in data/; integration test skipped.")

    settings = Settings(workspace_root=Path.cwd())
    pipeline = RefineryPipeline(settings)
    result = pipeline.ingest(pdfs[0])
    assert "doc_id" in result
    assert result["chunks"] >= 0
