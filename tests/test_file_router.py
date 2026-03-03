from pathlib import Path

from refinery.config import Settings
from refinery.file_router import FileRouter


def test_file_router_detects_supported_types(tmp_path: Path):
    router = FileRouter(Settings(workspace_root=tmp_path))
    assert router.detect_type(Path("a.pdf")) == "pdf"
    assert router.detect_type(Path("a.docx")) == "docx"
    assert router.detect_type(Path("a.md")) == "markdown"
    assert router.detect_type(Path("a.png")) == "image"
    assert router.detect_type(Path("a.jpg")) == "image"
    assert router.detect_type(Path("a.xlsx")) == "excel"
