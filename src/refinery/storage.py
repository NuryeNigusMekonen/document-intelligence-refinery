from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable

from .config import Settings


class ArtifactStore:
    def __init__(self, settings: Settings):
        self.settings = settings
        self.root = settings.resolved_artifacts_dir
        self.profiles_dir = self.root / "profiles"
        self.extracted_dir = self.root / "extracted"
        self.chunks_dir = self.root / "chunks"
        self.pageindex_dir = self.root / "pageindex"
        self.db_dir = self.root / "db"
        self.vector_dir = self.root / "vector"
        self.ledger_file = self.root / "extraction_ledger.jsonl"
        self.query_history_file = self.root / "query_history.jsonl"
        self._ensure_dirs()

    def _ensure_dirs(self) -> None:
        for d in [self.root, self.profiles_dir, self.extracted_dir, self.chunks_dir, self.pageindex_dir, self.db_dir, self.vector_dir]:
            d.mkdir(parents=True, exist_ok=True)
        if not self.ledger_file.exists():
            self.ledger_file.touch()
        if not self.query_history_file.exists():
            self.query_history_file.touch()

    def save_json(self, path: Path, data: dict) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

    def load_json(self, path: Path) -> dict:
        with path.open("r", encoding="utf-8") as f:
            return json.load(f)

    def append_jsonl(self, path: Path, row: dict) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    def write_jsonl(self, path: Path, rows: Iterable[dict]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as f:
            for row in rows:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")

    def read_jsonl(self, path: Path) -> list[dict]:
        if not path.exists():
            return []
        with path.open("r", encoding="utf-8") as f:
            return [json.loads(line) for line in f if line.strip()]
