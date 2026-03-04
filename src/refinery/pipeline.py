from __future__ import annotations

import json
import logging
import time
from pathlib import Path

from .chunking import Chunker
from .config import Settings
from .extraction import ExtractionRouter
from .facts import FactStore
from .file_router import FileRouter
from .file_types import build_non_pdf_profile, doc_id_from_file
from .lang_detect import detect_language
from .models import DocumentProfile, ExtractedDocument, LanguageHint
from .pageindex import PageIndexBuilder
from .storage import ArtifactStore
from .triage import TriageAgent
from .vector_store import VectorIndex

logger = logging.getLogger(__name__)


def doc_id_from_pdf(pdf_path: Path) -> str:
    return doc_id_from_file(pdf_path)


class RefineryPipeline:
    def __init__(self, settings: Settings):
        self.settings = settings
        self.store = ArtifactStore(settings)
        self.file_router = FileRouter(settings)
        self.triage = TriageAgent(self.store, settings)
        self.router = ExtractionRouter(settings, self.store)
        self.chunker = Chunker(settings, self.store)
        self.pageindex_builder = PageIndexBuilder(self.store)
        self.vector_index = VectorIndex(self.store)
        self.fact_store = FactStore(self.store.db_dir / "facts.db")

    def triage_only(self, file_path: Path) -> DocumentProfile:
        if self.file_router.detect_type(file_path) == "pdf":
            return self.triage.run(file_path)
        profile = build_non_pdf_profile(file_path, self.file_router)
        self.store.save_json(self.store.profiles_dir / f"{profile.doc_id}.json", profile.model_dump(mode="json"))
        return profile

    def _enrich_profile_language_from_extracted(self, profile: DocumentProfile, extracted: ExtractedDocument) -> DocumentProfile:
        texts: list[str] = []
        for page in extracted.pages:
            for block in page.blocks:
                txt = (block.text or "").strip()
                if txt:
                    texts.append(txt)
                if len(texts) >= 120:
                    break
            if len(texts) >= 120:
                break

        sample = "\n".join(texts)[:12000]
        if not sample.strip():
            return profile

        detected = detect_language(sample, mode=self.settings.language_detection_mode)
        language = str(detected.get("language", "unknown"))
        confidence = float(detected.get("confidence", 0.0) or 0.0)
        if language in {"am", "en"}:
            profile.language_hint = LanguageHint(language=language, confidence=confidence)
        return profile

    def extract_only(self, file_path: Path) -> tuple[DocumentProfile, ExtractedDocument]:
        ftype = self.file_router.detect_type(file_path)
        profile = self.triage_only(file_path)
        if ftype == "pdf":
            extracted = self.router.extract(file_path, profile)
            profile = self._enrich_profile_language_from_extracted(profile, extracted)
        else:
            extracted, confidence, strategy = self.file_router.extract_non_pdf(
                file_path,
                profile.doc_id,
                profile_language=profile.language_hint.language,
                profile_conf=profile.language_hint.confidence,
            )
            profile = self._enrich_profile_language_from_extracted(profile, extracted)
            ocr_lang_used = None
            if "ocr_lang_used=" in strategy:
                ocr_lang_used = strategy.split("ocr_lang_used=", 1)[1].split(" ", 1)[0]
            quality_agg = self.router._aggregate_page_quality(extracted) if extracted is not None else {}
            self.router._write_ledger(
                doc_id=profile.doc_id,
                origin_type=profile.origin_type,
                layout_complexity=profile.layout_complexity,
                strategy=f"{strategy}:{ftype}",
                confidence=confidence,
                cost_estimate=0.0,
                duration_ms=0,
                pages_processed=len(extracted.pages) if extracted is not None else None,
                blocks_extracted=sum(len(page.blocks) for page in extracted.pages) if extracted is not None else None,
                tables_extracted=sum(len(page.tables) for page in extracted.pages) if extracted is not None else None,
                table_pages_count=sum(1 for page in extracted.pages if page.tables) if extracted is not None else None,
                ocr_word_conf_mean=quality_agg.get("ocr_word_conf_mean"),
                ethiopic_ratio=quality_agg.get("ethiopic_ratio"),
                garbage_ratio=quality_agg.get("garbage_ratio"),
                avg_word_len=quality_agg.get("avg_word_len"),
                alpha_ratio=quality_agg.get("alpha_ratio"),
                token_diversity=quality_agg.get("token_diversity"),
                escalations=[],
                notes=f"non-pdf extraction for {file_path.name}",
                detected_language=profile.language_hint.language,
                ocr_lang_used=ocr_lang_used,
            )
        self.store.save_json(self.store.profiles_dir / f"{profile.doc_id}.json", profile.model_dump(mode="json"))
        self.store.save_json(self.store.extracted_dir / f"{profile.doc_id}.json", extracted.model_dump(mode="json"))
        return profile, extracted

    def ingest(self, file_path: Path) -> dict:
        started = time.perf_counter()
        profile, extracted = self.extract_only(file_path)
        ldus = self.chunker.run(extracted)
        self.vector_index.build(profile.doc_id, ldus)
        facts = self.fact_store.ingest_ldus(profile.doc_id, ldus)
        elapsed = int((time.perf_counter() - started) * 1000)
        logger.info("pipeline=ingest end doc=%s chunks=%s facts=%s duration_ms=%s", profile.doc_name, len(ldus), facts, elapsed)
        return {"doc_id": profile.doc_id, "chunks": len(ldus), "facts": facts}

    def build_index(self, file_path: Path) -> dict:
        existing_doc_id = doc_id_from_file(file_path)
        existing_profile_path = self.store.profiles_dir / f"{existing_doc_id}.json"
        if existing_profile_path.exists():
            profile = DocumentProfile.model_validate(json.loads(existing_profile_path.read_text(encoding="utf-8")))
        else:
            profile = self.triage_only(file_path)
        rows = self.store.read_jsonl(self.store.chunks_dir / f"{profile.doc_id}.jsonl")
        if not rows:
            result = self.ingest(file_path)
            rows = self.store.read_jsonl(self.store.chunks_dir / f"{result['doc_id']}.jsonl")
            doc_id = result["doc_id"]
        else:
            doc_id = profile.doc_id
        from .models import LogicalDocumentUnit

        ldus = [LogicalDocumentUnit.model_validate(r) for r in rows]
        pageindex = self.pageindex_builder.build(doc_id, profile.doc_name, ldus)
        return {"doc_id": doc_id, "sections": len(pageindex.root_sections)}
