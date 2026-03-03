from __future__ import annotations

import logging
import math
import re
import time
from collections import Counter
from typing import Literal

from .models import LogicalDocumentUnit, PageIndex, SectionNode
from .storage import ArtifactStore

logger = logging.getLogger(__name__)


ENTITY_RE = re.compile(r"\b[A-Z][a-zA-Z]{2,}(?:\s+[A-Z][a-zA-Z]{2,})*\b")


class PageIndexBuilder:
    def __init__(self, store: ArtifactStore):
        self.store = store

    def _extract_entities(self, text: str) -> list[str]:
        entities = ENTITY_RE.findall(text)
        counts = Counter(entities)
        return [e for e, _ in counts.most_common(5)]

    def build(self, doc_id: str, doc_name: str, ldus: list[LogicalDocumentUnit]) -> PageIndex:
        started = time.perf_counter()
        logger.info("stage=pageindex start doc=%s", doc_name)

        by_section: dict[str, list[LogicalDocumentUnit]] = {}
        for ldu in ldus:
            title = " > ".join(ldu.parent_section_path) if ldu.parent_section_path else "Document Root"
            by_section.setdefault(title, []).append(ldu)

        sections: list[SectionNode] = []
        for title, chunks in by_section.items():
            pages = sorted({ref.page_number for ldu in chunks for ref in ldu.page_refs if ref.page_number is not None}) or [1]
            text_join = " ".join([ldu.content or "" for ldu in chunks])
            summary = " ".join((ldu.content or "") for ldu in chunks[:2]).strip()[:260] or "No summary"
            data_types: list[Literal["tables", "figures", "equations", "lists"]] = []
            if any(ldu.chunk_type == "table" for ldu in chunks):
                data_types.append("tables")
            if any(ldu.chunk_type == "figure" for ldu in chunks):
                data_types.append("figures")
            if any(ldu.chunk_type == "list" for ldu in chunks):
                data_types.append("lists")
            sections.append(
                SectionNode(
                    title=title,
                    page_start=min(pages),
                    page_end=max(pages),
                    key_entities=self._extract_entities(text_join),
                    summary=summary,
                    data_types_present=data_types,
                )
            )

        pageindex = PageIndex(doc_id=doc_id, doc_name=doc_name, root_sections=sections)
        self.store.save_json(self.store.pageindex_dir / f"{doc_id}.json", pageindex.model_dump(mode="json"))
        elapsed = int((time.perf_counter() - started) * 1000)
        logger.info("stage=pageindex end doc=%s sections=%s duration_ms=%s", doc_name, len(sections), elapsed)
        return pageindex


def pageindex_navigate(pageindex: PageIndex, query: str, top_k: int = 5) -> list[SectionNode]:
    query_terms = [t.lower() for t in re.findall(r"\w+", query)]
    if not query_terms:
        return pageindex.root_sections[:top_k]

    docs = [s.title + " " + s.summary + " " + " ".join(s.key_entities) for s in pageindex.root_sections]
    tokenized_docs = [[t.lower() for t in re.findall(r"\w+", d)] for d in docs]
    avg_dl = sum(len(d) for d in tokenized_docs) / max(len(tokenized_docs), 1)

    scores: list[tuple[float, int]] = []
    for idx, doc_terms in enumerate(tokenized_docs):
        dl = len(doc_terms)
        term_counts = Counter(doc_terms)
        score = 0.0
        for q in query_terms:
            tf = term_counts.get(q, 0)
            if tf == 0:
                continue
            df = sum(1 for d in tokenized_docs if q in d)
            idf = math.log((len(tokenized_docs) - df + 0.5) / (df + 0.5) + 1)
            k1 = 1.5
            b = 0.75
            score += idf * ((tf * (k1 + 1)) / (tf + k1 * (1 - b + b * (dl / max(avg_dl, 1)))))
        scores.append((score, idx))

    ranked = sorted(scores, key=lambda x: x[0], reverse=True)
    return [pageindex.root_sections[idx] for score, idx in ranked if score > 0][:top_k] or pageindex.root_sections[:top_k]
