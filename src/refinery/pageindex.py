from __future__ import annotations

import logging
import math
import re
import time
from collections import Counter
from dataclasses import dataclass, field
from typing import Literal
import json
from urllib import request

from .models import LogicalDocumentUnit, PageIndex, SectionNode
from .storage import ArtifactStore

logger = logging.getLogger(__name__)


ENTITY_RE = re.compile(r"\b[A-Z][a-zA-Z]{2,}(?:\s+[A-Z][a-zA-Z]{2,})*\b")
SENTENCE_RE = re.compile(r"[^.!?]+[.!?]")
EQUATION_RE = re.compile(r"=|\b\d+\s*[+\-*/]\s*\d+")


@dataclass
class _NodeState:
    title: str
    path: tuple[str, ...]
    chunks: list[LogicalDocumentUnit] = field(default_factory=list)
    children: set[tuple[str, ...]] = field(default_factory=set)


class PageIndexBuilder:
    def __init__(self, store: ArtifactStore):
        self.store = store
        self._summary_cache_path = self.store.pageindex_dir / "summary_cache.json"

    def _extract_entities(self, text: str) -> list[str]:
        entities = ENTITY_RE.findall(text)
        counts = Counter(entities)
        return [e for e, _ in counts.most_common(5)]

    def _summarize_deterministic(self, chunks: list[LogicalDocumentUnit]) -> str:
        text_parts: list[str] = []
        for c in chunks:
            if c.content:
                text_parts.append(c.content.strip())
        merged = " ".join(text_parts).strip()
        if not merged:
            return "This section contains extracted content. It may include textual or structured data."

        sentences = [s.strip() for s in SENTENCE_RE.findall(merged) if s.strip()]
        if len(sentences) >= 2:
            return " ".join(sentences[:3])

        snippet = merged[:280].strip()
        if not snippet.endswith("."):
            snippet += "."
        return f"{snippet} This section groups related evidence for focused retrieval."

    def _summary_sentence_count(self, text: str) -> int:
        return len([s for s in SENTENCE_RE.findall(text or "") if s.strip()])

    def _enforce_sentence_window(self, candidate: str, fallback: str) -> str:
        sentence_list = [s.strip() for s in SENTENCE_RE.findall(candidate or "") if s.strip()]
        if len(sentence_list) >= 2:
            return " ".join(sentence_list[:3])

        fallback_sentences = [s.strip() for s in SENTENCE_RE.findall(fallback or "") if s.strip()]
        combined = sentence_list + fallback_sentences
        if not combined:
            combined = ["This section contains extracted content.", "It supports focused retrieval and navigation."]
        if len(combined) == 1:
            combined.append("It supports focused retrieval and navigation.")
        return " ".join(combined[:3])

    def _load_summary_cache(self) -> dict[str, str]:
        if not self._summary_cache_path.exists():
            return {}
        try:
            data = self.store.load_json(self._summary_cache_path)
            if isinstance(data, dict):
                return {str(k): str(v) for k, v in data.items()}
        except Exception:
            return {}
        return {}

    def _save_summary_cache(self, cache: dict[str, str]) -> None:
        self.store.save_json(self._summary_cache_path, cache)

    def _summary_cache_key(self, doc_id: str, path: tuple[str, ...]) -> str:
        return f"{doc_id}::{' > '.join(path)}"

    def _generate_summary_ollama(self, section_title: str, chunks: list[LogicalDocumentUnit]) -> str | None:
        if not self.store.settings.use_ollama_summaries:
            return None

        content_preview = " ".join((c.content or "") for c in chunks[:12]).strip()[:4000]
        prompt = (
            "Write a concise section summary in exactly 2 to 3 sentences. "
            "Focus on key facts and retrievable intent.\n"
            f"Section title: {section_title}\n"
            f"Section content:\n{content_preview}\n"
        )
        payload = {
            "model": self.store.settings.ollama_model,
            "prompt": prompt,
            "stream": False,
        }
        try:
            host = self.store.settings.ollama_host.rstrip("/")
            req = request.Request(
                f"{host}/api/generate",
                data=json.dumps(payload).encode("utf-8"),
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            with request.urlopen(req, timeout=15) as resp:
                raw = resp.read().decode("utf-8", errors="ignore")
            data = json.loads(raw)
            out = str(data.get("response", "")).strip()
            return out or None
        except Exception:
            return None

    def _collect_data_types(self, chunks: list[LogicalDocumentUnit]) -> list[Literal["tables", "figures", "equations", "lists"]]:
        out: list[Literal["tables", "figures", "equations", "lists"]] = []
        if any(c.chunk_type == "table" for c in chunks):
            out.append("tables")
        if any(c.chunk_type == "figure" for c in chunks):
            out.append("figures")
        if any(c.chunk_type == "list" for c in chunks):
            out.append("lists")
        if any(c.content and EQUATION_RE.search(c.content) for c in chunks):
            out.append("equations")
        return out

    def _all_chunks(self, path: tuple[str, ...], nodes: dict[tuple[str, ...], _NodeState]) -> list[LogicalDocumentUnit]:
        node = nodes[path]
        merged = list(node.chunks)
        for child in sorted(node.children):
            merged.extend(self._all_chunks(child, nodes))
        return merged

    def _build_section(
        self,
        doc_id: str,
        path: tuple[str, ...],
        nodes: dict[tuple[str, ...], _NodeState],
        summary_cache: dict[str, str],
    ) -> SectionNode:
        node = nodes[path]
        full_chunks = self._all_chunks(path, nodes)
        pages = sorted({ref.page_number for c in full_chunks for ref in c.page_refs if ref.page_number is not None}) or [1]
        text_join = " ".join([c.content or "" for c in full_chunks])

        children = [self._build_section(doc_id, child, nodes, summary_cache) for child in sorted(node.children)]

        cache_key = self._summary_cache_key(doc_id, path)
        fallback_summary = self._summarize_deterministic(full_chunks)
        if cache_key in summary_cache:
            summary = self._enforce_sentence_window(summary_cache[cache_key], fallback_summary)
        else:
            generated = self._generate_summary_ollama(node.title, full_chunks)
            if generated:
                summary = self._enforce_sentence_window(generated, fallback_summary)
                summary_cache[cache_key] = summary
            else:
                summary = self._enforce_sentence_window(fallback_summary, fallback_summary)
                summary_cache[cache_key] = summary

        return SectionNode(
            title=node.title,
            page_start=min(pages),
            page_end=max(pages),
            child_sections=children,
            key_entities=self._extract_entities(text_join),
            summary=summary,
            data_types_present=self._collect_data_types(full_chunks),
        )

    def build(self, doc_id: str, doc_name: str, ldus: list[LogicalDocumentUnit]) -> PageIndex:
        started = time.perf_counter()
        logger.info("stage=pageindex start doc=%s", doc_name)
        summary_cache = self._load_summary_cache()

        nodes: dict[tuple[str, ...], _NodeState] = {}

        def ensure(path: tuple[str, ...]) -> _NodeState:
            if path not in nodes:
                title = path[-1] if path else "Document Root"
                nodes[path] = _NodeState(title=title, path=path)
            return nodes[path]

        root_path = ("Document Root",)
        ensure(root_path)

        for ldu in ldus:
            parts = tuple(ldu.parent_section_path) if ldu.parent_section_path else root_path
            if parts == ():
                parts = root_path
            for i in range(1, len(parts) + 1):
                path = parts[:i]
                ensure(path)
                if i > 1:
                    ensure(parts[: i - 1]).children.add(path)
                elif path != root_path:
                    nodes[root_path].children.add(path)
            ensure(parts).chunks.append(ldu)

        root_sections: list[SectionNode] = []
        top = sorted(nodes[root_path].children) if nodes[root_path].children else [root_path]
        for path in top:
            root_sections.append(self._build_section(doc_id, path, nodes, summary_cache))

        self._save_summary_cache(summary_cache)

        pageindex = PageIndex(doc_id=doc_id, doc_name=doc_name, root_sections=root_sections)
        self.store.save_json(self.store.pageindex_dir / f"{doc_id}.json", pageindex.model_dump(mode="json"))
        elapsed = int((time.perf_counter() - started) * 1000)
        logger.info("stage=pageindex end doc=%s sections=%s duration_ms=%s", doc_name, len(root_sections), elapsed)
        return pageindex


def _flatten_sections(sections: list[SectionNode]) -> list[SectionNode]:
    out: list[SectionNode] = []
    stack = list(sections)
    while stack:
        node = stack.pop(0)
        out.append(node)
        if node.child_sections:
            stack = list(node.child_sections) + stack
    return out


def pageindex_navigate(pageindex: PageIndex, query: str, top_k: int = 5) -> list[SectionNode]:
    candidates = _flatten_sections(pageindex.root_sections)
    query_terms = [t.lower() for t in re.findall(r"\w+", query)]
    if not query_terms:
        return candidates[:top_k]

    docs = [s.title + " " + s.summary + " " + " ".join(s.key_entities) for s in candidates]
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
    return [candidates[idx] for score, idx in ranked if score > 0][:top_k] or candidates[:top_k]
