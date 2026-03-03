from __future__ import annotations

import logging
import re
from collections import Counter
from math import sqrt

from .config import Settings
from .models import LogicalDocumentUnit
from .storage import ArtifactStore

logger = logging.getLogger(__name__)


def _tokenize(text: str, multilingual: bool) -> list[str]:
    if multilingual:
        return [t.lower() for t in re.findall(r"\w+", text, flags=re.UNICODE)]
    return [t.lower() for t in re.findall(r"[A-Za-z0-9_]+", text)]


def _has_geez(text: str) -> bool:
    return bool(re.search(r"[\u1200-\u137F]", text or ""))


def _cosine(a: Counter[str], b: Counter[str]) -> float:
    if not a or not b:
        return 0.0
    keys = set(a.keys()) & set(b.keys())
    dot = sum(a[k] * b[k] for k in keys)
    na = sqrt(sum(v * v for v in a.values()))
    nb = sqrt(sum(v * v for v in b.values()))
    if na == 0 or nb == 0:
        return 0.0
    return dot / (na * nb)


class VectorIndex:
    def __init__(self, store: ArtifactStore):
        self.store = store
        self.settings: Settings = store.settings
        self._fallback_docs: dict[str, list[LogicalDocumentUnit]] = {}
        self._chroma_client = None
        self._collection = None
        try:
            import chromadb

            self._chroma_client = chromadb.PersistentClient(path=str(store.vector_dir))
        except Exception as e:
            logger.warning("chromadb unavailable, using lexical fallback: %s", e)
        logger.info(
            "vector_index init embedding_model=%s multilingual_embeddings=%s",
            self.settings.embedding_model,
            self.settings.multilingual_embeddings,
        )

    def build(self, doc_id: str, ldus: list[LogicalDocumentUnit]) -> None:
        if self._chroma_client:
            self._collection = self._chroma_client.get_or_create_collection(name=f"doc_{doc_id}")
        self._fallback_docs[doc_id] = ldus

    def search(self, doc_id: str, query: str, top_k: int = 5) -> list[LogicalDocumentUnit]:
        docs = self._fallback_docs.get(doc_id, [])
        qv = Counter(_tokenize(query, self.settings.multilingual_embeddings))
        query_has_geez = _has_geez(query)
        ranked = []
        for ldu in docs:
            text = ldu.content or str(ldu.structured_payload or "")
            score = _cosine(qv, Counter(_tokenize(text, self.settings.multilingual_embeddings)))
            if self.settings.multilingual_embeddings and query_has_geez and _has_geez(text):
                score += 0.25
            ranked.append((score, ldu))
        lexical_hits = [ldu for score, ldu in sorted(ranked, key=lambda x: x[0], reverse=True)[:top_k] if score > 0]
        return lexical_hits
