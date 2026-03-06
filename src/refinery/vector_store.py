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
        self._docs_by_id: dict[str, dict[str, LogicalDocumentUnit]] = {}
        self._chroma_client = None
        self._collections: dict[str, object] = {}
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
        self._fallback_docs[doc_id] = ldus
        self._docs_by_id[doc_id] = {ldu.ldu_id: ldu for ldu in ldus}

        if not self._chroma_client:
            return

        try:
            collection = self._chroma_client.get_or_create_collection(name=f"doc_{doc_id}")
            ids: list[str] = []
            docs: list[str] = []
            metadatas: list[dict[str, str]] = []
            seen_ids: dict[str, int] = {}
            for ldu in ldus:
                text = (ldu.content or str(ldu.structured_payload or "")).strip()
                if not text:
                    continue
                base_id = str(ldu.ldu_id)
                count = seen_ids.get(base_id, 0)
                seen_ids[base_id] = count + 1
                chroma_id = base_id if count == 0 else f"{base_id}__dup{count}"
                ids.append(chroma_id)
                docs.append(text)
                metadatas.append(
                    {
                        "doc_id": doc_id,
                        "chunk_type": ldu.chunk_type,
                        "content_hash": ldu.content_hash,
                        "parent_section": ldu.parent_section or "",
                        "ldu_id": base_id,
                    }
                )

            if ids:
                collection.upsert(ids=ids, documents=docs, metadatas=metadatas)
            self._collections[doc_id] = collection
        except Exception as e:
            logger.warning("chroma upsert failed for doc=%s, using lexical fallback: %s", doc_id, e)

    def search(self, doc_id: str, query: str, top_k: int = 5) -> list[LogicalDocumentUnit]:
        if self._chroma_client:
            collection = self._collections.get(doc_id)
            if collection is None:
                try:
                    collection = self._chroma_client.get_or_create_collection(name=f"doc_{doc_id}")
                    self._collections[doc_id] = collection
                except Exception:
                    collection = None

            if collection is not None:
                try:
                    result = collection.query(query_texts=[query], n_results=max(int(top_k), 1), include=["metadatas"])
                    ids_rows = result.get("ids") or []
                    metadata_rows = result.get("metadatas") or []
                    ids = ids_rows[0] if ids_rows else []
                    metadata = metadata_rows[0] if metadata_rows else []
                    doc_map = self._docs_by_id.get(doc_id, {})
                    hits: list[LogicalDocumentUnit] = []
                    for idx, chroma_id in enumerate(ids):
                        ldu_id = ""
                        if idx < len(metadata) and isinstance(metadata[idx], dict):
                            ldu_id = str(metadata[idx].get("ldu_id") or "")
                        if not ldu_id:
                            ldu_id = str(chroma_id).split("__dup", 1)[0]
                        if ldu_id in doc_map:
                            hits.append(doc_map[ldu_id])
                    if hits:
                        return hits[:top_k]
                except Exception as e:
                    logger.warning("chroma query failed for doc=%s, using lexical fallback: %s", doc_id, e)

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
