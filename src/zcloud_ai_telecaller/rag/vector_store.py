"""Persistent, tenant-isolated vector store for Z Cloud Voice RAG.

Design principles:
1. Multi-tenant isolation: Tenant chunks and vector indices are strictly partitioned by tenant_id.
2. Durable persistence: Serialized to disk in SQLite / structured index files; reloadable on worker startup.
3. Fully rebuildable: Index can be rebuilt from authoritative document chunks at any time.
4. Cosine similarity scoring with relevance thresholding and duplicate suppression.
5. Immediate deletion propagation when documents are updated or removed.
"""

from __future__ import annotations

import json
import logging
import math
import os
import threading
from pathlib import Path
from typing import Any, Optional

from zcloud_ai_telecaller.rag.types import KnowledgeChunk, RetrievalResult

logger = logging.getLogger("zcloud_ai_telecaller.rag.vector_store")


def cosine_similarity(v1: list[float], v2: list[float]) -> float:
    """Computes cosine similarity between two float vectors."""
    if len(v1) != len(v2) or not v1:
        return 0.0

    dot_product = 0.0
    norm_a = 0.0
    norm_b = 0.0

    for a, b in zip(v1, v2):
        dot_product += a * b
        norm_a += a * a
        norm_b += b * b

    if norm_a <= 0.0 or norm_b <= 0.0:
        return 0.0

    return max(0.0, min(1.0, dot_product / (math.sqrt(norm_a) * math.sqrt(norm_b))))


class PersistentTenantVectorStore:
    """Authoritative vector repository partitioned strictly per tenant."""

    def __init__(self, storage_dir: Optional[str] = None) -> None:
        if storage_dir:
            self.storage_dir = Path(storage_dir)
        else:
            base_dir = Path(__file__).resolve().parent.parent.parent.parent
            self.storage_dir = base_dir / "storage" / "vector_indices"

        self.storage_dir.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()
        # In-memory tenant index cache: tenant_id -> list of chunk dicts
        self._cache: dict[str, list[dict[str, Any]]] = {}

    def _get_tenant_file(self, tenant_id: str) -> Path:
        """Returns safe path for tenant vector index file."""
        safe_tenant = "".join(c for c in tenant_id if c.isalnum() or c in ("-", "_"))
        return self.storage_dir / f"tenant_{safe_tenant}_index.json"

    def _load_tenant_index(self, tenant_id: str) -> list[dict[str, Any]]:
        """Loads tenant index from disk into cache."""
        with self._lock:
            if tenant_id in self._cache:
                return self._cache[tenant_id]

            file_path = self._get_tenant_file(tenant_id)
            if not file_path.exists():
                self._cache[tenant_id] = []
                return []

            try:
                with open(file_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    records = data.get("records", [])
                    self._cache[tenant_id] = records
                    return records
            except Exception as exc:
                logger.error("Failed to load vector index for tenant %s: %s", tenant_id, exc)
                self._cache[tenant_id] = []
                return []

    def _save_tenant_index(self, tenant_id: str, records: list[dict[str, Any]]) -> None:
        """Persists tenant index to disk atomically."""
        with self._lock:
            self._cache[tenant_id] = records
            file_path = self._get_tenant_file(tenant_id)
            temp_path = file_path.with_suffix(".tmp")
            payload = {
                "tenantId": tenant_id,
                "recordCount": len(records),
                "records": records,
            }
            try:
                with open(temp_path, "w", encoding="utf-8") as f:
                    json.dump(payload, f)
                # Atomic replace
                temp_path.replace(file_path)
            except Exception as exc:
                logger.error("Failed to write vector index for tenant %s: %s", tenant_id, exc)
                if temp_path.exists():
                    try:
                        temp_path.unlink()
                    except Exception:
                        pass
                raise

    def upsert_chunks(
        self,
        tenant_id: str,
        document_id: str,
        chunks: list[KnowledgeChunk],
        vectors: list[list[float]],
        model_name: str,
    ) -> int:
        """Inserts or updates chunks for a document in tenant index."""
        if not chunks or not vectors or len(chunks) != len(vectors):
            raise ValueError("Chunks and vectors must be non-empty and matching in length")

        with self._lock:
            records = self._load_tenant_index(tenant_id)
            # Remove existing chunks for this document to ensure idempotency
            filtered = [r for r in records if r.get("documentId") != document_id]

            for chunk, vec in zip(chunks, vectors):
                record = {
                    "chunkId": chunk.id,
                    "documentId": document_id,
                    "tenantId": tenant_id,
                    "chunkIndex": chunk.chunk_index,
                    "content": chunk.content,
                    "cleanContent": chunk.clean_content,
                    "sourceTitle": chunk.metadata.source_title,
                    "sourceType": str(chunk.metadata.source_type),
                    "businessType": chunk.metadata.business_type,
                    "page": chunk.metadata.page,
                    "section": chunk.metadata.section,
                    "isFaq": chunk.metadata.is_faq,
                    "faqQuestion": chunk.metadata.faq_question,
                    "embeddingModel": model_name,
                    "vector": vec,
                }
                filtered.append(record)

            self._save_tenant_index(tenant_id, filtered)
            logger.info("Upserted %d chunks for doc %s in tenant %s", len(chunks), document_id, tenant_id)
            return len(chunks)

    def search(
        self,
        tenant_id: str,
        query_vector: list[float],
        top_k: int = 3,
        min_score: float = 0.65,
    ) -> list[RetrievalResult]:
        """Performs tenant-isolated vector search using cosine similarity.

        Guarantees:
        - Only records belonging to `tenant_id` are examined.
        - Results below `min_score` are filtered out.
        - Near-duplicate passages are de-duplicated.
        """
        with self._lock:
            records = self._load_tenant_index(tenant_id)
            if not records:
                return []

            scored: list[tuple[float, dict[str, Any]]] = []
            for r in records:
                doc_vec = r.get("vector")
                if not doc_vec:
                    continue
                score = cosine_similarity(query_vector, doc_vec)
                if score >= min_score:
                    scored.append((score, r))

            # Sort descending by score
            scored.sort(key=lambda x: x[0], reverse=True)

            results: list[RetrievalResult] = []
            seen_texts: set[str] = set()

            for score, r in scored:
                # De-duplicate near-identical texts
                short_text = r["content"][:100].strip().lower()
                if short_text in seen_texts:
                    continue
                seen_texts.add(short_text)

                results.append(
                    RetrievalResult(
                        chunk_id=r["chunkId"],
                        document_id=r["documentId"],
                        tenant_id=tenant_id,
                        content=r["content"],
                        score=round(score, 4),
                        source_title=r["sourceTitle"],
                        source_type=r["sourceType"],
                        business_type=r.get("businessType", "clinic"),
                        page=r.get("page"),
                        section=r.get("section"),
                        is_faq=r.get("isFaq", False),
                        faq_question=r.get("faqQuestion"),
                    )
                )
                if len(results) >= top_k:
                    break

            return results

    def delete_document(self, tenant_id: str, document_id: str) -> int:
        """Removes all chunks for a document from tenant index."""
        with self._lock:
            records = self._load_tenant_index(tenant_id)
            initial_count = len(records)
            filtered = [r for r in records if r.get("documentId") != document_id]
            removed = initial_count - len(filtered)
            if removed > 0:
                self._save_tenant_index(tenant_id, filtered)
                logger.info("Deleted %d chunks for doc %s in tenant %s", removed, document_id, tenant_id)
            return removed

    def clear_tenant(self, tenant_id: str) -> int:
        """Wipes all indexed chunks for a tenant."""
        with self._lock:
            records = self._load_tenant_index(tenant_id)
            count = len(records)
            self._save_tenant_index(tenant_id, [])
            return count

    def get_chunk_count(self, tenant_id: str) -> int:
        """Returns total active chunks for a tenant."""
        with self._lock:
            return len(self._load_tenant_index(tenant_id))
