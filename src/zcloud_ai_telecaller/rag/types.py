"""Type definitions and schemas for Z Cloud Voice RAG (Retrieval-Augmented Generation).

Enforces strict tenant scoping, metadata tracking, and security boundaries.
"""

from __future__ import annotations

import datetime
from enum import Enum
from typing import Any, Optional
from pydantic import BaseModel, ConfigDict, Field


class IngestionStatus(str, Enum):
    """Lifecycle status of a knowledge document in the ingestion pipeline."""
    PENDING = "PENDING"
    UPLOADING = "UPLOADING"
    PROCESSING = "PROCESSING"
    INDEXING = "INDEXING"
    READY = "READY"
    FAILED = "FAILED"


class DocumentType(str, Enum):
    """Supported source file formats."""
    PDF = "PDF"
    DOCX = "DOCX"
    TXT = "TXT"
    MARKDOWN = "MARKDOWN"
    FAQ = "FAQ"


class EmbeddingProviderType(str, Enum):
    """Supported embedding generation backends."""
    LOCAL_ONNX = "local_onnx"
    GEMINI_CLOUD = "gemini_cloud"
    HYBRID = "hybrid"


class KnowledgeChunkMetadata(BaseModel):
    """Metadata attached to an individual passage chunk."""
    model_config = ConfigDict(extra="ignore")

    document_id: str
    tenant_id: str
    source_title: str
    business_type: str = "clinic"
    source_type: DocumentType = DocumentType.TXT
    page: Optional[int] = None
    section: Optional[str] = None
    chunk_index: int = 0
    total_chunks: int = 1
    is_faq: bool = False
    faq_question: Optional[str] = None


class KnowledgeChunk(BaseModel):
    """A single normalized, indexable text chunk."""
    model_config = ConfigDict(extra="ignore")

    id: str
    tenant_id: str
    document_id: str
    chunk_index: int
    content: str
    clean_content: str
    char_length: int
    token_count: int = 0
    metadata: KnowledgeChunkMetadata
    embedding: Optional[list[float]] = None
    created_at: datetime.datetime = Field(default_factory=lambda: datetime.datetime.now(datetime.timezone.utc))


class DocumentIngestionResult(BaseModel):
    """Report returned upon document parsing and chunking."""
    model_config = ConfigDict(extra="ignore")

    document_id: str
    tenant_id: str
    business_type: str
    title: str
    filename: Optional[str] = None
    mime_type: Optional[str] = None
    type: DocumentType
    size_bytes: int
    sha256_hash: str
    status: IngestionStatus
    chunk_count: int
    chunks: list[KnowledgeChunk] = Field(default_factory=list)
    embedding_model: str
    embedding_dimension: int
    index_version: str = "v1"
    doc_version: int = 1
    error_message: Optional[str] = None
    created_at: datetime.datetime = Field(default_factory=lambda: datetime.datetime.now(datetime.timezone.utc))


class RetrievalResult(BaseModel):
    """A scored passage retrieved from the vector index."""
    model_config = ConfigDict(extra="ignore")

    chunk_id: str
    document_id: str
    tenant_id: str
    content: str
    score: float  # Cosine similarity score [0.0 - 1.0]
    source_title: str
    source_type: str
    business_type: str
    page: Optional[int] = None
    section: Optional[str] = None
    is_faq: bool = False
    faq_question: Optional[str] = None


class RetrievalResponse(BaseModel):
    """Aggregated response containing authorized context for the voice agent."""
    model_config = ConfigDict(extra="ignore")

    tenant_id: str
    query: str
    results: list[RetrievalResult] = Field(default_factory=list)
    has_sufficient_evidence: bool = False
    top_score: float = 0.0
    retrieval_latency_ms: float = 0.0
    provider_used: str = "local_onnx"
    cloud_fallback_triggered: bool = False
    context_text: str = ""  # Formatted untrusted evidence for prompt injection defense
    grounding_guidance: str = ""
