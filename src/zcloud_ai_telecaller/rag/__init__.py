"""Z Cloud Voice Phase 3 Multi-Tenant RAG Knowledge Base package."""

from zcloud_ai_telecaller.rag.types import (
    DocumentIngestionResult,
    DocumentType,
    IngestionStatus,
    KnowledgeChunk,
    KnowledgeChunkMetadata,
    RetrievalResponse,
    RetrievalResult,
)
from zcloud_ai_telecaller.rag.extractors import (
    DocumentExtractionError,
    SafeTextExtractor,
    compute_sha256,
    normalize_extracted_text,
)
from zcloud_ai_telecaller.rag.chunker import KnowledgeChunker, chunk_text
from zcloud_ai_telecaller.rag.vector_store import PersistentTenantVectorStore, cosine_similarity
from zcloud_ai_telecaller.rag.retriever import TenantKnowledgeRetriever
from zcloud_ai_telecaller.rag.embeddings.base import (
    BaseEmbeddingProvider,
    EmbeddingDimensionMismatchError,
    EmbeddingProviderError,
    EmbeddingTimeoutError,
)
from zcloud_ai_telecaller.rag.embeddings.local_provider import LocalOnnxEmbeddingProvider
from zcloud_ai_telecaller.rag.embeddings.cloud_provider import GeminiEmbeddingProvider
from zcloud_ai_telecaller.rag.embeddings.hybrid_provider import HybridEmbeddingProvider

__all__ = [
    "DocumentIngestionResult",
    "DocumentType",
    "IngestionStatus",
    "KnowledgeChunk",
    "KnowledgeChunkMetadata",
    "RetrievalResponse",
    "RetrievalResult",
    "DocumentExtractionError",
    "SafeTextExtractor",
    "compute_sha256",
    "normalize_extracted_text",
    "KnowledgeChunker",
    "chunk_text",
    "PersistentTenantVectorStore",
    "cosine_similarity",
    "TenantKnowledgeRetriever",
    "BaseEmbeddingProvider",
    "EmbeddingDimensionMismatchError",
    "EmbeddingProviderError",
    "EmbeddingTimeoutError",
    "LocalOnnxEmbeddingProvider",
    "GeminiEmbeddingProvider",
    "HybridEmbeddingProvider",
]
