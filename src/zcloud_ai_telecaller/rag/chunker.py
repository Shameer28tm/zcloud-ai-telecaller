"""Boundary-aware text chunking and normalization for semantic voice retrieval.

Chunks are sized for low-latency retrieval and concise spoken agent context:
- Target chunk size: 350 to 500 characters.
- Overlap: ~50 characters to prevent loss of boundary context.
- Respects sentence and paragraph breaks.
- Keeps Question & Answer FAQ items atomic.
"""

from __future__ import annotations

import re
import uuid
from typing import Any, Optional

from zcloud_ai_telecaller.rag.types import (
    DocumentType,
    KnowledgeChunk,
    KnowledgeChunkMetadata,
)


def split_into_sentences(text: str) -> list[str]:
    """Splits text on paragraph, sentence, and punctuation boundaries."""
    # First split on explicit double line breaks
    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
    sentences: list[str] = []

    for para in paragraphs:
        # Split on sentence ending punctuation followed by space or newline
        raw_splits = re.split(r"(?<=[.?!;।])\s+", para)
        for s in raw_splits:
            s_clean = s.strip()
            if s_clean:
                sentences.append(s_clean)

    return sentences


def chunk_text(
    text: str,
    target_size: int = 400,
    overlap: int = 50,
) -> list[str]:
    """Recursively groups sentences into bounded chunks with overlap.

    Args:
        text: Normalized text to chunk.
        target_size: Desired max character length of a single chunk.
        overlap: Character overlap between consecutive chunks.

    Returns:
        List of chunk strings.
    """
    sentences = split_into_sentences(text)
    if not sentences:
        return [text] if text.strip() else []

    chunks: list[str] = []
    current_sentences: list[str] = []
    current_length = 0

    for s in sentences:
        s_len = len(s)
        # If adding sentence exceeds target size and we have at least 1 sentence
        if current_sentences and (current_length + s_len + 1 > target_size):
            chunk_str = " ".join(current_sentences)
            chunks.append(chunk_str)

            # Build overlap from the end of current_sentences
            overlap_sentences: list[str] = []
            overlap_len = 0
            for prev_s in reversed(current_sentences):
                if overlap_len + len(prev_s) <= overlap:
                    overlap_sentences.insert(0, prev_s)
                    overlap_len += len(prev_s)
                else:
                    break

            current_sentences = overlap_sentences + [s]
            current_length = sum(len(x) for x in current_sentences) + len(current_sentences) - 1
        else:
            current_sentences.append(s)
            current_length += s_len + (1 if len(current_sentences) > 1 else 0)

    if current_sentences:
        final_chunk = " ".join(current_sentences)
        # Avoid duplicate trailing chunk if identical
        if not chunks or chunks[-1] != final_chunk:
            chunks.append(final_chunk)

    return chunks


class KnowledgeChunker:
    """Builds KnowledgeChunk instances from raw text, PDF pages, DOCX sections, or FAQs."""

    @staticmethod
    def chunk_document(
        tenant_id: str,
        document_id: str,
        source_title: str,
        source_type: DocumentType,
        business_type: str = "clinic",
        full_text: str = "",
        pages_or_sections: Optional[list[dict[str, Any]]] = None,
        faqs: Optional[list[dict[str, str]]] = None,
        target_size: int = 400,
        overlap: int = 50,
    ) -> list[KnowledgeChunk]:
        """Produces a list of KnowledgeChunk models ready for indexing.

        For FAQs: Each Q/A is created as an atomic chunk with is_faq=True.
        For Documents: Sentences are grouped with boundaries and page metadata.
        """
        chunks: list[KnowledgeChunk] = []

        if source_type == DocumentType.FAQ and faqs:
            for idx, faq in enumerate(faqs):
                q = faq["question"]
                a = faq["answer"]
                cat = faq.get("category", business_type)
                content = f"Question: {q}\nAnswer: {a}"
                clean_content = f"{q}. {a}"
                chunk_id = f"chunk_{document_id}_{idx}"

                meta = KnowledgeChunkMetadata(
                    document_id=document_id,
                    tenant_id=tenant_id,
                    source_title=source_title,
                    business_type=cat,
                    source_type=DocumentType.FAQ,
                    chunk_index=idx,
                    total_chunks=len(faqs),
                    is_faq=True,
                    faq_question=q,
                )
                chunk = KnowledgeChunk(
                    id=chunk_id,
                    tenant_id=tenant_id,
                    document_id=document_id,
                    chunk_index=idx,
                    content=content,
                    clean_content=clean_content,
                    char_length=len(content),
                    token_count=max(1, len(clean_content.split())),
                    metadata=meta,
                )
                chunks.append(chunk)

            return chunks

        # Handle page-aware documents (e.g. PDF)
        if pages_or_sections and len(pages_or_sections) > 1 and "page_number" in pages_or_sections[0]:
            global_idx = 0
            for page in pages_or_sections:
                p_num = page.get("page_number", 1)
                p_text = page.get("text", "")
                p_chunks = chunk_text(p_text, target_size=target_size, overlap=overlap)
                for c_text in p_chunks:
                    chunk_id = f"chunk_{document_id}_{global_idx}"
                    meta = KnowledgeChunkMetadata(
                        document_id=document_id,
                        tenant_id=tenant_id,
                        source_title=source_title,
                        business_type=business_type,
                        source_type=source_type,
                        page=p_num,
                        chunk_index=global_idx,
                    )
                    chunks.append(
                        KnowledgeChunk(
                            id=chunk_id,
                            tenant_id=tenant_id,
                            document_id=document_id,
                            chunk_index=global_idx,
                            content=c_text,
                            clean_content=c_text,
                            char_length=len(c_text),
                            token_count=max(1, len(c_text.split())),
                            metadata=meta,
                        )
                    )
                    global_idx += 1

            # Update total_chunks count on all metadata
            for c in chunks:
                c.metadata.total_chunks = len(chunks)
            return chunks

        # Fallback to standard text chunking
        raw_chunks = chunk_text(full_text, target_size=target_size, overlap=overlap)
        for idx, c_text in enumerate(raw_chunks):
            chunk_id = f"chunk_{document_id}_{idx}"
            meta = KnowledgeChunkMetadata(
                document_id=document_id,
                tenant_id=tenant_id,
                source_title=source_title,
                business_type=business_type,
                source_type=source_type,
                chunk_index=idx,
                total_chunks=len(raw_chunks),
            )
            chunks.append(
                KnowledgeChunk(
                    id=chunk_id,
                    tenant_id=tenant_id,
                    document_id=document_id,
                    chunk_index=idx,
                    content=c_text,
                    clean_content=c_text,
                    char_length=len(c_text),
                    token_count=max(1, len(c_text.split())),
                    metadata=meta,
                )
            )

        return chunks
