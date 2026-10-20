"""Safe, resource-bounded text extractors for PDF, DOCX, TXT, and Manual FAQs.

Security rules:
1. File signature (magic bytes) validation.
2. Max file size enforcement (10 MB limit).
3. Active content / macro execution prevention (rejects or strips VBA).
4. Safe handling of empty or scanned/OCR-only PDFs with explicit user-friendly errors.
5. Deterministic text normalization.
"""

from __future__ import annotations

import hashlib
import io
import json
import logging
import re
import xml.etree.ElementTree as ET
import zipfile
from typing import Any, Optional, Tuple

import pypdf

from zcloud_ai_telecaller.rag.types import DocumentType

logger = logging.getLogger("zcloud_ai_telecaller.rag.extractors")

MAX_FILE_SIZE_BYTES = 10 * 1024 * 1024  # 10 MB limit


class DocumentExtractionError(Exception):
    """Raised when document extraction fails or validation fails."""
    def __init__(self, message: str, code: str = "EXTRACTION_FAILED"):
        super().__init__(message)
        self.code = code
        self.message = message


def compute_sha256(data: bytes) -> str:
    """Computes standard hex SHA-256 hash of binary content."""
    return hashlib.sha256(data).hexdigest()


def normalize_extracted_text(text: str) -> str:
    """Normalizes whitespace, trims runs of blank lines, and strips control characters."""
    if not text:
        return ""
    # Strip null bytes and non-printable control characters except \n and \t
    cleaned = re.sub(r"[\x00-\x08\x0B\x0C\x0E-\x1F\x7F]", "", text)
    # Normalize unicode whitespace
    cleaned = re.sub(r"[ \t\r\f\v]+", " ", cleaned)
    # Collapse multiple consecutive newlines into double newlines
    cleaned = re.sub(r"\n\s*\n\s*\n+", "\n\n", cleaned)
    return cleaned.strip()


def validate_file_signature(data: bytes, expected_type: DocumentType) -> None:
    """Verifies file headers (magic bytes) to prevent extension spoofing."""
    if len(data) == 0:
        raise DocumentExtractionError("File is empty (0 bytes).", code="EMPTY_FILE")
    if len(data) > MAX_FILE_SIZE_BYTES:
        raise DocumentExtractionError(
            f"File size ({len(data)} bytes) exceeds the maximum allowed limit of {MAX_FILE_SIZE_BYTES} bytes (10MB).",
            code="FILE_TOO_LARGE",
        )

    if expected_type == DocumentType.PDF:
        if not data.startswith(b"%PDF-"):
            raise DocumentExtractionError(
                "Invalid PDF file signature. The uploaded file does not begin with '%PDF-'.",
                code="INVALID_FILE_SIGNATURE",
            )
    elif expected_type == DocumentType.DOCX:
        if not data.startswith(b"PK\x03\x04"):
            raise DocumentExtractionError(
                "Invalid DOCX file signature. The uploaded file is not a valid ZIP/OpenXML package.",
                code="INVALID_FILE_SIGNATURE",
            )
    elif expected_type in (DocumentType.TXT, DocumentType.MARKDOWN, DocumentType.FAQ):
        # Ensure it's decodable as text and not binary executable
        sample = data[:1024]
        if b"\x00" in sample:
            raise DocumentExtractionError(
                "Binary data detected in text document. Please upload plain text or valid UTF-8.",
                code="BINARY_DATA_REJECTED",
            )


class SafeTextExtractor:
    """Extracts text content and structural metadata from supported file types."""

    @staticmethod
    def extract_from_pdf(data: bytes) -> Tuple[str, list[dict[str, Any]]]:
        """Safely extracts text page-by-page from a PDF byte array.

        Returns:
            Tuple of (full_normalized_text, page_records)
        """
        validate_file_signature(data, DocumentType.PDF)
        pages_metadata: list[dict[str, Any]] = []
        full_text_parts: list[str] = []

        try:
            reader = pypdf.PdfReader(io.BytesIO(data))
            num_pages = len(reader.pages)
            if num_pages == 0:
                raise DocumentExtractionError("PDF contains zero pages.", code="EMPTY_PDF")

            total_chars = 0
            for idx, page in enumerate(reader.pages):
                page_text = page.extract_text() or ""
                cleaned = normalize_extracted_text(page_text)
                total_chars += len(cleaned)
                if cleaned:
                    full_text_parts.append(cleaned)
                    pages_metadata.append({
                        "page_number": idx + 1,
                        "char_count": len(cleaned),
                        "text": cleaned,
                    })

            # Check for scanned or image-only PDF
            if total_chars < 20:
                raise DocumentExtractionError(
                    "Unsupported scanned or image-only document. No extractable text was found in the PDF. "
                    "Please provide a text-based document or OCR-processed PDF.",
                    code="UNSUPPORTED_SCANNED_DOCUMENT",
                )

            combined_text = "\n\n".join(full_text_parts)
            return combined_text, pages_metadata

        except DocumentExtractionError:
            raise
        except Exception as exc:
            logger.error("Failed to parse PDF: %s", exc)
            raise DocumentExtractionError(
                f"Failed to process PDF document: {str(exc)}",
                code="PDF_PARSING_FAILED",
            ) from exc

    @staticmethod
    def extract_from_docx(data: bytes) -> Tuple[str, list[dict[str, Any]]]:
        """Safely extracts paragraphs from a DOCX file without executing macros.

        Inspects the ZIP archive, rejects any VBA macro code (`word/vbaProject.bin`),
        and parses `word/document.xml`.
        """
        validate_file_signature(data, DocumentType.DOCX)
        try:
            with zipfile.ZipFile(io.BytesIO(data), "r") as z:
                # Security check: detect and reject active macro content
                namelist = z.namelist()
                for name in namelist:
                    if "vba" in name.lower() or name.endswith(".bin"):
                        raise DocumentExtractionError(
                            "Embedded active macros detected in DOCX file. For security reasons, macro-enabled documents are rejected.",
                            code="MACRO_CONTENT_REJECTED",
                        )

                if "word/document.xml" not in namelist:
                    raise DocumentExtractionError(
                        "Malformed DOCX: Missing 'word/document.xml'.",
                        code="INVALID_DOCX_STRUCTURE",
                    )

                xml_content = z.read("word/document.xml")
                root = ET.fromstring(xml_content)

                # OpenXML namespace
                namespaces = {"w": "http://schemas.openxmlformats.org/wordprocessingml/2006/main"}

                paragraphs = []
                for p in root.iterfind(".//w:p", namespaces):
                    texts = [t.text for t in p.iterfind(".//w:t", namespaces) if t.text]
                    if texts:
                        p_str = "".join(texts).strip()
                        if p_str:
                            paragraphs.append(p_str)

                combined_text = normalize_extracted_text("\n\n".join(paragraphs))
                if not combined_text:
                    raise DocumentExtractionError(
                        "No readable text found in DOCX document.",
                        code="EMPTY_DOCX",
                    )

                sections = [{"paragraph_index": i + 1, "text": p} for i, p in enumerate(paragraphs)]
                return combined_text, sections

        except DocumentExtractionError:
            raise
        except zipfile.BadZipFile as exc:
            raise DocumentExtractionError(
                "Invalid or corrupted DOCX archive.",
                code="CORRUPT_DOCX",
            ) from exc
        except Exception as exc:
            logger.error("Failed to parse DOCX: %s", exc)
            raise DocumentExtractionError(
                f"Failed to process DOCX document: {str(exc)}",
                code="DOCX_PARSING_FAILED",
            ) from exc

    @staticmethod
    def extract_from_text(data: bytes) -> Tuple[str, list[dict[str, Any]]]:
        """Extracts text from plain text or Markdown with encoding fallbacks."""
        validate_file_signature(data, DocumentType.TXT)
        text = None
        for enc in ("utf-8", "utf-8-sig", "latin-1", "cp1252"):
            try:
                text = data.decode(enc)
                break
            except UnicodeDecodeError:
                continue

        if text is None:
            raise DocumentExtractionError(
                "Could not decode text document using standard UTF-8 or Latin-1 encodings.",
                code="ENCODING_DECODE_FAILED",
            )

        cleaned = normalize_extracted_text(text)
        if not cleaned:
            raise DocumentExtractionError("Text document is empty.", code="EMPTY_TEXT")

        return cleaned, [{"text": cleaned}]

    @staticmethod
    def extract_from_faq(data_or_text: Any) -> list[dict[str, str]]:
        """Parses manual FAQs from JSON or structured Q/A text.

        Supported formats:
        1. JSON: [{"question": "...", "answer": "...", "category": "..."}]
        2. Structured text:
           Q: What are your hours?
           A: Mon-Fri 8am-6pm.
        """
        faq_items = []
        if isinstance(data_or_text, bytes):
            text_str = data_or_text.decode("utf-8", errors="replace").strip()
        else:
            text_str = str(data_or_text).strip()

        # Try JSON parsing
        if text_str.startswith("[") or text_str.startswith("{"):
            try:
                parsed = json.loads(text_str)
                if isinstance(parsed, dict) and "faqs" in parsed:
                    parsed = parsed["faqs"]
                if isinstance(parsed, list):
                    for item in parsed:
                        q = item.get("question") or item.get("q") or ""
                        a = item.get("answer") or item.get("a") or ""
                        c = item.get("category") or item.get("business_type") or "clinic"
                        if q.strip() and a.strip():
                            faq_items.append({
                                "question": q.strip(),
                                "answer": a.strip(),
                                "category": c.strip(),
                            })
                    if faq_items:
                        return faq_items
            except Exception:
                pass

        # Parse Q/A text lines
        q_pattern = re.compile(r"^(?:q(?:uestion)?|வி(?:னா)?)\s*[:.-]\s*(.+)$", re.IGNORECASE)
        a_pattern = re.compile(r"^(?:a(?:nswer)?|ப(?:தில்)?)\s*[:.-]\s*(.+)$", re.IGNORECASE)

        lines = text_str.split("\n")
        current_q = None
        current_a_lines = []

        for line in lines:
            trimmed = line.strip()
            if not trimmed:
                continue

            q_match = q_pattern.match(trimmed)
            if q_match:
                if current_q and current_a_lines:
                    faq_items.append({
                        "question": current_q,
                        "answer": " ".join(current_a_lines).strip(),
                        "category": "clinic",
                    })
                    current_a_lines = []
                current_q = q_match.group(1).strip()
                continue

            a_match = a_pattern.match(trimmed)
            if a_match and current_q:
                current_a_lines.append(a_match.group(1).strip())
                continue

            if current_q and current_a_lines:
                current_a_lines.append(trimmed)

        if current_q and current_a_lines:
            faq_items.append({
                "question": current_q,
                "answer": " ".join(current_a_lines).strip(),
                "category": "clinic",
            })

        if not faq_items:
            raise DocumentExtractionError(
                "Could not parse valid Question/Answer pairs from FAQ content. "
                "Please format as Q: [question] and A: [answer], or as a JSON list of {question, answer}.",
                code="INVALID_FAQ_FORMAT",
            )

        return faq_items
