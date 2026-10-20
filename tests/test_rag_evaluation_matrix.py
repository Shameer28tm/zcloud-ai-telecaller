"""Comprehensive retrieval evaluation matrix for Z Cloud Voice RAG.

Evaluates real retrieval accuracy and semantic relevance across:
1. English direct clinic questions.
2. Tamil clinic questions.
3. Tanglish (Tamil-English mixed) questions.
4. Synonyms and paraphrased phrasing.
5. Out-of-domain / unanswerable questions (evidence threshold rejection).

Uses the active default local embedding provider (BAAI/bge-small-en-v1.5).
"""

from __future__ import annotations

import pytest
from zcloud_ai_telecaller.rag.embeddings.hybrid_provider import HybridEmbeddingProvider
from zcloud_ai_telecaller.rag.embeddings.local_provider import LocalOnnxEmbeddingProvider
from zcloud_ai_telecaller.rag.retriever import TenantKnowledgeRetriever
from zcloud_ai_telecaller.rag.types import KnowledgeChunk, KnowledgeChunkMetadata
from zcloud_ai_telecaller.rag.vector_store import PersistentTenantVectorStore


CLINIC_CORPUS = [
    {
        "doc_id": "doc_hours",
        "title": "Clinic Hours & Location",
        "text": (
            "Apex Dental Clinic is located at 124 Anna Salai, Chennai, near Thousand Lights. "
            "Our clinic is open Monday through Friday from 8:00 AM to 6:00 PM, "
            "and Saturday from 9:00 AM to 1:00 PM. We are closed on Sundays and public holidays."
        ),
    },
    {
        "doc_id": "doc_insurance",
        "title": "Insurance & Payment Methods",
        "text": (
            "Apex Dental accepts MetLife, Delta Dental Premier, Aetna PPO, and Star Health Insurance. "
            "Patients can pay via UPI (Google Pay, PhonePe), credit cards, or cash at the reception desk. "
            "Consultation copayment is 500 INR due before treatment."
        ),
    },
    {
        "doc_id": "doc_rootcanal",
        "title": "Root Canal Procedure Guide",
        "text": (
            "Single sitting root canal treatment takes approximately 60 to 90 minutes under local anesthesia. "
            "Mild tooth soreness is expected for 24 to 48 hours following the procedure. "
            "Painkillers and antibiotics are prescribed if necessary."
        ),
    },
    {
        "doc_id": "doc_ortho",
        "title": "Orthodontic Aligners & Braces",
        "text": (
            "We provide clear ceramic braces and invisible Invisalign aligners for teeth alignment. "
            "Treatment duration typically spans 12 to 18 months depending on severity. "
            "Adjustment appointments are scheduled every 4 to 6 weeks."
        ),
    },
]


@pytest.fixture(scope="module")
def populated_clinic_retriever(tmp_path_factory):
    """Initializes and indexes the realistic clinic corpus with local ONNX embeddings."""
    tmp_dir = tmp_path_factory.mktemp("rag_eval_matrix")
    store = PersistentTenantVectorStore(storage_dir=str(tmp_dir))
    local_p = LocalOnnxEmbeddingProvider(model_name="BAAI/bge-small-en-v1.5")
    hybrid = HybridEmbeddingProvider(local_provider=local_p)

    tenant_id = "eval-clinic-tenant"
    for item in CLINIC_CORPUS:
        chunks = [
            KnowledgeChunk(
                id=f"{item['doc_id']}_c0",
                tenant_id=tenant_id,
                document_id=item["doc_id"],
                chunk_index=0,
                content=item["text"],
                clean_content=item["text"],
                char_length=len(item["text"]),
                metadata=KnowledgeChunkMetadata(
                    document_id=item["doc_id"],
                    tenant_id=tenant_id,
                    source_title=item["title"],
                ),
            )
        ]
        vectors = hybrid.embed_documents([item["text"]])
        store.upsert_chunks(tenant_id, item["doc_id"], chunks, vectors, "BAAI/bge-small-en-v1.5")

    retriever = TenantKnowledgeRetriever(
        vector_store=store,
        embedding_provider=hybrid,
        min_score=0.50,
        top_k=2,
    )
    return retriever, tenant_id


def test_eval_english_hours(populated_clinic_retriever):
    """Test 1: English question on weekday operating hours."""
    retriever, tenant_id = populated_clinic_retriever
    resp = retriever.search(tenant_id, "What are your weekday operating hours?")
    assert resp.has_sufficient_evidence is True
    assert resp.results[0].document_id == "doc_hours"
    assert resp.top_score >= 0.50
    print(f"\n[EVAL 1: English Hours] Score: {resp.top_score:.3f} | Top Doc: {resp.results[0].source_title}")


def test_eval_english_insurance(populated_clinic_retriever):
    """Test 2: English question on accepted dental insurance."""
    retriever, tenant_id = populated_clinic_retriever
    resp = retriever.search(tenant_id, "Do you accept Star Health or MetLife insurance?")
    assert resp.has_sufficient_evidence is True
    assert resp.results[0].document_id == "doc_insurance"
    assert resp.top_score >= 0.65
    print(f"\n[EVAL 2: English Insurance] Score: {resp.top_score:.3f} | Top Doc: {resp.results[0].source_title}")


def test_eval_tamil_location(populated_clinic_retriever):
    """Test 3: Pure Tamil inquiry on clinic location."""
    retriever, tenant_id = populated_clinic_retriever
    # "Where is your clinic located in Chennai?"
    resp = retriever.search(tenant_id, "உங்கள் பல் மருத்துவமனை சென்னையில் எங்கே உள்ளது?")
    assert resp.has_sufficient_evidence is True
    assert resp.results[0].document_id == "doc_hours"
    print(f"\n[EVAL 3: Pure Tamil Location] Score: {resp.top_score:.3f} | Top Doc: {resp.results[0].source_title}")


def test_eval_tanglish_saturday_hours(populated_clinic_retriever):
    """Test 4: Tanglish (Tamil-English) inquiry on Saturday timing."""
    retriever, tenant_id = populated_clinic_retriever
    resp = retriever.search(tenant_id, "Saturday clinic open irukka, timing enna?")
    assert resp.has_sufficient_evidence is True
    assert resp.results[0].document_id == "doc_hours"
    print(f"\n[EVAL 4: Tanglish Saturday Hours] Score: {resp.top_score:.3f} | Top Doc: {resp.results[0].source_title}")


def test_eval_tanglish_insurance_acceptance(populated_clinic_retriever):
    """Test 5: Tanglish inquiry on insurance acceptance."""
    retriever, tenant_id = populated_clinic_retriever
    resp = retriever.search(tenant_id, "Aetna insurance edupaangala, payment epdi?")
    assert resp.has_sufficient_evidence is True
    assert resp.results[0].document_id == "doc_insurance"
    print(f"\n[EVAL 5: Tanglish Insurance] Score: {resp.top_score:.3f} | Top Doc: {resp.results[0].source_title}")


def test_eval_paraphrase_nerve_treatment(populated_clinic_retriever):
    """Test 6: Paraphrased / Synonym inquiry for root canal ('nerve treatment')."""
    retriever, tenant_id = populated_clinic_retriever
    resp = retriever.search(tenant_id, "How long does nerve filling treatment in tooth take?")
    assert resp.has_sufficient_evidence is True
    assert resp.results[0].document_id == "doc_rootcanal"
    print(f"\n[EVAL 6: Synonym Root Canal] Score: {resp.top_score:.3f} | Top Doc: {resp.results[0].source_title}")


def test_eval_paraphrase_upi_payment(populated_clinic_retriever):
    """Test 7: Paraphrased inquiry for UPI payment methods (GPay/PhonePe)."""
    retriever, tenant_id = populated_clinic_retriever
    resp = retriever.search(tenant_id, "Can I scan QR code with GPay at reception?")
    assert resp.has_sufficient_evidence is True
    assert resp.results[0].document_id == "doc_insurance"
    print(f"\n[EVAL 7: Synonym UPI Payment] Score: {resp.top_score:.3f} | Top Doc: {resp.results[0].source_title}")


def test_eval_out_of_domain_stock_market(populated_clinic_retriever):
    """Test 8: Completely out-of-domain inquiry (stock market)."""
    retriever, tenant_id = populated_clinic_retriever
    resp = retriever.search(tenant_id, "What is the stock price of Apple on NASDAQ today?")
    # Must reject with has_sufficient_evidence False (score < min_score 0.50)
    assert resp.has_sufficient_evidence is False
    print(f"\n[EVAL 8: Out-of-Domain Stock Market] Evidence: {resp.has_sufficient_evidence} | Score: {resp.top_score:.3f}")


def test_eval_out_of_domain_weather(populated_clinic_retriever):
    """Test 9: Completely out-of-domain inquiry (weather forecast)."""
    retriever, tenant_id = populated_clinic_retriever
    resp = retriever.search(tenant_id, "Will it rain tomorrow in San Francisco California?")
    assert resp.has_sufficient_evidence is False
    print(f"\n[EVAL 9: Out-of-Domain Weather] Evidence: {resp.has_sufficient_evidence} | Score: {resp.top_score:.3f}")


def test_eval_ortho_duration(populated_clinic_retriever):
    """Test 10: Orthodontic aligners treatment duration."""
    retriever, tenant_id = populated_clinic_retriever
    resp = retriever.search(tenant_id, "How many months will Invisalign teeth alignment take?")
    assert resp.has_sufficient_evidence is True
    assert resp.results[0].document_id == "doc_ortho"
    print(f"\n[EVAL 10: Ortho Duration] Score: {resp.top_score:.3f} | Top Doc: {resp.results[0].source_title}")
