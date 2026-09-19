from pathlib import Path

from services.document_service import DocumentService
from services.chunking_service import ChunkingService
from services.embedding_service import EmbeddingService
from services.rag_service import RAGService


PDF_PATH = Path("data/test.pdf")


def test_rag_retrieval():
    document_service = DocumentService()
    document = document_service.extract(str(PDF_PATH))

    chunking_service = ChunkingService(
        chunk_size=1000,
        overlap=200,
    )

    chunks = chunking_service.chunk_document(document)

    embedding_service = EmbeddingService()

    embeddings = embedding_service.embed_chunks(
        chunks,
        batch_size=32,
    )

    rag_service = RAGService(
        chunks=chunks,
        embeddings=embeddings,
    )

    question = "Qu'est-ce qu'un firewall ?"

    results = rag_service.retrieve(
        question,
        top_k=5,
    )

    print("\n" + "=" * 80)
    print("RAG RETRIEVAL")
    print("=" * 80)

    print(f"Question : {question}")
    print(f"Résultats : {len(results)}")

    for rank, result in enumerate(
        results,
        start=1,
    ):
        print(f"\n--- Résultat #{rank} ---")
        print(f"Chunk ID : {result.chunk_id}")
        print(f"Page     : {result.page_number}")
        print(f"Score    : {result.score:.4f}")
        print(f"Texte    : {result.text[:500]}")

    print("=" * 80)

    assert len(results) == 5

    assert all(
        result.chunk_id
        for result in results
    )

    assert all(
        result.text
        for result in results
    )

    assert all(
        result.page_number > 0
        for result in results
    )