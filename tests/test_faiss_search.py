from pathlib import Path

from services.document_service import DocumentService
from services.chunking_service import ChunkingService
from services.embedding_service import EmbeddingService
from services.faiss_service import FAISSService


PDF_PATH = Path("data/test.pdf")


def test_faiss_search():
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

    dimension = len(embeddings[0])

    faiss_service = FAISSService(
        dimension=dimension,
    )

    faiss_service.add(embeddings)

    query = (
        "Quelle est la différence entre un WAF et un firewall traditionnel ?"
    )

    query_embedding = embedding_service.embed(query)

    distances, indices = faiss_service.search(
        query_embedding,
        top_k=5,
    )

    print("\n" + "=" * 80)
    print("FAISS SEMANTIC SEARCH")
    print("=" * 80)

    print(f"Nombre de chunks      : {len(chunks)}")
    print(f"Nombre de vecteurs    : {faiss_service.size}")
    print(f"Dimension             : {dimension}")
    print(f"Question              : {query}")
    print("=" * 80)

    for rank, (distance, index) in enumerate(
        zip(distances, indices),
        start=1,
    ):
        chunk = chunks[index]

        print(f"\n--- Résultat #{rank} ---")
        print(f"FAISS index : {index}")
        print(f"Chunk ID    : {chunk.chunk_id}")
        print(f"Page        : {chunk.page_number}")
        print(f"Score       : {distance:.4f}")
        print(f"Texte       : {chunk.text[:500]}")

    print("=" * 80)

    assert faiss_service.size == len(chunks)
    assert len(indices) == 5
    assert len(distances) == 5