import time
from pathlib import Path

from services.chunking_service import ChunkingService
from services.document_service import DocumentService
from services.embedding_service import EmbeddingService


PDF_PATH = Path("data/test.pdf")


def test_embed_document_chunks():
    document_service = DocumentService()
    document = document_service.extract(str(PDF_PATH))

    chunking_service = ChunkingService(
        chunk_size=1000,
        overlap=200,
    )

    chunks = chunking_service.chunk_document(document)

    embedding_service = EmbeddingService()

    start_time = time.perf_counter()

    embeddings = embedding_service.embed_chunks(chunks)

    elapsed_time = time.perf_counter() - start_time

    print("\n" + "=" * 80)
    print("DOCUMENT EMBEDDINGS")
    print("=" * 80)

    print(f"Nombre de chunks       : {len(chunks)}")
    print(f"Nombre d'embeddings    : {len(embeddings)}")

    if embeddings:
        print(f"Dimension des vecteurs : {len(embeddings[0])}")

    print(f"Temps total            : {elapsed_time:.2f} secondes")

    if chunks:
        print(
            f"Temps moyen / chunk    : "
            f"{elapsed_time / len(chunks):.3f} secondes"
        )

    print("=" * 80)

    assert len(embeddings) == len(chunks)
    assert all(embedding for embedding in embeddings)

    assert all(
        len(embedding) == len(embeddings[0])
        for embedding in embeddings
    )