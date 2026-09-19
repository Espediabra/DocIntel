from pathlib import Path

from services.chunking_service import ChunkingService
from services.document_service import DocumentService


PDF_PATH = Path("data/test.pdf")


def test_document_chunking():
    document_service = DocumentService()
    document = document_service.extract(str(PDF_PATH))

    chunking_service = ChunkingService(
        chunk_size=1000,
        overlap=200,
    )

    chunks = chunking_service.chunk_document(document)

    print("\n" + "=" * 80)
    print("CHUNKING")
    print("=" * 80)

    print(f"Nombre de pages : {document.page_count}")
    print(f"Nombre de chunks : {len(chunks)}")

    print("\nPremiers chunks :")

    for chunk in chunks[:5]:
        print("\n" + "-" * 80)
        print(f"ID      : {chunk.chunk_id}")
        print(f"Page    : {chunk.page_number}")
        print(f"Taille  : {len(chunk.text)} caractères")
        print(f"Texte   : {chunk.text[:200]}...")

    print("=" * 80)

    assert chunks
    assert all(chunk.text.strip() for chunk in chunks)
    assert all(chunk.page_number > 0 for chunk in chunks)
    assert all(chunk.chunk_id for chunk in chunks)

    page_numbers = {page.page_number for page in document.pages}

    assert all(
        chunk.page_number in page_numbers
        for chunk in chunks
    )
    
    # Comptage des chunks par page pour vérifier la distribution des chunks
    
    from collections import Counter

    chunks_per_page = Counter(
        chunk.page_number
        for chunk in chunks
    )

    print("\nChunks par page :")

    for page_number, count in sorted(chunks_per_page.items()):
        print(f"Page {page_number:2d} : {count} chunks")
        
        
    #Statistiques de chunks pour le lissage des ratios de chunks 
    
    chunk_sizes = [len(chunk.text) for chunk in chunks]

    print("\nStatistiques des chunks :")
    print(f"Taille minimale : {min(chunk_sizes)}")
    print(f"Taille maximale : {max(chunk_sizes)}")
    print(f"Taille moyenne  : {sum(chunk_sizes) / len(chunk_sizes):.1f}")