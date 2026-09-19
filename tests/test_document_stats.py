from pathlib import Path

from services.document_service import DocumentService


PDF_PATH = Path("data/test.pdf")


def test_document_stats():
    service = DocumentService()
    document = service.extract(str(PDF_PATH))

    total_characters = sum(len(page.text) for page in document.pages)
    total_words = sum(len(page.text.split()) for page in document.pages)
    estimated_tokens = total_characters // 4

    print(f"\nNombre de pages       : {document.page_count}")
    print(f"Nombre de caractères : {total_characters}")
    print(f"Nombre de mots       : {total_words}")
    print(f"Tokens estimés       : {estimated_tokens}")

    assert document.page_count > 0
    assert total_characters > 0
    assert total_words > 0