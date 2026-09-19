from pathlib import Path

from services.document_service import DocumentService


PDF_PATH = Path("data/test.pdf")


def test_extract_pdf():
    service = DocumentService()

    document = service.extract(str(PDF_PATH))

    assert document.filename == str(PDF_PATH)
    assert document.page_count == 10


def test_pages_have_text():
    service = DocumentService()

    document = service.extract(str(PDF_PATH))

    assert len(document.pages) == 10

    for page in document.pages:
        assert page.page_number >= 1
        assert page.text.strip() != ""