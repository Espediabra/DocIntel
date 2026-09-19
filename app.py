from services.document_service import DocumentService


document_service = DocumentService()

document = document_service.extract("data/test.pdf")

print(f"Document : {document.filename}")
print(f"Nombre de pages : {document.page_count}")

for page in document.pages:
    print(f"\n--- Page {page.page_number} ---")
    print(page.text[:500])