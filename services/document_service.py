import pymupdf

from models.schemas import Document, Page


class DocumentService:

    def extract(self, pdf_path: str) -> Document:
        pdf = pymupdf.open(pdf_path)

        pages = []

        for page_number, page in enumerate(pdf, start=1):
            text = page.get_text()

            pages.append(
                Page(
                    page_number=page_number,
                    text=text,
                )
            )

        pdf.close()

        return Document(
            filename=pdf_path,
            pages=pages,
        )