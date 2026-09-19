from models.schemas import Chunk, Document


class ChunkingService:
    def __init__(self, chunk_size: int = 1000, overlap: int = 200):
        if overlap >= chunk_size:
            raise ValueError("overlap must be smaller than chunk_size")

        self.chunk_size = chunk_size
        self.overlap = overlap

    def chunk_document(self, document: Document) -> list[Chunk]:
        chunks = []

        for page in document.pages:
            page_chunks = self._chunk_text(
                text=page.text,
                page_number=page.page_number,
            )

            chunks.extend(page_chunks)

        return chunks

    def _chunk_text(self, text: str, page_number: int) -> list[Chunk]:
        chunks = []

        start = 0
        chunk_index = 0

        step = self.chunk_size - self.overlap

        while start < len(text):
            end = start + self.chunk_size
            chunk_text = text[start:end].strip()

            if chunk_text:
                chunk_id = f"page-{page_number}-chunk-{chunk_index:03d}"

                chunks.append(
                    Chunk(
                        chunk_id=chunk_id,
                        page_number=page_number,
                        text=chunk_text,
                    )
                )

                chunk_index += 1

            start += step

        return chunks