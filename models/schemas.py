from dataclasses import dataclass


@dataclass
class Page:
    page_number: int
    text: str


@dataclass
class Document:
    filename: str
    pages: list[Page]

    @property
    def page_count(self) -> int:
        return len(self.pages)


@dataclass
class Chunk:
    chunk_id: str
    page_number: int
    text: str