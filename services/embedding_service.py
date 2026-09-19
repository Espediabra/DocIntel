import requests

from config import LM_STUDIO_BASE_URL, EMBEDDING_MODEL


class EmbeddingService:
    def __init__(self):
        self.url = f"{LM_STUDIO_BASE_URL}/embeddings"
        self.model = EMBEDDING_MODEL

    def embed(self, text: str) -> list[float]:
        payload = {
            "model": self.model,
            "input": text,
        }

        response = requests.post(
            self.url,
            json=payload,
        )

        response.raise_for_status()

        data = response.json()

        return data["data"][0]["embedding"]

    def embed_chunks(self, chunks) -> list[list[float]]:
        embeddings = []

        for chunk in chunks:
            embedding = self.embed(chunk.text)
            embeddings.append(embedding)

        return embeddings