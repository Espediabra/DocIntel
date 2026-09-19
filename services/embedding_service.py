import requests

from config import LM_STUDIO_BASE_URL, EMBEDDING_MODEL


class EmbeddingService:
    def __init__(self):
        self.url = f"{LM_STUDIO_BASE_URL}/embeddings"
        self.model = EMBEDDING_MODEL

    def embed(self, text: str) -> list[float]:
        embeddings = self.embed_batch([text])
        return embeddings[0]

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []

        payload = {
            "model": self.model,
            "input": texts,
        }

        response = requests.post(
            self.url,
            json=payload,
        )

        response.raise_for_status()

        data = response.json()

        return [
            item["embedding"]
            for item in data["data"]
        ]

    def embed_chunks(
        self,
        chunks,
        batch_size: int = 32,
    ) -> list[list[float]]:
        embeddings = []

        for start in range(0, len(chunks), batch_size):
            batch = chunks[start:start + batch_size]

            texts = [
                chunk.text
                for chunk in batch
            ]

            batch_embeddings = self.embed_batch(texts)

            embeddings.extend(batch_embeddings)

        return embeddings