import faiss
import numpy as np


class FAISSService:
    def __init__(self, dimension: int):
        self.dimension = dimension
        self.index = faiss.IndexFlatIP(dimension)

    def add(self, embeddings: list[list[float]]) -> None:
        vectors = np.array(
            embeddings,
            dtype="float32",
        )

        # Normaliser chaque vecteur pour obtenir une norme L2 égale à 1.
        # Ainsi, le produit scalaire entre deux vecteurs normalisés
        # est exactement égal à leur similarité cosinus.
        faiss.normalize_L2(vectors)

        self.index.add(vectors)

    def search(
        self,
        query_embedding: list[float],
        top_k: int = 5,
    ):
        query_vector = np.array(
            [query_embedding],
            dtype="float32",
        )

        faiss.normalize_L2(query_vector)

        distances, indices = self.index.search(
            query_vector,
            top_k,
        )

        return distances[0], indices[0]

    @property
    def size(self) -> int:
        return self.index.ntotal