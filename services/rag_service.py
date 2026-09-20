from models.schemas import Chunk, SearchResult
from services.embedding_service import EmbeddingService
from services.faiss_service import FAISSService
from services.llm_client import LLMClient


class RAGService:

    def __init__(
        self,
        chunks,
        embeddings,
        embedding_service=None,
        faiss_service=None,
        llm_client=None,
    ):
        self.chunks = chunks

        self.embedding_service = (
            embedding_service or EmbeddingService()
        )

        dimension = len(embeddings[0])

        self.faiss_service = (
            faiss_service or FAISSService(dimension)
        )

        self.faiss_service.add(embeddings)

        self.llm_client = llm_client or LLMClient()

    def retrieve(self, question, top_k=5):

        query_embedding = self.embedding_service.embed(question)

        distances, indices = self.faiss_service.search(
            query_embedding,
            top_k=top_k,
        )

        results = []

        for distance, index in zip(distances, indices):

            chunk = self.chunks[index]

            results.append(
                SearchResult(
                    chunk_id=chunk.chunk_id,
                    page_number=chunk.page_number,
                    text=chunk.text,
                    score=float(distance),
                )
            )

        return results

    def generate_answer(self, question, results):

        context = "\n\n".join(
            f"[Page {result.page_number}]\n{result.text}"
            for result in results
        )

        messages = [
            {
                "role": "system",
                "content": (
                    "You are a document question-answering assistant. "
                    "Answer only from the provided context. "
                    "Do not invent information. "
                    "If the context does not contain enough information "
                    "to answer the question, say so clearly. "
                    "Answer in French."
                ),
            },
            {
                "role": "user",
                "content": f"""
Réponds à la question en utilisant uniquement le contexte fourni.

QUESTION :

{question}

CONTEXTE :

{context}

Consignes :
- Réponds en français.
- Utilise uniquement les informations du contexte.
- Ne complète pas avec des connaissances externes.
- Si le contexte ne permet pas de répondre, indique-le clairement.
""",
            },
        ]

        return self.llm_client.generate(messages)