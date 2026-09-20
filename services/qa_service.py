from models.schemas import Document
from services.llm_client import LLMClient


class QAService:
    def __init__(self, llm_client=None):
        self.llm_client = llm_client or LLMClient()

    def _build_messages(self, document: Document, question: str):
        document_text = "\n\n".join(
            f"[Page {page.page_number}]\n{page.text}"
            for page in document.pages
        )

        return [
            {
                "role": "system",
                "content": (
                    "You are a document question-answering assistant. "
                    "Answer only from the provided document. "
                    "Do not use external knowledge. "
                    "Do not invent information. "
                    "If the document does not contain enough information "
                    "to answer the question, say so clearly. "
                    "Answer in French."
                ),
            },
            {
                "role": "user",
                "content": f"""
Réponds à la question en utilisant uniquement le document fourni.

QUESTION :

{question}

DOCUMENT :

{document_text}

Consignes :
- Réponds en français.
- Utilise uniquement les informations présentes dans le document.
- Ne complète pas avec des connaissances externes.
- Si le document ne permet pas de répondre, indique-le clairement.
""",
            },
        ]

    def answer(self, document: Document, question: str) -> str:
        result = self.answer_with_metrics(document, question)

        return result["content"]

    def answer_with_metrics(self, document: Document, question: str):
        messages = self._build_messages(document, question)

        return self.llm_client.generate(messages)