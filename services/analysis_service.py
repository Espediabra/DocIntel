from models.schemas import Document
from services.llm_client import LLMClient


class AnalysisService:
    def __init__(self, llm_client=None):
        self.llm_client = llm_client or LLMClient()

    def analyze(self, document: Document) -> str:
        document_text = "\n\n".join(
            f"[Page {page.page_number}]\n{page.text}"
            for page in document.pages
        )

        messages = [
            {
                "role": "system",
                "content": (
                    "You are a document analysis assistant. "
                    "Analyze only the information contained in the provided document. "
                    "Do not invent facts that are not supported by the document. "
                    "Answer in French."
                ),
            },
            {
                "role": "user",
                "content": f"""
Analyse le document fourni et produis une analyse structurée selon les six sections suivantes :

1. Executive Summary
2. Key Points
3. Important Facts
4. Risks / Limitations
5. Recommendations
6. Open Questions

Pour chaque section, base-toi uniquement sur le contenu du document.
Si une information n'est pas disponible dans le document, indique-le clairement.

DOCUMENT :

{document_text}
""",
            },
        ]

        return self.llm_client.generate(messages)