import time

from services.document_service import DocumentService
from services.qa_service import QAService


PDF_PATH = "data/test.pdf"

QUESTION = "Qu'est-ce qu'un firewall ?"


def test_qa_service():

    start = time.perf_counter()

    document_service = DocumentService()
    document = document_service.extract(PDF_PATH)

    extraction_time = time.perf_counter() - start

    qa_service = QAService()

    start = time.perf_counter()

    result = qa_service.answer_with_metrics(
        document=document,
        question=QUESTION,
    )

    generation_time = time.perf_counter() - start

    total_time = extraction_time + generation_time

    print("\n")
    print("=" * 60)
    print("LLM-ONLY QA")
    print("=" * 60)

    print(f"Question : {QUESTION}")

    print("\nRéponse :")
    print(result["content"])

    print("\nTiming :")
    print(f"Extraction : {extraction_time:.2f}s")
    print(f"Génération : {generation_time:.2f}s")
    print(f"Total      : {total_time:.2f}s")

    print("\nLLM :")

    usage = result["usage"]

    print(f"Prompt tokens     : {usage['prompt_tokens']}")
    print(f"Completion tokens : {usage['completion_tokens']}")
    print(f"Total tokens      : {usage['total_tokens']}")
    print(f"Reasoning tokens  : {usage['reasoning_tokens']}")
    print(f"Finish reason     : {result['finish_reason']}")

    assert result["content"]