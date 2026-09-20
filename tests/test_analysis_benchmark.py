import json
import time
from pathlib import Path

from services.analysis_service import AnalysisService
from services.document_service import DocumentService


PDF_PATH = "data/test.pdf"
EVALUATIONS_PATH = Path("data/evaluations.json")


def test_analysis_benchmark():

    # ---------------------------------------------------------
    # 1. Extraction
    # ---------------------------------------------------------

    start = time.perf_counter()

    document_service = DocumentService()
    document = document_service.extract(PDF_PATH)

    extraction_time = time.perf_counter() - start

    # ---------------------------------------------------------
    # 2. Analyse LLM-only
    # ---------------------------------------------------------

    analysis_service = AnalysisService()

    start = time.perf_counter()

    result = analysis_service.analyze_with_metrics(document)

    generation_time = time.perf_counter() - start

    total_time = extraction_time + generation_time

    # ---------------------------------------------------------
    # 3. Affichage
    # ---------------------------------------------------------

    print("\n")
    print("=" * 60)
    print("DOCUMENT ANALYSIS — LLM ONLY")
    print("=" * 60)

    print(f"Document : {document.filename}")
    print(f"Pages    : {document.page_count}")

    print("\nAnalyse :")
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

    # ---------------------------------------------------------
    # 4. Sauvegarde de l'expérience
    # ---------------------------------------------------------

    if EVALUATIONS_PATH.exists():
        with open(EVALUATIONS_PATH, "r", encoding="utf-8") as file:
            evaluations = json.load(file)
    else:
        evaluations = {"experiments": []}

    experiment = {
        "id": "exp-003",
        "type": "document_analysis",
        "method": "llm_only",
        "document": document.filename,
        "pages": document.page_count,
        "timing": {
            "extraction_seconds": round(extraction_time, 2),
            "generation_seconds": round(generation_time, 2),
            "total_seconds": round(total_time, 2),
        },
        "llm": {
            "prompt_tokens": usage["prompt_tokens"],
            "completion_tokens": usage["completion_tokens"],
            "total_tokens": usage["total_tokens"],
            "reasoning_tokens": usage["reasoning_tokens"],
            "finish_reason": result["finish_reason"],
        },
        "answer": result["content"],
    }

    evaluations["experiments"].append(experiment)

    with open(EVALUATIONS_PATH, "w", encoding="utf-8") as file:
        json.dump(
            evaluations,
            file,
            indent=2,
            ensure_ascii=False,
        )

    assert result["content"]