import json
import time
from pathlib import Path

from services.document_service import DocumentService
from services.qa_service import QAService

from benchmark_questions import QUESTIONS


PDF_PATH = "data/test.pdf"
EVALUATIONS_PATH = Path("data/evaluations.json")


def load_evaluations():

    if EVALUATIONS_PATH.exists():
        with open(EVALUATIONS_PATH, "r", encoding="utf-8") as file:
            return json.load(file)

    return {"experiments": []}


def save_evaluations(evaluations):

    with open(EVALUATIONS_PATH, "w", encoding="utf-8") as file:
        json.dump(
            evaluations,
            file,
            indent=2,
            ensure_ascii=False,
        )


def test_benchmark_llm_only():

    # ---------------------------------------------------------
    # Document
    # ---------------------------------------------------------

    start = time.perf_counter()

    document_service = DocumentService()
    document = document_service.extract(PDF_PATH)

    extraction_time = time.perf_counter() - start

    # ---------------------------------------------------------
    # QA service
    # ---------------------------------------------------------

    qa_service = QAService()

    evaluations = load_evaluations()

    results = []

    # ---------------------------------------------------------
    # Questions
    # ---------------------------------------------------------

    for item in QUESTIONS:

        question_id = item["id"]
        question = item["question"]
        category = item["category"]

        print("\n")
        print("=" * 60)
        print(f"LLM-ONLY — {question_id}")
        print("=" * 60)

        print(f"Question : {question}")

        start = time.perf_counter()

        result = qa_service.answer_with_metrics(
            document=document,
            question=question,
        )

        generation_time = time.perf_counter() - start

        total_time = extraction_time + generation_time

        usage = result["usage"]

        print("\nRéponse :")
        print(result["content"])

        print("\nTiming :")
        print(f"Génération : {generation_time:.2f}s")
        print(f"Total      : {total_time:.2f}s")

        print("\nLLM :")
        print(f"Prompt tokens     : {usage['prompt_tokens']}")
        print(f"Completion tokens : {usage['completion_tokens']}")
        print(f"Total tokens      : {usage['total_tokens']}")
        print(f"Finish reason     : {result['finish_reason']}")

        experiment = {
            "question_id": question_id,
            "question": question,
            "category": category,
            "method": "llm_only",
            "document": document.filename,
            "timing": {
                "extraction_seconds": round(extraction_time, 2),
                "generation_seconds": round(generation_time, 2),
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

        results.append(experiment)

    # ---------------------------------------------------------
    # Save
    # ---------------------------------------------------------

    def replace_experiment(evaluations, experiment):
        experiments = evaluations["experiments"]

        for index, existing in enumerate(experiments):

            if existing.get("id") == experiment["id"]:
                experiments[index] = experiment
                return

        experiments.append(experiment)
        
    replace_experiment(
        evaluations,
        {
            "id": "benchmark-llm-only-001",
            "type": "qa_benchmark",
            "method": "llm_only",
            "questions": results,
        },
    )

    save_evaluations(evaluations)

    assert len(results) == len(QUESTIONS)