import json
import time
from pathlib import Path

from services.document_service import DocumentService
from services.chunking_service import ChunkingService
from services.embedding_service import EmbeddingService
from services.rag_service import RAGService

from benchmark_questions import QUESTIONS


PDF_PATH = "data/test.pdf"
EVALUATIONS_PATH = Path("data/evaluations.json")

TOP_K = 5


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


def replace_experiment(evaluations, experiment):

    experiments = evaluations["experiments"]

    for index, existing in enumerate(experiments):

        if existing.get("id") == experiment["id"]:
            experiments[index] = experiment
            return

    experiments.append(experiment)


def test_benchmark_rag():

    # =========================================================
    # 1. EXTRACTION
    # =========================================================

    start = time.perf_counter()

    document_service = DocumentService()
    document = document_service.extract(PDF_PATH)

    extraction_time = time.perf_counter() - start

    # =========================================================
    # 2. CHUNKING
    # =========================================================

    start = time.perf_counter()

    chunking_service = ChunkingService()

    chunks = chunking_service.chunk_document(document)

    chunking_time = time.perf_counter() - start

    # =========================================================
    # 3. EMBEDDINGS / INDEX BUILD
    # =========================================================

    embedding_service = EmbeddingService()

    start = time.perf_counter()

    embeddings = embedding_service.embed_chunks(chunks)

    embedding_time = time.perf_counter() - start

    # =========================================================
    # 4. RAG SERVICE
    # =========================================================

    rag_service = RAGService(
        chunks=chunks,
        embeddings=embeddings,
        embedding_service=embedding_service,
    )

    # =========================================================
    # 5. QUESTIONS
    # =========================================================

    results = []

    for item in QUESTIONS:

        question_id = item["id"]
        question = item["question"]
        category = item["category"]

        print("\n")
        print("=" * 60)
        print(f"RAG — {question_id}")
        print("=" * 60)

        print(f"Question : {question}")

        # -----------------------------------------------------
        # Retrieval
        # -----------------------------------------------------

        start = time.perf_counter()

        retrieved = rag_service.retrieve(
            question,
            top_k=TOP_K,
        )

        retrieval_time = time.perf_counter() - start

        # -----------------------------------------------------
        # Generation
        # -----------------------------------------------------

        start = time.perf_counter()

        llm_result = rag_service.generate_answer(
            question,
            retrieved,
        )

        generation_time = time.perf_counter() - start

        # -----------------------------------------------------
        # Query total
        # -----------------------------------------------------

        query_time = retrieval_time + generation_time

        usage = llm_result["usage"]

        # -----------------------------------------------------
        # Output
        # -----------------------------------------------------

        print("\nSources :")

        for rank, source in enumerate(retrieved, start=1):

            print(
                f"#{rank} "
                f"page={source.page_number} "
                f"score={source.score:.4f}"
            )

        print("\nRéponse :")
        print(llm_result["content"])

        print("\nTiming :")
        print(f"Retrieval : {retrieval_time:.2f}s")
        print(f"Generation: {generation_time:.2f}s")
        print(f"Query     : {query_time:.2f}s")

        print("\nLLM :")
        print(f"Prompt tokens     : {usage['prompt_tokens']}")
        print(f"Completion tokens : {usage['completion_tokens']}")
        print(f"Total tokens      : {usage['total_tokens']}")
        print(f"Finish reason     : {llm_result['finish_reason']}")

        # -----------------------------------------------------
        # Save result
        # -----------------------------------------------------

        sources = []

        for rank, source in enumerate(retrieved, start=1):

            sources.append(
                {
                    "rank": rank,
                    "chunk_id": source.chunk_id,
                    "page": source.page_number,
                    "score": source.score,
                }
            )

        experiment = {
            "question_id": question_id,
            "question": question,
            "category": category,
            "method": "rag",
            "top_k": TOP_K,
            "document": document.filename,
            "timing": {
                "retrieval_seconds": round(retrieval_time, 2),
                "generation_seconds": round(generation_time, 2),
                "query_seconds": round(query_time, 2),
            },
            "llm": {
                "prompt_tokens": usage["prompt_tokens"],
                "completion_tokens": usage["completion_tokens"],
                "total_tokens": usage["total_tokens"],
                "reasoning_tokens": usage["reasoning_tokens"],
                "finish_reason": llm_result["finish_reason"],
            },
            "sources": sources,
            "answer": llm_result["content"],
        }

        results.append(experiment)

    # =========================================================
    # 6. SAVE
    # =========================================================

    evaluations = load_evaluations()

    replace_experiment(
        evaluations,
        {
            "id": "benchmark-rag-001",
            "type": "qa_benchmark",
            "method": "rag",
            "document": document.filename,
            "index": {
                "chunks": len(chunks),
                "embedding_dimension": len(embeddings[0]),
                "embedding_seconds": round(embedding_time, 2),
                "chunking_seconds": round(chunking_time, 2),
                "extraction_seconds": round(extraction_time, 2),
            },
            "questions": results,
        },
    )

    save_evaluations(evaluations)

    assert len(results) == len(QUESTIONS)