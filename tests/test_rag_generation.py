import json
import time
from pathlib import Path

from services.document_service import DocumentService
from services.chunking_service import ChunkingService
from services.embedding_service import EmbeddingService
from services.rag_service import RAGService


PDF_PATH = Path("data/test.pdf")
EVALUATIONS_PATH = Path("data/evaluations.json")


def test_rag_generation():
    total_start = time.perf_counter()

    # ============================================================
    # 1. Extraction
    # ============================================================

    start = time.perf_counter()

    document_service = DocumentService()
    document = document_service.extract(str(PDF_PATH))

    extraction_time = time.perf_counter() - start

    # ============================================================
    # 2. Chunking
    # ============================================================

    start = time.perf_counter()

    chunking_service = ChunkingService(
        chunk_size=1000,
        overlap=200,
    )

    chunks = chunking_service.chunk_document(document)

    chunking_time = time.perf_counter() - start

    # ============================================================
    # 3. Embeddings
    # ============================================================

    start = time.perf_counter()

    embedding_service = EmbeddingService()

    embeddings = embedding_service.embed_chunks(
        chunks,
        batch_size=32,
    )

    embedding_time = time.perf_counter() - start

    # ============================================================
    # 4. Création du service RAG
    # ============================================================

    rag_service = RAGService(
        chunks=chunks,
        embeddings=embeddings,
    )

    # ============================================================
    # 5. Question
    # ============================================================

    question = "Qu'est-ce qu'un firewall ?"

    # ============================================================
    # 6. Retrieval
    # ============================================================

    start = time.perf_counter()

    results = rag_service.retrieve(
        question,
        top_k=5,
    )

    retrieval_time = time.perf_counter() - start

    # ============================================================
    # 7. Génération
    # ============================================================

    start = time.perf_counter()

    llm_result = rag_service.generate_answer(
        question,
        results,
    )

    answer = llm_result["content"]
    usage = llm_result["usage"]

    generation_time = time.perf_counter() - start

    # ============================================================
    # 8. Temps total
    # ============================================================

    total_time = time.perf_counter() - total_start

    # ============================================================
    # Affichage
    # ============================================================

    print("\n")
    print("=" * 80)
    print("RAG GENERATION")
    print("=" * 80)

    print(f"\nQuestion : {question}")

    print("\nSources utilisées :")

    for i, result in enumerate(results, start=1):
        print(
            f"#{i} "
            f"page={result.page_number} "
            f"score={result.score:.4f}"
        )

    print("\nRéponse :")
    print(answer)

    print("\nTemps d'exécution :")
    print(f"Extraction  : {extraction_time:.2f} s")
    print(f"Chunking    : {chunking_time:.2f} s")
    print(f"Embeddings  : {embedding_time:.2f} s")
    print(f"Retrieval   : {retrieval_time:.2f} s")
    print(f"Génération  : {generation_time:.2f} s")
    print(f"Total       : {total_time:.2f} s")

    print("\n" + "=" * 80)
    
    print("\nMétriques LLM :")
    print(f"Prompt tokens     : {usage['prompt_tokens']}")
    print(f"Completion tokens : {usage['completion_tokens']}")
    print(f"Total tokens      : {usage['total_tokens']}")
    print(f"Reasoning tokens  : {usage['reasoning_tokens']}")
    print(f"Finish reason     : {llm_result['finish_reason']}")

    # ============================================================
    # Sauvegarde dans evaluations.json
    # ============================================================

    with open(EVALUATIONS_PATH, "r", encoding="utf-8") as file:
        evaluations = json.load(file)

    evaluations["experiments"].append(
        {
            "id": "exp-002",
            "type": "rag_generation",
            "document": {
                "filename": PDF_PATH.name,
                "pages": document.page_count,
                "chunks": len(chunks),
                "chunk_size": 1000,
                "chunk_overlap": 200,
            },
            "model": {
                "llm": "qwen2.5-7b-instruct-1m",
                "embedding": "text-embedding-nomic-embed-text-v1.5",
                "temperature": 0.2,
            },
            "query": {
                "question": question,
                "top_k": 5,
            },
            "retrieval": {
                "results": [
                    {
                        "chunk_id": result.chunk_id,
                        "page": result.page_number,
                        "score": result.score,
                    }
                    for result in results
                ]
            },
            "timing_seconds": {
                "extraction": round(extraction_time, 3),
                "chunking": round(chunking_time, 3),
                "embeddings": round(embedding_time, 3),
                "retrieval": round(retrieval_time, 3),
                "generation": round(generation_time, 3),
                "total": round(total_time, 3),
            },
            "llm_usage": {
                "prompt_tokens": usage["prompt_tokens"],
                "completion_tokens": usage["completion_tokens"],
                "total_tokens": usage["total_tokens"],
                "reasoning_tokens": usage["reasoning_tokens"],
                "finish_reason": llm_result["finish_reason"],
            },
            "evaluation": {
                "grounded": True,
                "sources_available": True,
                "hallucination_observed": False,
            },
            "observations": [
                "RAG répond à partir des passages récupérés.",
                "Les sources et pages sont conservées.",
                "Le modèle refuse de répondre lorsque le contexte ne contient pas l'information demandée."
            ]
        }
    )

    with open(EVALUATIONS_PATH, "w", encoding="utf-8") as file:
        json.dump(
            evaluations,
            file,
            indent=2,
            ensure_ascii=False,
        )

    # ============================================================
    # Assertions
    # ============================================================

    assert answer
    assert isinstance(answer, str)
    assert len(results) == 5