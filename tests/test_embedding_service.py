from services.embedding_service import EmbeddingService


def test_embedding():
    service = EmbeddingService()

    text = "FortiGate provides network security capabilities."

    embedding = service.embed(text)

    print("\n" + "=" * 80)
    print("EMBEDDING TEST")
    print("=" * 80)

    print(f"Nombre de dimensions : {len(embedding)}")
    print(f"Premières valeurs    : {embedding[:5]}")
    print("=" * 80)

    assert embedding
    assert len(embedding) > 0
    assert all(isinstance(value, float) for value in embedding)