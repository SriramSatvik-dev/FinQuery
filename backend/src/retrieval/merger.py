from src.models import Chunk

def merge(bm25_chunks: list[Chunk], dense_chunks: list[Chunk], max_chunks: int = 20) -> list[Chunk]:
    seen = set()
    merged = []

    for chunk in dense_chunks + bm25_chunks:
        if chunk.chunk_id not in seen:
            seen.add(chunk.chunk_id)
            merged.append(chunk)

    return merged[:max_chunks]