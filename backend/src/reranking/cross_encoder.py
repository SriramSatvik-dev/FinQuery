from sentence_transformers import CrossEncoder
from src.models import Chunk, RankedChunk

reranking_model = CrossEncoder(model_name_or_path="cross-encoder/ms-marco-MiniLM-L-6-v2")

def rerank(query_str: str, retrieved_chunks: list[Chunk]) -> tuple[list[RankedChunk], float]:
    input_pairs = []

    for chunk in retrieved_chunks:
        input_pairs.append((query_str, chunk.text))

    scores = reranking_model.predict(input_pairs)

    ranked_chunks = []

    for idx, chunk in enumerate(retrieved_chunks):
        ranked_chunks.append(RankedChunk(
            chunk=chunk,
            relevance_score=float(scores[idx])
        ))

    ranked_chunks.sort(key=lambda x: x.relevance_score, reverse=True)

    top_reranker_score = ranked_chunks[0].relevance_score if ranked_chunks else 0.0

    return ranked_chunks, top_reranker_score