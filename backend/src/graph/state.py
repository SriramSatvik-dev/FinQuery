from typing import TypedDict, Optional, Literal
from src.models import Chunk, RankedChunk, Citation

class RAGState(TypedDict):
    query: str
    query_type: Literal["specific_reference", "conceptual"]
    hyde_passage: str
    candidate_chunks: list[Chunk]
    reranked_chunks: list[RankedChunk]
    top_reranker_score: float
    abstain: bool
    answer: Optional[str]
    citations: list[Citation]
    grounding_passed: bool
    final_response: dict