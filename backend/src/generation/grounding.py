from src.models import Citation, RankedChunk
from src.config import llm

system_prompt = """You are a fact checker. Given an answer and source chunks, 
        verify every claim in the answer is supported by the chunks.
        Return only "PASS" or "FAIL"."""

def check_grounding(answer: str, citations: list[Citation], reranked_chunks: list[RankedChunk]) -> bool :
    retrieved_ids = [chunk.chunk.chunk_id for chunk in reranked_chunks]

    for citation in citations:
        if citation.chunk_id not in retrieved_ids:
            return False
        
    context = ""

    for chunk in reranked_chunks[:3]:
        context += f"text: {chunk.chunk.text}\n\n"

    user_prompt = f"""Answer: {answer}

Source chunks: {context}

Is every claim in answer supported by source chunks?"""
    
    response = llm.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ],
        temperature= 0.0
    )

    result = response.choices[0].message.content.strip().upper()
    return result.startswith("PASS")