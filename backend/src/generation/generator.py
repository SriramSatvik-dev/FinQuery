from src.models import RankedChunk, Citation
from src.config import llm
import re

system_prompt = """You are a financial regulation assistant. Your job is to answer questions 
about Indian banking and financial regulations in plain, simple English 
that any common person can understand.

Rules you must follow:
- Answer ONLY using the provided context chunks
- For every claim you make, cite the chunk ID inline like this: [chunk_id]
- If the context does not contain enough information, say "I cannot find 
  sufficient information in the available documents to answer this question"
- Never make up information not present in the context
- Explain regulatory language in simple terms
- Be concise and direct
- If you find relevant information in the context, answer directly and confidently
- Do not say you cannot find information if you can actually find it in the chunks
- If partial information is available, give what you have directly without expressing uncertainty.
- Answer only the question asked by the user. Do not include related or supplementary information unless the user explicitly requests it. Ignore retrieved passages that are not necessary to answer the user's question."""

pattern = r'\[chunk_id:\s*([a-f0-9]{16})\]'

def generate(query_str: str, context_chunks: list[RankedChunk], limit: int = 5) -> tuple[str, list[Citation]]:
    context = ""
    for chunk in context_chunks:
        context += f"[chunk_id: {chunk.chunk.chunk_id}]\n{chunk.chunk.text}\n\n"

    user_prompt = f"""
Context:
{context}

Question: {query_str}

Answer: """
    
    response = llm.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ],
        temperature= 0.0
    )

    answer = response.choices[0].message.content

    cited_ids = re.findall(pattern, answer)

    chunk_map = {rc.chunk.chunk_id: rc.chunk for rc in context_chunks}

    citations = []
    seen_ids = set()

    for cited_id in cited_ids:
        if cited_id in chunk_map and cited_id not in seen_ids:
            chunk = chunk_map[cited_id]

            citations.append(Citation(
                chunk_id= chunk.chunk_id,
                source_file= chunk.source_file,
                page_num= chunk.page_num,
                excerpt= chunk.text
            ))
            seen_ids.add(cited_id)

    return answer, citations