from src.config import llm

def hyde_gen(query_str: str) -> str :
    system_prompt = """You are a financial advisor who takes an input query from the user and generates an answer
    based on your knowledge. Your answer will then be used to perform semantic search from a vector database.
    
    Strictly follow the following instructions:
    1. Write a single dense paragraph.
    2. Don't reference the question directly.
    3. Write as if you are the document that contains the answer.
    4. Accuracy doesn't matter - semantic coverage does
    """

    user_prompt = f""" User query: {query_str}

Answer: """
    
    response = llm.chat.completions.create(
        model= "llama-3.1-8b-instant",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ],
        temperature= 0.5
    )

    return response.choices[0].message.content