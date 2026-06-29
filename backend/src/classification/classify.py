from src.config import llm

def classify_query(query_str: str) -> str:
    system_prompt = """You are a query classification agent. Classify the user query into one of two categories:

- If the query asks about something specific (a rule, a number, a deadline, a specific regulation): output exactly: specific_reference
- If the query asks to explain a concept or how something works: output exactly: conceptual

Output only one word. No explanation. No punctuation."""

    user_prompt = f"""User query: {query_str}

Classification: """
    
    response = llm.chat.completions.create(
        model= 'llama-3.1-8b-instant',
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}, 
        ],
        temperature= 0.0
    )

    result = response.choices[0].message.content.strip().lower()

    if "conceptual" in result: return "conceptual"
    return "specific_reference"

