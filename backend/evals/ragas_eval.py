import json
import sys
import os
from dotenv import load_dotenv

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

load_dotenv()

from ragas import evaluate, EvaluationDataset
from ragas.metrics import Faithfulness, AnswerRelevancy
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper
from langchain_openai import ChatOpenAI
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.rate_limiters import InMemoryRateLimiter

ratelimiter = InMemoryRateLimiter(
    requests_per_second= 0.5,
    check_every_n_seconds= 0.1,
    max_bucket_size= 1,
)

# load golden eval set
with open("golden_eval_set.json", "r") as f:
    golden_set = json.load(f)

# separate entries
answered = [e for e in golden_set if not e["abstained"] and e["system_answer"] and e["retrieved_contexts"]]
should_abstain = [e for e in golden_set if e["should_abstain"]]
correctly_abstained = [e for e in golden_set if e["abstained"] and e["should_abstain"]]

print(f"Total entries: {len(golden_set)}")
print(f"Answered entries for RAGAS eval: {len(answered)}")
print(f"Should abstain: {len(should_abstain)}")
print(f"Correctly abstained: {len(correctly_abstained)}")
print()

abstention_accuracy = len(correctly_abstained) / len(should_abstain) if should_abstain else 0.0
print(f"Abstention Accuracy: {abstention_accuracy:.4f} ({len(correctly_abstained)}/{len(should_abstain)})")
print()

# build EvaluationDataset
dataset_entries = []
for entry in answered:
    dataset_entries.append({
        "user_input": entry["question"],
        "response": entry["system_answer"],
        "retrieved_contexts": entry["retrieved_contexts"],
        "reference": entry["expected_answer"]
    })

eval_dataset = EvaluationDataset.from_list(dataset_entries)

# Groq via OpenAI-compatible endpoint — no langchain-groq needed
evaluator_llm = LangchainLLMWrapper(
    ChatOpenAI(
        model="llama-3.1-8b-instant",
        api_key=os.getenv("GROQ_API_KEY"),
        base_url="https://api.groq.com/openai/v1",
        temperature=0,
        rate_limiter=ratelimiter
    )
)

# local embeddings — no API cost
evaluator_embeddings = LangchainEmbeddingsWrapper(
    HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-mpnet-base-v2"
    )
)

# define metrics
metrics = [
    Faithfulness(llm=evaluator_llm),
    AnswerRelevancy(llm=evaluator_llm, embeddings=evaluator_embeddings),
]

print("Running RAGAS evaluation...")
print("This will make multiple LLM calls — may take a few minutes...")
print()

results = evaluate(
    dataset=eval_dataset,
    metrics=metrics,
    llm=evaluator_llm
)

# print results
df = results.to_pandas()

print()
print("=" * 60)
print("RAGAS EVALUATION RESULTS")
print("=" * 60)
print(f"Faithfulness:        {df['faithfulness'].mean():.4f}")
print(f"Answer Relevancy:    {df['answer_relevancy'].mean():.4f}")
print(f"Abstention Accuracy: {abstention_accuracy:.4f}")
print("=" * 60)

# save results
output = {
    "summary": {
        "faithfulness": float(df["faithfulness"].mean()),
        "answer_relevancy": float(df["answer_relevancy"].mean()),
        "abstention_accuracy": abstention_accuracy,
        "total_entries": len(golden_set),
        "answered_entries": len(answered),
        "abstained_correctly": len(correctly_abstained),
        "total_should_abstain": len(should_abstain)
    },
    "per_question": []
}

with open("ragas_results.json", "w") as f:
    json.dump(output, f, indent=2)

print()
print("Detailed results saved to ragas_results.json")