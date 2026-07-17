import json
import sys
import os
import time

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.graph.pipeline import run_pipeline

queries = [
    "Is KYC mandatory to open a bank account?",
    "How can a person be identified based on KYC?",
    "What are the update policies of KYC?",
    "Can I change the billing cycle of my credit card?",
    "My card issuer has been billing more than required. How can I lodge a complaint against them?",
    "Are banks required to report frauds to police?",
    "What is the udgam portal all about?",
    "What happens in case of a failed atm transaction?",
    "How should we keep our atm transaction secure?",
    "Is cts mandatory for cheques?",
    "Does the bank charge for cheque collection?",
    "Explain about pss act.",
    "What are the terms involved in pss act?",
    "What happens if an enterprise is unable to get registered on the URP portal due to lack of required documents?",
    "Are there any targets set by rbi to banks for lending to msmes?",
    "What kind of guidances are given by banks to mse entrepreneurs?",
    "Does the rbi contact center lodge complaints on behalf of complainants?",
    "Is there any alternate grievance redressal mechanism for customer complaints?",
    "How is e rupee different from upi?",
    "Can e rupee be accessed offline?",
    "What are the benefits of using e rupee?",
    "Are rupee 1, rupee 2 and rupee 5 notes still printed?",
    "On what factors does the amount of each value of banknotes printed depend on?",
    "Are coins with value less than 1 rupee still valid?",
    "What is a public deposit?",
    "What is the maximum time by which a mutual fund scheme should be launched after the grant of mutual fund registration?",
    "What are the different ways of classification of mutual fund schemes?",
    "What are the different types of risks associated with a product/scheme in which an investor is investing?",
    "What is the current repo rate?",
    "Who is the current RBI governor?",
    "What is the GST rate on insurance?",
    "How do I file income tax returns?",
    "What is today's Sensex value?",
    "What is the current home loan interest rate at SBI?",
    "Who is the current SEBI chairman?"
]

out_of_scope = [
    "What is the current repo rate?",
    "Who is the current RBI governor?",
    "What is the GST rate on insurance?",
    "How do I file income tax returns?",
    "What is today's Sensex value?",
    "What is the current home loan interest rate at SBI?",
    "Who is the current SEBI chairman?"
]

results = []

for i, query in enumerate(queries):
    print(f"\n[{i+1}/{len(queries)}] {query}")
    print("-" * 60)
    
    try:
        response = run_pipeline(query)
        
        if response["abstained"]:
            print(f"ABSTAINED — reason: {response['reason']}")
        else:
            print(f"ANSWERED")
            print(f"Answer: {response['answer'][:200]}")
            print(f"Score: {response['top_reranker_score']:.4f}")
            print(f"Citations: {len(response['citations'])}")
        
        results.append({
            "question": query,
            "expected_answer": "",
            "system_answer": response["answer"],
            "retrieved_contexts": [citation["excerpt"] for citation in response["citations"]],
            "should_abstain": query in out_of_scope,
            "abstained": response["abstained"],
        })
        
    except Exception as e:
        print(f"ERROR: {e}")
        results.append({
            "question": query,
            "response": None,
            "error": str(e)
        })

    time.sleep(10)

# save results
with open("golden_eval_set.json", "w") as f:
    json.dump(results, f, indent=2)

print("\n" + "=" * 60)
print(f"Total queries: {len(queries)}")
