# FinQuery — Plain-Language Financial Regulation Assistant

FinQuery is a closed-corpus question answering system that helps common people understand Indian banking and financial regulations. It answers questions grounded exclusively in official RBI and SEBI documents, cites the exact source for every claim, and explicitly refuses to answer when the corpus does not contain sufficient information — because a wrong answer about financial rules is worse than no answer.

---

## The Problem

Millions of Indians interact with banks, investments, and financial institutions daily but remain unaware of their rights and entitlements. RBI and SEBI publish all the relevant guidelines publicly, but the documents are dense, scattered, and written for regulators — not people. FinQuery bridges that gap.

---

## Architecture

The system is built around a **hybrid retrieval pipeline** orchestrated through a **stateful LangGraph graph**.

```
User Query
    ↓
[classify_query] — specific_reference or conceptual
                          ↓                  ↓
                     [retrieve]   <-   [hyde_generate]
            (dense retrieve + bm25 retrieve)
                          ↓
                      [rerank] — cross-encoder scores each (query, chunk) pair
                          ↓
                  [confidence_gate]
                          ├── score < threshold → [abstain] → structured refusal
                          └── score ≥ threshold → [generate_answer]
                                                         ↓
                                                  [grounding_check]
                                                         ↓
                                                  [format_response]
```

### Query Classification

Every query is classified as either `specific_reference` (asking about a concrete rule, number, or deadline) or `conceptual` (asking how something works or why). Conceptual queries go through **HyDE (Hypothetical Document Embeddings)** — the LLM generates a hypothetical answer passage first, which is then embedded for dense retrieval. This works because the embedding of a plausible answer is geometrically closer to relevant document chunks than the embedding of a question.

### Hybrid Retrieval

Two retrievers run in parallel and their results are union-merged after deduplication:

- **BM25** (via `rank-bm25`) catches exact keyword matches — critical for regulatory terminology like specific circular numbers, defined legal terms, and scheme names that semantic search misses.
- **Dense retrieval** (via ChromaDB + `all-mpnet-base-v2`) catches semantic matches — queries paraphrased differently from how the regulation is worded.

Neither retriever alone is sufficient for regulatory text. BM25 misses paraphrased queries; dense retrieval misses exact terminology.

### Cross-Encoder Reranking

The merged candidate chunks are reranked by `ms-marco-MiniLM-L-6-v2`, a cross-encoder that reads each `(query, chunk)` pair jointly. Unlike bi-encoders that score query and chunk independently, the cross-encoder attends to both simultaneously — far more accurate but too slow for full-corpus retrieval, which is why it operates only on the top 20 candidates.

### Confidence Gating and Abstention

The top reranker score acts as a confidence signal. If it falls below a calibrated threshold, the system returns a structured refusal rather than generating an answer. This is the most important product decision in the system — in a financial regulation context, a confidently wrong answer causes real harm.

### Answer Generation and Grounding Check

The top reranked chunks are passed to `llama-3.1-8b-instant` via Groq API with a prompt that instructs it to answer only from the provided context, explain in plain English, and cite chunk IDs inline. A second LLM pass (grounding check) verifies that every claim in the answer is supported by the retrieved chunks before the response is returned.

---

## Tech Stack

| Layer | Tool |
|---|---|
| Orchestration | LangGraph |
| Embeddings | sentence-transformers/all-mpnet-base-v2 |
| Vector Store | ChromaDB (local persistent) |
| Sparse Retrieval | BM25 via rank-bm25 |
| Reranking | cross-encoder/ms-marco-MiniLM-L-6-v2 |
| LLM | Groq API — llama-3.1-8b-instant |
| Document Parsing | PyMuPDF |
| API | FastAPI |
| Evaluation | RAGAS |

---

## Corpus

24 official PDF documents sourced from **rbi.org.in** and **sebi.gov.in**, totalling **2,588 chunks** after ingestion. Topics covered include KYC, credit and debit cards, ATM transactions, cheque clearing, digital lending, MSME lending, grievance redressal, digital rupee, Indian currency, NBFC regulations, payment systems, mutual funds, and financial literacy.

The corpus is fixed at ingestion time — FinQuery is a closed-corpus system. Queries outside the corpus scope are refused rather than answered from model knowledge.

---

## Project Structure

```
backend/
├── src/
│   ├── models.py                  # RawPage, Chunk, RankedChunk, Citation
│   ├── config.py                  # settings, confidence threshold
│   ├── ingestion/
│   │   ├── loader.py              # PDF → RawPage list
│   │   ├── chunker.py             # RawPage → Chunk list
│   │   └── embedder.py            # Chunk → ChromaDB + BM25 index
│   ├── classification/
│   │   ├── classify.py            # query type classification
│   │   └── hyde_query.py          # hypothetical passage generation
│   ├── retrieval/
│   │   ├── bm25_retriever.py      # BM25 keyword retrieval
│   │   ├── dense_retriever.py     # semantic vector retrieval
│   │   └── merger.py              # union merge + deduplication
│   ├── reranking/
│   │   └── cross_encoder.py       # cross-encoder reranking
│   ├── generation/
│   │   ├── generator.py           # answer generation with citations
│   │   └── grounding.py           # grounding verification
│   └── graph/
│       ├── state.py               # RAGState TypedDict
│       ├── nodes.py               # LangGraph node functions
│       ├── edges.py               # conditional routing functions
│       └── pipeline.py            # graph assembly and compilation
├── evals/
│   ├── golden_eval_set.json       # 33-query manually curated evaluation set
│   ├── ragas_eval.py              # RAGAS evaluation script
│   └── ragas_results.json         # evaluation results
├── data/
│   ├── raw/                       # source PDFs (gitignored)
│   ├── chroma_db/                 # ChromaDB persistent store (gitignored)
│   └── processed/                 # BM25 serialized index (gitignored)
├── main.py                        # FastAPI app with /ingest, /query, /delete endpoints
└── requirements.txt
```

---

## API Endpoints

### `POST /ingest`
Upload a PDF to the corpus. Chunks the document, embeds it into ChromaDB, and rebuilds the BM25 index. Rejects duplicate uploads by filename.

**Request:** `multipart/form-data` with a PDF file

**Response:**
```json
{
  "message": "Document ingested successfully",
  "filename": "KYC09062025.pdf",
  "pages": 10,
  "chunks": 87
}
```

### `POST /askquery`
Ask a question in plain English.

**Request:**
```json
{
  "query": "What happens if my bank fails to resolve my ATM complaint?"
}
```

**Response (answered):**
```json
{
  "abstained": false,
  "answer": "If your bank fails to re-credit your account within T+5 calendar days after a failed ATM transaction, it must pay you compensation of ₹100 per day for the delay [chunk_id: 4f9a87bfdeb385ca].",
  "citations": [
    {
      "chunk_id": "4f9a87bfdeb385ca",
      "source_file": "FAQATM04072025.pdf",
      "page_num": 3,
      "excerpt": "..."
    }
  ],
  "query_type": "specific_reference",
  "top_reranker_score": 7.72
}
```

**Response (abstained):**
```json
{
  "abstained": true,
  "reason": "insufficient_corpus_coverage",
  "message": "The available documents do not contain sufficient information to answer this query confidently.",
  "answer": null,
  "citations": []
}
```

### `DELETE /delete/{filename}`
Remove a document from the corpus by filename. Deletes from ChromaDB, rebuilds the BM25 index, and removes the raw PDF from disk.

---

## Evaluation

Evaluated on a manually curated golden set of **40 queries** — 33 answerable queries covering factual and conceptual questions across the corpus, and 7 out-of-scope queries that the system should refuse.

| Metric | Score |
|---|---|
| RAGAS Faithfulness | 0.82 |
| RAGAS Answer Relevancy | 0.77 |
| Abstention Accuracy | 1.00 |

Faithfulness measures whether every claim in the answer is supported by the retrieved chunks. Answer relevancy measures whether the answer addresses the question asked. Abstention accuracy measures whether the system correctly refused all 7 out-of-scope queries — a perfect score.

---

## Known Limitations

**Conversational queries perform poorly.** The cross-encoder (`ms-marco-MiniLM`) was trained on short keyword queries from the MS MARCO dataset. Long narrative queries like "I've been saving money for 5 years and want to start investing but I'm scared..." score poorly against short regulatory chunks even when semantically relevant. The fix is a query preprocessing step that extracts the core question before retrieval — documented as a future improvement.

**Chunking is not FAQ-aware.** The current chunker uses character-based splitting with overlap. RBI FAQ documents follow varying question-answer formats (`Q.1/Ans`, `Query/Response`, bold questions with plain answers) — no single pattern covers all 24 documents. A format-aware chunker would improve retrieval precision but was descoped given the format diversity.

**Confidence threshold is manually calibrated.** The threshold was set based on observed score distributions on the golden evaluation set. A production system would calibrate this automatically using a larger labelled dataset.

---

For a detailed record of key engineering decisions and tradeoffs made 
during development, see [ENGINEERING_DECISIONS.md](./ENGINEERING_DECISIONS.md).

---

## Setup

```bash
# clone and navigate
git clone https://github.com/SriramSatvik-dev/FinQuery
cd finquery/backend

# create and activate virtual environment
python -m venv venv
venv\Scripts\activate       # Windows
source venv/bin/activate    # Mac/Linux

# install dependencies
pip install -r requirements.txt

# set up environment variables
cp .env.example .env
# add your GROQ_API_KEY to .env

# create required data directories
mkdir -p data/raw data/chroma_db data/processed

# start the server
uvicorn main:app --reload
```

Visit `http://localhost:8000/docs` to access the interactive API.

---

## Environment Variables

```
GROQ_API_KEY=your_groq_api_key_here
CONFIDENCE_THRESHOLD=2.0
```

Get a free Groq API key at [console.groq.com](https://console.groq.com).
