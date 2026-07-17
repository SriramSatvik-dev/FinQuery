# Engineering Decisions — FinQuery

This document captures the key technical decisions made during the development of FinQuery, the reasoning behind each one, and the alternatives that were considered and rejected. These decisions collectively shaped the system's architecture and reliability.

---

## 1. Hybrid Retrieval Over Dense-Only

**Decision:** Use BM25 and dense retrieval in parallel, union-merge the results, and pass the merged set to the reranker.

**Why:** Early testing showed that dense retrieval alone missed queries containing exact regulatory terminology. A query like "What is MCLR?" retrieved semantically related chunks about interest rates but missed the chunk that literally defined MCLR — because the embedding of the question and the embedding of a definition don't always land close together in vector space. BM25 catches exact keyword matches that dense retrieval misses. Conversely, BM25 fails completely on paraphrased queries where the user's words don't appear in the document. Neither retriever alone is sufficient for regulatory text, which is simultaneously dense with defined terminology and written in language common people don't use.

**Alternative rejected:** Dense-only retrieval with a larger top-K. Increasing K without BM25 retrieves more chunks but doesn't fix the fundamental precision gap on exact terminology — it just gives the reranker more noise to sort through.

---

## 2. Cross-Encoder Reranking as a Second Pass

**Decision:** Use `ms-marco-MiniLM-L-6-v2` to rerank the merged candidate set rather than relying on cosine similarity scores from the vector store.

**Why:** Cosine similarity scores each candidate independently — the query embedding and chunk embedding are compared without either seeing the other. A cross-encoder reads the query and chunk jointly, attending to both simultaneously, which makes it far more accurate at judging relevance. The tradeoff is speed — a cross-encoder cannot scale to full corpus retrieval. The two-stage approach — fast approximate retrieval followed by precise reranking on a small candidate set — is the standard industry solution to this tradeoff.

**Alternative rejected:** Using the vector store's similarity scores directly for ranking. Testing showed this produced worse ordering than the cross-encoder on ambiguous queries where multiple chunks were plausibly relevant.

---

## 3. Confidence Gating and Structured Abstention

**Decision:** Use the top reranker score as a confidence signal. If it falls below a calibrated threshold, return a structured refusal rather than generating an answer.

**Why:** This is a product decision as much as a technical one. In a financial regulation context, a system that confidently answers incorrectly is actively harmful — a user might make a financial decision based on wrong information. Most RAG tutorials generate an answer regardless of retrieval quality. The deliberate choice to abstain when confidence is low is the defining characteristic of FinQuery and the decision that most interviewers find interesting.

The threshold is a tunable parameter rather than a hardcoded value. Higher values produce fewer false positives but more abstentions. Lower values answer more queries but risk hallucination. The current value was calibrated against the golden evaluation set.

**Alternative rejected:** Generating an answer with a disclaimer ("I'm not sure but..."). This is worse than abstaining — it gives the user a potentially wrong answer while hedging responsibility. A structured refusal is cleaner and more honest.

---

## 4. HyDE for Conceptual Queries

**Decision:** For queries classified as conceptual, generate a hypothetical answer passage using the LLM before performing dense retrieval, and embed the passage instead of the original query.

**Why:** The embedding of a question and the embedding of a relevant answer don't always land close together in vector space. "How does the term deposit renewal process work?" and a chunk that explains renewal terms have low cosine similarity because one is a question and the other is a declarative paragraph. By generating a hypothetical passage that sounds like a document answer, the embedding is geometrically much closer to real document chunks. BM25 still uses the original query — HyDE only helps dense retrieval, not keyword matching.

**Alternative rejected:** Using the original query for dense retrieval on all query types. Testing showed that conceptual queries retrieved weaker candidates when the raw question was embedded, and HyDE improved both retrieval quality and final answer quality on these queries.

---

## 5. Query Classification Before Retrieval

**Decision:** Add an LLM-based classification step at the start of the pipeline that routes queries into `specific_reference` or `conceptual` paths.

**Why:** Different query types need different retrieval strategies. A query asking for a specific rule ("What is the penalty for late ATM credit?") benefits from direct retrieval where exact keyword matches matter. A query asking for an explanation ("How does the KYC process work?") benefits from HyDE-enhanced retrieval where semantic coverage matters more than exact terms. Treating all queries identically forces a compromise that serves neither type well.

The classifier uses `temperature=0.0` for deterministic output and includes an explicit fallback — if the LLM returns an unexpected value, the system defaults to `specific_reference`. This prevents a classification failure from breaking the pipeline.

**Alternative rejected:** Routing all queries through HyDE. This adds an unnecessary LLM call for queries that don't need it and can actually hurt retrieval on specific-reference queries where the hypothetical passage drifts from the exact terminology in the document.

---

## 6. BM25 Index Stored Separately from ChromaDB

**Decision:** Serialize the BM25 index to disk as a pickle file alongside ChromaDB rather than relying solely on ChromaDB for retrieval.

**Why:** ChromaDB only supports vector similarity search — it has no BM25 capability. BM25 requires an inverted index, a fundamentally different data structure from ChromaDB's HNSW vector index. The two indexes must coexist as separate artifacts built from identical chunks so that chunk IDs mean the same thing across both.

The BM25 pickle stores the index object, the ordered list of chunk IDs, the chunk texts, and the chunk metadata. All four must be saved together because BM25 returns positional scores — without the corresponding ID list, the scores cannot be mapped back to actual chunks.

**Key implementation detail:** The BM25 index is loaded once at module level when the retriever is first imported, not on every query. Loading from pickle on every request would add significant latency. When a new document is ingested, `reload_index()` is called after rebuilding the pickle to keep the in-memory index in sync without restarting the server.

**Engineering instinct:** Before implementing, the question was raised — why store full chunk metadata in the pickle when ChromaDB already has it? The answer is that a second ChromaDB lookup after every BM25 query adds unnecessary database traffic, especially since both retrievers run simultaneously during query time. Keeping everything in the pickle eliminates that round trip entirely.

---

## 7. Source Filename Stored as Metadata, Not Full Path

**Decision:** Store only the filename (`KYC09062025.pdf`) in ChromaDB metadata rather than the full absolute path (`C:\Users\...\data\raw\KYC09062025.pdf`).

**Why:** This was caught during testing when the `/delete` endpoint consistently returned "file not found" despite the file clearly existing in the database. The root cause was that documents were ingested with full absolute paths as `source_file` metadata but the delete endpoint was querying by filename only. The mismatch meant the ChromaDB `where` filter never matched.

Storing only the filename also makes the system portable — the database works correctly regardless of which machine it runs on or where the project is located on disk.

---

## 8. Duplicate Detection Before Ingestion

**Decision:** Before ingesting a new PDF, query ChromaDB for existing chunks with the same `source_file` metadata. If any exist, reject the upload with a 400 error rather than silently adding duplicate chunks.

**Why:** Without this check, uploading the same PDF twice doubles the chunks for that document in ChromaDB. This corrupts both the vector index (duplicate embeddings) and the BM25 index (duplicate term frequencies that skew IDF scores). Duplicate chunks also cause the same content to appear multiple times in retrieved results, wasting the reranker's candidate budget.

An explicit rejection with a clear error message is better than silent deduplication — the admin knows exactly what happened and can choose to delete the existing document before re-ingesting an updated version.

**Alternative rejected:** Silent deduplication by chunk ID. This would silently skip already-existing chunks, giving the admin no feedback about whether the upload actually changed anything.

---

## 9. BM25 Index Hot-Reloading on Ingestion

**Decision:** After rebuilding the BM25 pickle on every new document ingestion, call `reload_index()` in the BM25 retriever module to update the in-memory index without restarting the server.

**Why:** The BM25 index is loaded at module level when the server starts. Without hot-reloading, ingesting a new document updates the pickle on disk but the in-memory index remains stale — the new document's chunks are in ChromaDB but invisible to BM25 retrieval until the server restarts. This is a subtle bug that would be hard to detect in testing because ChromaDB retrieval would work correctly while BM25 retrieval silently missed new content.

Hot-reloading eliminates this inconsistency. The in-memory index and the on-disk pickle always reflect the same corpus state after every ingestion.

---

## 10. Grounding Check as a Post-Generation Safety Layer

**Decision:** After generating an answer, run a second LLM call that verifies every claim in the answer is supported by the retrieved chunks before returning the response.

**Why:** The generation LLM occasionally "fills in" details that aren't explicitly stated in the chunks — reasonable inferences that go beyond what the source documents actually say. In a financial regulation context, a reasonable inference that happens to be wrong is still a wrong answer. The grounding check catches these cases by treating the problem as textual entailment: does the answer stay strictly within what the chunks say?

If the grounding check fails, the system abstains rather than retrying generation. Retrying risks the same behaviour recurring with different wording.

**Note on the judge LLM:** The grounding check uses the same `llama-3.1-8b-instant` as the generator. The judge doesn't need domain knowledge — it only needs to check whether the answer text is entailed by the context text. A smaller, faster model is appropriate here.

---

## 11. Separation of Helper Functions from LangGraph Nodes

**Decision:** Every component — retriever, reranker, generator, grounding check — is implemented as a standalone function with clean typed inputs and outputs. LangGraph nodes are thin wrappers that read from state, call these functions, and write results back to state.

**Why:** This separation means every component can be tested independently without needing a full LangGraph state object. During development, each module was tested in isolation — BM25 retrieval, merger deduplication, reranker scoring — before being wired into the graph. If a node fails in production, the error is immediately locatable to the underlying function rather than tangled with graph orchestration logic.

It also means the helper functions are reusable outside the graph. The retriever, reranker, and generator could be used in a different orchestration framework or a simple sequential script without any changes.

---

## 12. LangChain Integration for ChromaDB Over Raw Client

**Decision:** Use LangChain's ChromaDB wrapper (`langchain-chroma`) rather than the raw ChromaDB Python client.

**Why:** The raw ChromaDB client requires manually calling the embedding model, converting results, and managing document insertion — three separate steps that LangChain collapses into one `add_documents()` call. LangChain handles embedding generation internally when a document is added, and returns plain text documents directly on retrieval rather than raw vectors. This means the rest of the pipeline — retriever, reranker, generator — thinks entirely in terms of text and never touches a vector directly.

**Engineering instinct:** This decision came from pushing back on the initial approach of using `SentenceTransformer` directly with the raw ChromaDB client. The observation was that LangChain's `HuggingFaceEmbeddings` wrapper provides the same model behind a LangChain-compatible interface, and `langchain-chroma` then handles everything else transparently. The simplification removed roughly 30 lines of manual embedding and insertion code with no loss of control or functionality.

---

## 13. Absolute Paths via `__file__` Over Relative Paths

**Decision:** Resolve all file paths using `Path(__file__).resolve().parent` rather than hardcoded relative paths like `../../data/chroma_db`.

**Why:** Relative paths resolve from the current working directory at runtime, not from the file's location. A path like `../../data/chroma_db` works only if the server is started from a specific directory. If `uvicorn` is run from a different directory, or if a module deep in `src/ingestion/` tries to resolve the same path, the resolution silently points somewhere wrong and the application fails with a misleading "file not found" error rather than a path error.

`Path(__file__).resolve()` always returns the absolute path of the current file, regardless of where the process was started from. Paths built relative to `__file__` work correctly from any working directory.

**Engineering instinct:** This was caught early when the suggestion to use `../../data/...` was questioned directly — "this works only if I always run from the same folder." That instinct is correct and the fix prevents an entire class of environment-dependent bugs.

---

## 14. Studying Corpus Structure Before Deciding Chunking Strategy

**Decision:** Download and inspect actual PDF documents before implementing any chunking logic. Study the raw extracted text chunk by chunk before deciding whether character-based splitting or FAQ-aware splitting is appropriate.

**Why:** FAQ-aware chunking — splitting on question boundaries like `Q.1`, `Query`, or bold text — seems like the obvious choice for FAQ documents. But RBI and SEBI publish FAQs in at least three different formats across their document library. A chunker designed for one format breaks silently on others, producing malformed chunks without any error.

By printing the first 15 chunks from several different PDFs before writing the chunking code, it became clear that formats varied enough that a single pattern-based splitter would not generalise. The decision to use `RecursiveCharacterTextSplitter` with overlap was made after seeing the data, not before — because the overlap ensures that content at chunk boundaries appears in at least two consecutive chunks, providing a safety net against mid-sentence splits regardless of the document format.

**Engineering instinct:** The refusal to implement FAQ-aware chunking before studying the actual corpus was the right call. Premature optimisation for one document format would have created a brittle ingestion pipeline that failed silently on a significant portion of the corpus.

---

## 15. Honest Evaluation — Keeping Incorrect Answers in the Golden Set

**Decision:** Include answers that the system got wrong in the RAGAS evaluation golden set rather than removing them to improve the reported scores.

**Why:** A golden evaluation set that only contains queries the system answered correctly is not an evaluation — it is a demonstration. The purpose of evaluation is to measure real system performance, including failures. Removing incorrect answers would inflate faithfulness and answer relevancy scores in a way that does not reflect how the system would perform on real user queries.

Two specific cases were kept despite knowing they would hurt the scores: an answer about ₹1, ₹2, and ₹5 notes that was partially incorrect (the system said ₹1 notes were discontinued when they are still issued by the Government of India), and an answer about public deposits that was verbose and uncertain in tone. Both remained in the evaluation set because they represent real system behaviour on real corpus content.

**Engineering instinct:** The explicit choice to not filter the evaluation set for score inflation, even when it would have been easy and undetectable, reflects the right engineering standard. Evaluation numbers on a resume mean nothing if they were produced by a curated subset designed to make the system look better than it is.

---

## 16. Documenting Conversational Query Failure as a Known Limitation

**Decision:** When the system was observed to abstain on long conversational queries that should have been answerable, the decision was made to document this as a known limitation rather than add a query preprocessing node to fix it.

**Why:** The root cause is clear — `ms-marco-MiniLM` was trained on short keyword queries from MS MARCO, not on multi-sentence narrative input. The cross-encoder scores a long story-format query poorly against short regulatory chunks even when the semantic content is relevant. The fix is a preprocessing step that extracts the core question from a conversational query before retrieval.

The decision not to implement this fix was a conscious scope decision, not an oversight. Adding a preprocessing node would have extended the pipeline significantly, introduced another LLM call on every query, and required additional evaluation to verify it didn't hurt performance on direct queries. Given the target user base of common people asking genuine questions — who are more likely to ask "What happens if I don't maintain minimum balance?" than "I've been a customer for 10 years and I wanted to know about minimum balance rules..." — the failure mode affects a minority of real queries.

Documenting a limitation with its root cause and known fix is more valuable than silently patching it in a way that might introduce other failure modes.

---

## 18. FastAPI Backend Over CLI Scripts

**Decision:** Expose the ingestion pipeline and query pipeline as FastAPI endpoints rather than running them as command-line scripts.

**Why:** A running API is more demonstrable, more realistic, and more extensible than CLI scripts. The `/ingest` and `/askquery` endpoints are immediately testable via the built-in `/docs` interface without writing any additional code. It also enforces a clean separation between the pipeline logic (in `src/`) and the HTTP interface (in `main.py`) — the API layer handles request validation, file handling, and response formatting while knowing nothing about how retrieval actually works.

The `/ingest` endpoint is framed as admin-only. In a production deployment it would be behind authentication — the system is designed with that extension point in mind even though authentication was not implemented for this project scope.
