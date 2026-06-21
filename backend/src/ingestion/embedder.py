from src.models import Chunk
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from fastapi import HTTPException
from langchain_core.documents import Document
from rank_bm25 import BM25Okapi
import pickle
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent.parent

embedding_model = HuggingFaceEmbeddings(model_name = "sentence-transformers/all-mpnet-base-v2")
chroma_path = str(BASE_DIR / "data" / "chroma_db")
collection_name = "finquery"
bm25_path = str(BASE_DIR / "data" / "processed" / "bm25_index.pkl")

vector_store = Chroma(
    collection_name= collection_name,
    embedding_function= embedding_model,
    persist_directory= chroma_path
)

def delete_doc(filepath: str) -> int:
    existing = vector_store.get(where={"source_file": filepath})
    
    if not existing["ids"]:
        raise HTTPException(status_code=404, detail=f"{filepath} not found")
    
    vector_store.delete(ids=existing["ids"])
    
    _rebuild_bm25()
    
    return len(existing["ids"])

def embed_and_store(chunks: list[Chunk], source_file: str) -> None:
    _check_duplicate(source_file)
    _store_in_vectordb(chunks)
    _rebuild_bm25()

def _check_duplicate(source_file: str) -> None:
    existing = vector_store.get(where= {"source_file": source_file})
    if existing["ids"]:
        raise HTTPException(status_code= 400, detail= f"{source_file} is already ingested. First delete it to update existing data")

def _store_in_vectordb(chunks: list[Chunk]) -> None:
    documents = [
        Document (
            page_content = chunk.text,
            metadata = {
                "source_file": chunk.source_file,
                "page_num": chunk.page_num,
                "chunk_index": chunk.chunk_index,
                "chunk_id": chunk.chunk_id
            }
        )
        for chunk in chunks
    ]

    vector_store.add_documents(documents)
    print(f"Stored {len(documents)} chunks in chromaDB")

def _rebuild_bm25() -> None:
    all_data = vector_store.get()
    all_texts = all_data["documents"]
    all_ids = all_data["ids"]

    if not all_texts:
        print("No chunks in ChromaDB - BM25 index not built")
        return
    
    tokenized = [text.lower().split() for text in all_texts]
    bm_25 = BM25Okapi(tokenized)

    with open(bm25_path, "wb") as f:
        pickle.dump({
            "bm25": bm_25,
            "ids": all_ids,
            "texts": all_texts
        }, f)

    print("bm 25 index rebuilt and stored for all chunks")