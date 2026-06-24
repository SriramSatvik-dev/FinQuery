import pickle
from pathlib import Path
from src.models import Chunk

BASE_DIR = Path(__file__).resolve().parent.parent.parent
BM25_PATH = str(BASE_DIR / "data" / "processed" / "bm25_index.pkl")

with open(BM25_PATH, "rb") as f:
    bm25_data = pickle.load(f)

bm25 = bm25_data["bm25"]
all_ids = bm25_data["ids"]
all_texts = bm25_data["texts"]
all_metadatas = bm25_data["metadatas"]

def query(query_text: str, top_k: int = 10) -> list[Chunk]:
    query_tokens = query_text.lower().split(sep=' ')
    bm25_scores = bm25.get_scores(query_tokens)
    
    top_indices = sorted(
        range(len(bm25_scores)),
        key=lambda i: bm25_scores[i],
        reverse= True
    )[:top_k]

    chunks = []
    for idx in top_indices:
        metadata = all_metadatas[idx]
        chunks.append(Chunk(
            chunk_id= metadata["chunk_id"],
            text= all_texts[idx],
            source_file= metadata["source_file"],
            page_num= metadata["page_num"],
            chunk_index= metadata["chunk_index"]
        ))

    return chunks


def reload_index() -> None:
    global bm25, all_ids, all_texts, all_metadatas
    
    with open(BM25_PATH, "rb") as f:
        bm25_data = pickle.load(f)
    
    bm25 = bm25_data["bm25"]
    all_ids = bm25_data["ids"]
    all_texts = bm25_data["texts"]
    all_metadatas = bm25_data["metadatas"]