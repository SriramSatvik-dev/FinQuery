from src.ingestion.embedder import vector_store
from src.models import Chunk

def query(query_text: str, top_k: int = 10) -> list[Chunk]:
    search_results = vector_store.similarity_search(query= query_text, k= top_k)
    
    chunks = []

    for doc in search_results:
        metadata = doc.metadata
        text = doc.page_content
        chunk = Chunk(
            chunk_id=metadata["chunk_id"],
            text= text,
            source_file= metadata["source_file"],
            page_num= metadata["page_num"],
            chunk_index= metadata["chunk_index"]
        )
        chunks.append(chunk)

    return chunks