from langchain_text_splitters import RecursiveCharacterTextSplitter
from src.models import RawPage, Chunk
import hashlib

def generate_chunk_id(source_file: str, chunk_index: int) -> str:
    raw = f"{source_file}_{chunk_index}"
    return hashlib.sha256(raw.encode()).hexdigest()[:16]

def chunk_doc(pages: list[RawPage]) -> list[Chunk]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size = 500,
        chunk_overlap = 100,
        length_function = len
    )

    chunks = []
    chunk_index = 0

    for page in pages:
        splits = splitter.split_text(page.text)

        for split in splits:
            chunk_id = generate_chunk_id(page.source_file, chunk_index)

            chunks.append(Chunk(
                chunk_id= chunk_id,
                text= split,
                source_file= page.source_file,
                page_num= page.page_num,
                chunk_index= chunk_index
            ))
            chunk_index+=1

    return chunks