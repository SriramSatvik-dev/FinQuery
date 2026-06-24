from dataclasses import dataclass

@dataclass
class RawPage:
    text:  str
    source_file: str
    page_num: int

@dataclass
class Chunk:
    chunk_id: str
    text: str
    source_file: str
    page_num: int
    chunk_index: int

@dataclass
class RankedChunk:
    chunk: Chunk
    relevance_score: float

@dataclass
class Citation:
    chunk_id: str
    source_file: str
    page_num: int
    excerpt: str