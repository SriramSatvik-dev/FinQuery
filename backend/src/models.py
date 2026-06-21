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