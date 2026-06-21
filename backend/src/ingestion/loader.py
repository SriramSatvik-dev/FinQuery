import fitz
from src.models import RawPage
from pathlib import Path

def load_data(file_path: str) -> list[RawPage]:
    pages = []
    doc = fitz.open(file_path)

    for page_num, page in enumerate(doc, start=1):
        text = page.get_text(sort= True).strip()

        if not text:
            continue

        pages.append(RawPage(
            text= text,
            source_file= Path(file_path).name,
            page_num= page_num
        ))

    if all(not page.text for page in pages):
        raise ValueError(f"{file_path} appears to be scanned doc. No text extracted")

    doc.close()
    return pages