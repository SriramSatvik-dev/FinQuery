from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from src.ingestion.loader import load_data
from src.ingestion.chunker import chunk_doc
from src.ingestion.embedder import embed_and_store, delete_doc
from pathlib import Path
import shutil

app = FastAPI(title="FinQuery")

app.add_middleware(
    CORSMiddleware,
    allow_origins=['*'],
    allow_credentials=True,
    allow_methods=['*'],
    allow_headers=['*']
)

@app.get("/health")
async def health():
    return {"status": "ok"}

BASE_DIR = Path(__file__).resolve().parent

@app.post("/ingest")
async def uploadroute(file: UploadFile = File(...)):
    print(file.filename)
    print(file.content_type)

    if(file.content_type != "application/pdf"):
        raise HTTPException(status_code=400, detail="Only pdf files accepted")

    save_path = str(BASE_DIR / "data" / "raw" / file.filename)

    with open(save_path, "wb") as f:
        shutil.copyfileobj(file.file, f)

    raw_doc = load_data(save_path)
    chunks = chunk_doc(raw_doc)
    embed_and_store(chunks, file.filename)

    return {
    "message": "Document ingested successfully",
    "filename": file.filename,
    "pages": len(raw_doc),
    "chunks": len(chunks)
    }

@app.delete("/delete/{filename}")
async def delete_route(filename: str):
    chunks_deleted = delete_doc(filename)

    pdf_path = BASE_DIR / "data" / "raw" / filename
    if pdf_path.exists():
        pdf_path.unlink()

    return {
        "message": f"{filename} deleted successfully",
        "chunks_deleted": chunks_deleted
    }