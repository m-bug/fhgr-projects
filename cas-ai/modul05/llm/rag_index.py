"""
rag_index.py
------------

Local Retrieval-Augmented Generation (RAG) indexing pipeline
optimized for FAST query-time inference.

Key goals:
- Fewer but higher-quality chunks
- Faster FAISS retrieval
- Less prompt noise for the LLM

PDF -> Text -> Cleaning -> Chunking (+ overlap)
-> Embeddings (normalized, float16)
-> FAISS IVF index

download pdf: https://report.bvb.de/annual-report/2024-2025/services/downloads.html

"""

from pathlib import Path
import json
import pdfplumber
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer

PDF_PATH = "gesamt-bvb-gb2425_en.pdf"
INDEX_PATH = "bvb.index"
CHUNKS_PATH = "bvb_chunks.json"

EMBEDDING_MODEL = "all-mpnet-base-v2"


# -----------------------------
# PDF extraction
# -----------------------------
def extract_pdf_text(pdf_path: Path) -> str:
    pages = []
    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            text = page.extract_text()
            if text:
                pages.append(text)
    return "\n".join(pages)


# -----------------------------
# Cleaning (important for LLM speed)
# -----------------------------
def clean_text(text: str) -> str:
    cleaned = []
    for line in text.splitlines():
        line = line.strip()

        if len(line) < 30:
            continue
        if line.isdigit():
            continue
        if line.lower().startswith(("seite ", "page ")):
            continue

        cleaned.append(line)

    return "\n".join(cleaned)


# -----------------------------
# Chunking optimized for retrieval
# -----------------------------
def chunk_text(text: str, max_chars=1200, overlap=250):
    chunks = []
    start = 0
    length = len(text)

    while start < length:
        end = start + max_chars
        chunk = text[start:end].strip()

        if len(chunk) > 200:
            chunks.append(chunk)

        start = end - overlap

    return chunks



# -----------------------------
# Main indexing pipeline
# -----------------------------
def main():
    print("📄 Extracting PDF...")
    raw_text = extract_pdf_text(Path(PDF_PATH))

    print("🧹 Cleaning text...")
    raw_text = clean_text(raw_text)

    print("✂️ Chunking...")
    chunks = chunk_text(raw_text)
    print(f"   → {len(chunks)} chunks")

    print("🧠 Creating embeddings...")
    model = SentenceTransformer(EMBEDDING_MODEL, device="cpu")

    embeddings = model.encode(
        chunks,
        convert_to_numpy=True,
        show_progress_bar=True,
        normalize_embeddings=True  # IMPORTANT for faster & better similarity
    ).astype("float16")

    dim = embeddings.shape[1]

    # -----------------------------
    # FAISS index optimized for fast queries
    # -----------------------------
    print("📐 Building FAISS IVF index...")
    nlist = max(16, int(len(chunks) ** 0.5))  # adaptive clustering
    quantizer = faiss.IndexFlatL2(dim)
    index = faiss.IndexIVFFlat(quantizer, dim, nlist)

    index.train(embeddings)
    index.add(embeddings)
    index.nprobe = 8  # query-time speed/quality tradeoff

    print("💾 Saving FAISS index...")
    faiss.write_index(index, INDEX_PATH)

    print("💾 Saving chunks...")
    with open(CHUNKS_PATH, "w", encoding="utf-8") as f:
        json.dump(chunks, f, ensure_ascii=False, indent=2)

    print("✅ RAG index created (query-optimized)")


if __name__ == "__main__":
    main()
