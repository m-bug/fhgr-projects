"""
rag_bvb.py
----------

Local Retrieval-Augmented Generation (RAG) pipeline for querying
a Borussia Dortmund (BVB) business report PDF using a local LLaMA
model via Ollama.

Pipeline:
PDF -> Text extraction -> Cleaning -> Chunking -> Embeddings (SentenceTransformer)
-> Vector search (FAISS) -> Prompt -> LLaMA (Ollama)

Requirements:
pip install pdfplumber sentence-transformers faiss-cpu requests
"""

from pathlib import Path
import re
import pdfplumber
import requests
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer


# ---------------------------------------------------------------------
# 1. PDF TEXT EXTRACTION
# ---------------------------------------------------------------------

def extract_pdf_text(pdf_path: Path) -> str:
    """Extract raw text from a PDF file."""
    text_parts = []
    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            txt = page.extract_text()
            if txt:
                text_parts.append(txt)
    return "\n".join(text_parts)


# ---------------------------------------------------------------------
# 2. TEXT NORMALIZATION
# ---------------------------------------------------------------------

def normalize_text(text: str) -> str:
    """
    Light cleanup:
    - collapse excessive newlines
    - normalize whitespace
    """
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = re.sub(r"[ \t]+", " ", text)
    return text.strip()


# ---------------------------------------------------------------------
# 3. CHUNKING (PARAGRAPH-BASED)
# ---------------------------------------------------------------------

def chunk_text(text: str, max_chars: int = 900):
    """
    Split text into semantically meaningful chunks based on paragraphs.
    """
    paragraphs = text.split("\n\n")
    chunks = []
    current = ""

    for para in paragraphs:
        if len(current) + len(para) > max_chars:
            if current.strip():
                chunks.append(current.strip())
            current = para
        else:
            current += "\n\n" + para

    if current.strip():
        chunks.append(current.strip())

    return chunks


# ---------------------------------------------------------------------
# 4. VECTOR STORE (EMBEDDINGS + FAISS)
# ---------------------------------------------------------------------

class VectorStore:
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        self.embedder = SentenceTransformer(model_name)
        self.index = None
        self.texts = []

    def add_texts(self, texts):
        texts = [t for t in texts if len(t.strip()) > 50]
        if not texts:
            raise ValueError("No valid text chunks to index.")

        embeddings = self.embedder.encode(texts, convert_to_numpy=True)

        if self.index is None:
            self.index = faiss.IndexFlatL2(embeddings.shape[1])

        self.index.add(embeddings)
        self.texts.extend(texts)

    def search(self, query: str, k: int = 4):
        if self.index is None:
            raise RuntimeError("Vector index is empty.")

        q_emb = self.embedder.encode([query], convert_to_numpy=True)
        _, indices = self.index.search(q_emb, k)

        return [self.texts[i] for i in indices[0] if i < len(self.texts)]


# ---------------------------------------------------------------------
# 5. PROMPTING + LOCAL LLAMA (OLLAMA)
# ---------------------------------------------------------------------

def build_prompt(context_chunks, user_query):
    context = "\n\n".join(context_chunks)
    return (
        "You are an expert analyst for football club business reports.\n"
        "You answer questions STRICTLY based on the provided context.\n\n"
        "Rules:\n"
        "- Use ONLY the provided context\n"
        "- If the answer is not explicitly stated, say: 'Not stated in the report'\n"
        "- Be factual and concise\n\n"
        "Context:\n"
        f"{context}\n\n"
        "Question:\n"
        f"{user_query}\n\n"
        "Answer:\n"
    )


def query_local_llama(prompt: str, model: str = "llama3.2:3b") -> str:
    """Send prompt to a local Ollama LLaMA instance."""
    url = "http://localhost:11434/api/generate"
    payload = {
        "model": model,
        "prompt": prompt,
        "stream": False
    }

    response = requests.post(url, json=payload, timeout=300)
    response.raise_for_status()
    return response.json()["response"]


# ---------------------------------------------------------------------
# 6. MAIN RAG PIPELINE
# ---------------------------------------------------------------------

def run_rag(pdf_path: str, query: str):
    print("Loading and processing PDF...")
    raw_text = extract_pdf_text(Path(pdf_path))
    clean_text = normalize_text(raw_text)

    print("Chunking text...")
    chunks = chunk_text(clean_text)

    print(f"Creating embeddings for {len(chunks)} chunks...")
    store = VectorStore()
    store.add_texts(chunks)

    print("Retrieving relevant context...")
    retrieved = store.search(query, k=4)

    prompt = build_prompt(retrieved, query)

    print("Querying local LLaMA via Ollama...")
    answer = query_local_llama(prompt)

    print("\n=== Antwort vom LLaMA ===\n")
    print(answer)


# ---------------------------------------------------------------------
# ENTRY POINT
# ---------------------------------------------------------------------

if __name__ == "__main__":
    run_rag(
        pdf_path="gesamt-bvb-gb2425.pdf",
        query="Who are the members of the executive board (Vorstand)?"
    )