# rag_query.py (English-only, optimized)
import json
import faiss
import requests
from sentence_transformers import SentenceTransformer

INDEX_PATH = "bvb.index"
CHUNKS_PATH = "bvb_chunks.json"

EMBEDDING_MODEL = "all-mpnet-base-v2"
OLLAMA_MODEL = "phi3:mini"

TOP_K = 4
MAX_CONTEXT_CHARS = 1500

def limit_context(chunks):
    text = ""
    for c in chunks:
        if len(text) + len(c) > MAX_CONTEXT_CHARS:
            break
        text += c + "\n\n"
    return text.strip()


def build_prompt(context, question):
    return f"""
Use ONLY the provided context. 
If the information is not in the context, respond: "Not stated in the report". 
Do NOT guess. 
Provide exact names/numbers as in the context.

Context:
{context}

Question:
{question}

Answer (concise, factual, no explanation):
"""


def query_ollama(prompt: str) -> str:
    response = requests.post(
        "http://localhost:11434/api/generate",
        json={
            "model": OLLAMA_MODEL,
            "prompt": prompt,
            "stream": False,
            "options": {
                "num_predict": 120,
                "temperature": 0.0,
            },
        },
        timeout=300,
    )
    response.raise_for_status()
    return response.json()["response"].strip()


def main():
    print("📦 Loading FAISS index and chunks...")
    index = faiss.read_index(INDEX_PATH)

    with open(CHUNKS_PATH, "r", encoding="utf-8") as f:
        chunks = json.load(f)

    embedder = SentenceTransformer(EMBEDDING_MODEL, device="cpu")

    while True:
        question = input("\n❓ Question (ENTER to quit): ")
        if not question.strip():
            break

        q_emb = embedder.encode([question], convert_to_numpy=True)
        _, indices = index.search(q_emb, TOP_K)

        retrieved = [chunks[i] for i in indices[0]]
        context = limit_context(retrieved)

        prompt = build_prompt(context, question)
        answer = query_ollama(prompt)

        print("\n🧠 Answer:\n")
        print(answer)


if __name__ == "__main__":
    main()
