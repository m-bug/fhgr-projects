## pip install gpt4all

from gpt4all import GPT4All
from pypdf import PdfReader

# PDF einlesen
reader = PdfReader("bezoz_2026_project_prometheus_tnyt.pdf")
text = ""
for page in reader.pages:
    text += page.extract_text()

# Modell laden
model = GPT4All("Llama-3.2-3B-Instruct-Q4_0.gguf")

prompt = f"""
Analysiere folgendes Dokument und fasse die wichtigsten Fakten zusammen:

{text[:4000]}
"""

with model.chat_session():
    response = model.generate(prompt)

print(response)