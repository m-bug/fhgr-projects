## pip install gpt4all

from gpt4all import GPT4All

model = GPT4All(
    "Llama-3.2-3B-Instruct-Q4_0.gguf",
    device="cpu"
)

with model.chat_session():
    response = model.generate("Erkläre mir kurz, was ein LLM ist (im Kontext von maschinellem Lernen).")
    print(response)
