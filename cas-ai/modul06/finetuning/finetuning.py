from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
from datasets import Dataset
import torch

# pip install torch transformers datasets accelerate

# vortrainiertes Modell und Tokenizer laden
model_name = "distilbert-base-uncased"  # Für Textklassifikation
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=3)  # z.B. Bug, Feature, Support

# Beispiel-Datensatz (Log-Daten und Jira-Antworten)
data = [
    {"log": "Error: Database connection failed. Timeout after 30 seconds.", "label": 0},  # Bug
    {"log": "Request: User needs access to new feature.", "label": 1},  # Feature
    {"log": "Warning: Disk space running low.", "label": 2},  # Support
]

# konvertieren in Hugging Face Dataset-Format
dataset = Dataset.from_dict(data)

# Tokenisieren der Daten
def tokenize_data(example):
    return tokenizer(example['log'], padding='max_length', truncation=True, max_length=512)

dataset = dataset.map(tokenize_data, batched=True)

# Trainer-Setup für das Finetuning
training_args = TrainingArguments(
    output_dir="./results",          # Verzeichnis für Ergebnisse
    evaluation_strategy="epoch",     # Evaluation nach jeder Epoche
    learning_rate=2e-5,              # Lernrate
    per_device_train_batch_size=4,   # Batch-Grösse
    num_train_epochs=3,              # Epochenanzahl
    weight_decay=0.01,               # Gewichtung der Regularisierung
)

trainer = Trainer(
    model=model,                     # Das zu trainierende Modell
    args=training_args,              # Trainingseinstellungen
    train_dataset=dataset,           # Trainingsdatensatz
)

# Trainiere das Modell
trainer.train()

# Modell abspeichern
model.save_pretrained("./finetuned_log_model")
tokenizer.save_pretrained("./finetuned_log_model")

# Testen des Modells mit einem neuen Log-Ereignis
def predict_issue(log):
    inputs = tokenizer(log, return_tensors="pt", padding=True, truncation=True)
    with torch.no_grad():
        outputs = model(**inputs)
        predicted_class = outputs.logits.argmax().item()  # Gibt den vorhergesagten Issue-Typ zurück
    return predicted_class

# Beispiel
log = "Error: Database connection failed. Timeout after 30 seconds."
predicted_issue = predict_issue(log)
print(f"Vorhergesagter Jira-Issue-Typ: {predicted_issue}")