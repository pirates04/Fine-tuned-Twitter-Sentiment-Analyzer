import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

MODEL_PATH = "../models/best_model"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_PATH).to(device)

tweets = [
    "I absolutely love this product! 😍",
    "This is the worst thing ever...",
    "It's okay, nothing special."
]

inputs = tokenizer(tweets, return_tensors="pt", padding=True, truncation=True).to(device)
outputs = model(**inputs)
predictions = torch.argmax(outputs.logits, dim=-1)

label_map = {0: "Negative", 1: "Neutral", 2: "Positive"}

for t, p in zip(tweets, predictions):
    print(f"Tweet: {t} | Sentiment: {label_map[p.item()]}")
