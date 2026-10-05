import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split
from transformers import (
    DataCollatorWithPadding,
    DistilBertForSequenceClassification,
    DistilBertTokenizerFast,
    Trainer,
    TrainingArguments,
)
import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = PROJECT_ROOT / "data" / "combined_reviews_with_sentiment.csv"
MODEL_PATH = PROJECT_ROOT / "models" / "distilbert_sentiment"
SAMPLE_SIZE = 100_000
RANDOM_STATE = 42

# Load dataset
df = pd.read_csv(DATA_PATH)
df = df.dropna(subset=['review_clean', 'sentiment'])
df = df.sample(n=min(SAMPLE_SIZE, len(df)), random_state=RANDOM_STATE)
texts = df['review_clean'].astype(str).tolist()
labels = df['sentiment'].str.upper().map({'POSITIVE': 1, 'NEGATIVE': 0}).tolist()

train_texts, test_texts, train_labels, test_labels = train_test_split(
    texts,
    labels,
    test_size=0.2,
    random_state=RANDOM_STATE,
    stratify=labels,
)

tokenizer = DistilBertTokenizerFast.from_pretrained("distilbert-base-uncased")


class SentimentDataset(torch.utils.data.Dataset):
    def __init__(self, texts, labels, tokenizer):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return {
            **self.tokenizer(self.texts[idx], truncation=True, max_length=128),
            "labels": self.labels[idx],
        }

train_dataset = SentimentDataset(train_texts, train_labels, tokenizer)
test_dataset = SentimentDataset(test_texts, test_labels, tokenizer)
data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

model = DistilBertForSequenceClassification.from_pretrained("distilbert-base-uncased", num_labels=2)

training_args = TrainingArguments(
    output_dir=str(MODEL_PATH),
    num_train_epochs=2,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    eval_strategy="epoch",
    save_strategy="epoch",
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=test_dataset,
    data_collator=data_collator,
)

trainer.train()
model.save_pretrained(MODEL_PATH)
tokenizer.save_pretrained(MODEL_PATH)
