import json
from pathlib import Path

import pandas as pd
import torch
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    precision_recall_fscore_support,
)
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset
from transformers import DataCollatorWithPadding, DistilBertForSequenceClassification, DistilBertTokenizerFast

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = PROJECT_ROOT / "data" / "combined_reviews_with_sentiment.csv"
MODEL_PATH = PROJECT_ROOT / "models" / "distilbert_sentiment"
SAMPLE_SIZE = 100_000
RANDOM_STATE = 42
BATCH_SIZE = 32
MAX_LENGTH = 128


class SentimentDataset(Dataset):
    def __init__(self, texts, labels, tokenizer):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        return {
            **self.tokenizer(
                self.texts[index],
                truncation=True,
                max_length=MAX_LENGTH,
            ),
            "labels": self.labels[index],
        }


def load_test_data(tokenizer):
    df = pd.read_csv(DATA_PATH)
    df = df.dropna(subset=["review_clean", "sentiment"])
    df = df.sample(n=min(SAMPLE_SIZE, len(df)), random_state=RANDOM_STATE)

    texts = df["review_clean"].astype(str).tolist()
    labels = df["sentiment"].str.upper().map({"POSITIVE": 1, "NEGATIVE": 0}).tolist()

    _, test_texts, _, test_labels = train_test_split(
        texts,
        labels,
        test_size=0.2,
        random_state=RANDOM_STATE,
        stratify=labels,
    )
    return SentimentDataset(test_texts, test_labels, tokenizer), test_labels


def evaluate_model():
    tokenizer = DistilBertTokenizerFast.from_pretrained(str(MODEL_PATH))
    model = DistilBertForSequenceClassification.from_pretrained(str(MODEL_PATH))
    model.config.id2label = {0: "NEGATIVE", 1: "POSITIVE"}
    model.config.label2id = {"NEGATIVE": 0, "POSITIVE": 1}

    test_dataset, test_labels = load_test_data(tokenizer)
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
    data_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        collate_fn=data_collator,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

    predicted_labels = []
    with torch.no_grad():
        for batch in data_loader:
            labels = batch.pop("labels")
            inputs = {key: value.to(device) for key, value in batch.items()}
            logits = model(**inputs).logits
            predicted_labels.extend(logits.argmax(dim=-1).cpu().tolist())

    class_ids = [0, 1]
    class_names = ["NEGATIVE", "POSITIVE"]
    precision, recall, f1, _ = precision_recall_fscore_support(
        test_labels,
        predicted_labels,
        labels=class_ids,
        zero_division=0,
    )
    report = classification_report(
        test_labels,
        predicted_labels,
        labels=class_ids,
        target_names=class_names,
        output_dict=True,
        zero_division=0,
    )
    results = {
        "device": str(device),
        "test_samples": len(test_labels),
        "accuracy": accuracy_score(test_labels, predicted_labels),
        "precision": dict(zip(class_names, precision.tolist())),
        "recall": dict(zip(class_names, recall.tolist())),
        "f1_score": dict(zip(class_names, f1.tolist())),
        "classification_report": report,
        "confusion_matrix": confusion_matrix(
            test_labels,
            predicted_labels,
            labels=class_ids,
        ).tolist(),
    }

    print("Evaluation results:")
    print(f"Device: {results['device']}")
    print(f"Test samples: {results['test_samples']}")
    print(f"Accuracy: {results['accuracy']:.4f}")
    print(
        classification_report(
            test_labels,
            predicted_labels,
            labels=class_ids,
            target_names=class_names,
            zero_division=0,
        )
    )
    print("Confusion matrix (rows=true, columns=predicted):")
    print(results["confusion_matrix"])

    output_path = MODEL_PATH / "evaluation_metrics.json"
    with output_path.open("w", encoding="utf-8") as metrics_file:
        json.dump(results, metrics_file, indent=2)
    print(f"Saved evaluation results to: {output_path}")


if __name__ == "__main__":
    evaluate_model()
