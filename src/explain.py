from transformers import pipeline
import shap
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODEL_PATH = PROJECT_ROOT / "models" / "distilbert_sentiment"

# Load fine-tuned model
sentiment_pipeline = pipeline("sentiment-analysis", model=str(MODEL_PATH))
sentiment_pipeline.model.config.id2label = {
    0: "NEGATIVE",
    1: "POSITIVE",
}
sentiment_pipeline.model.config.label2id = {
    "NEGATIVE": 0,
    "POSITIVE": 1,
}

def explain_text(text):
    explainer = shap.Explainer(sentiment_pipeline)
    shap_values = explainer([text])
    return shap.plots.text(shap_values[0])
