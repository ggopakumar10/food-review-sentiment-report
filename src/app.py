import streamlit as st
from pathlib import Path
import numpy as np
import pandas as pd
import shap
from transformers import pipeline
import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODEL_PATH = PROJECT_ROOT / "models" / "distilbert_sentiment"
MODEL_ID = "ggopakumar/food-review-sentiment"
MODEL_SOURCE = str(MODEL_PATH) if MODEL_PATH.is_dir() else MODEL_ID

@st.cache_resource
def load_model():
    device = 0 if torch.cuda.is_available() else -1
    pipe = pipeline("sentiment-analysis", model=MODEL_SOURCE, device=device)
    pipe.model.config.id2label = {
        0: "NEGATIVE",
        1: "POSITIVE",
    }
    pipe.model.config.label2id = {
        "NEGATIVE": 0,
        "POSITIVE": 1,
    }
    return pipe

sentiment_pipeline = load_model()

def predict_proba(texts):
    if isinstance(texts, str):
        texts = [texts]
    elif isinstance(texts, np.ndarray):
        texts = texts.reshape(-1).tolist()
    else:
        texts = list(texts)

    if not all(isinstance(text, str) for text in texts):
        raise TypeError("Sentiment predictions require a string or a sequence of strings.")

    outputs = sentiment_pipeline(texts)
    probs = []

    for out in outputs:
        score = float(out["score"])
        label = out["label"]

        if label == "POSITIVE":
            probs.append([1.0 - score, score])
        else:
            probs.append([score, 1.0 - score])

    return np.asarray(probs)

st.title("Sentiment Analysis Demo (DistilBERT)")

user_input = st.text_area("Enter a review:")

if st.button("Analyze") and user_input.strip():
    st.session_state.analysis_text = user_input.strip()
    explainer = shap.Explainer(predict_proba, masker=shap.maskers.Text(r'\W+'))
    shap_values = explainer([st.session_state.analysis_text])
    st.session_state.word_shap_values = {
        "tokens": [str(token).strip() for token in shap_values.data[0]],
        "values": shap_values.values[0].tolist(),
        "base_values": shap_values.base_values[0].tolist(),
    }

if (
    st.session_state.get("analysis_text") == user_input.strip()
    and st.session_state.get("word_shap_values") is not None
):
    analysis_text = st.session_state.analysis_text
    result = sentiment_pipeline(analysis_text)[0]
    predicted_label = result["label"]
    st.write(f"**Sentiment:** {predicted_label} ({result['score'] * 100:.2f}% confidence)")

    st.write("Word-level contributions")
    selected_label = st.segmented_control(
        "Show contributions for",
        options=["NEGATIVE", "POSITIVE"],
        default=predicted_label,
        key="contribution_sentiment",
    )
    selected_index = {"NEGATIVE": 0, "POSITIVE": 1}[selected_label]
    explanation = st.session_state.word_shap_values
    contributions = np.asarray(explanation["values"])[:, selected_index]
    baseline = float(explanation["base_values"][selected_index])
    explained_probability = baseline + float(contributions.sum())

    baseline_col, explained_col, change_col = st.columns(3)
    baseline_col.metric("Baseline probability", f"{baseline:.1%}")
    explained_col.metric("Explained probability", f"{explained_probability:.1%}")
    change_col.metric("Net contribution", f"{contributions.sum():+.1%}")
    st.caption(
        "Each word's contribution is measured in percentage points. "
        "A positive value raises the selected sentiment probability; "
        "a negative value lowers it. Words are ordered by contribution magnitude."
    )

    contribution_rows = sorted(
        (
            {
                "Word": word or "(space)",
                "Contribution (percentage points)": value * 100,
                "Effect": (
                    f"Increases {selected_label.lower()} probability"
                    if value > 0
                    else f"Decreases {selected_label.lower()} probability"
                    if value < 0
                    else "No change"
                ),
            }
            for word, value in zip(explanation["tokens"], contributions)
        ),
        key=lambda row: abs(row["Contribution (percentage points)"]),
        reverse=True,
    )
    st.dataframe(
        pd.DataFrame(contribution_rows),
        hide_index=True,
        width="stretch",
        column_config={
            "Contribution (percentage points)": st.column_config.NumberColumn(
                format="%+.2f pp"
            )
        },
    )
