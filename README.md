# Food review sentiment analysis

A Streamlit app for classifying food reviews as positive or negative and exploring word-level SHAP contributions.

## Run locally

From the project root, install the app dependencies and start Streamlit:

```powershell
python -m pip install -r requirements.txt
python -m streamlit run src/app.py
```

The app uses the local model in `models/distilbert_sentiment` when present. Otherwise, it downloads the public fine-tuned model from [Hugging Face](https://huggingface.co/ggopakumar/food-review-sentiment).

## Deploy on Streamlit Community Cloud

- Repository entry point: `src/app.py`
- Dependencies: `requirements.txt`
- Model: `ggopakumar/food-review-sentiment` on Hugging Face

The local dataset, model folder, virtual environment, and editor settings are excluded from this repository. Do not add private data or credentials.

## Project scripts

- `src/app.py`: Streamlit app and word-level sentiment explanations
- `src/train.py`: fine-tunes the DistilBERT classifier
- `src/evaluate.py`: evaluates the classifier
- `src/explain.py`: SHAP explanation helper
