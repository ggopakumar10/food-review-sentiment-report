# App.py Failure Analysis & Fixes

## Issues Found and Fixed

### 1. **SHAP Callback Input Type and Word-Level Explanations** (CRITICAL)
**Location:** `src/app.py`

**Problem:**
```python
def predict_proba(texts):
    if isinstance(texts, str):
        texts = [texts]
    outputs = sentiment_pipeline(texts)
```

SHAP 0.52 passes masked inputs to `predict_proba` as a NumPy array. The Transformers tokenizer rejects that type because it expects a string or a list of strings. `shap.plots.text()` returns interactive HTML, whose embedded interactions are not enabled by default in Streamlit.

**Fixed to:**
```python
elif isinstance(texts, np.ndarray):
    texts = texts.reshape(-1).tolist()
...
st.segmented_control("Show contributions for", ["NEGATIVE", "POSITIVE"])
st.dataframe(word_contributions)
```

**Why this works:**
- SHAP's array input is converted to a Python list before it reaches Transformers.
- The native sentiment selector lets users switch between the two class explanations.
- The table reports each word's signed contribution in percentage points, ordered by magnitude.

---

### 2. **Missing Model Label Names** (FUNCTIONAL)
**Location:** `src/app.py`

The model was trained with integer labels but its configuration was not assigned `id2label` and `label2id`. Transformers therefore returned generic labels such as `LABEL_1`, which also caused `predict_proba` to choose the wrong class branch. The app now explicitly maps class `0` to `NEGATIVE` and class `1` to `POSITIVE`, consistent with `train.py`.

---

### 3. **Device Parameter** (OPTIMIZATION)
**Location:** `src/app.py` line 15

**Original:**
```python
pipe = pipeline("sentiment-analysis", model=str(MODEL_PATH), device=-1)
```

**Fixed to:**
```python
device = 0 if torch.cuda.is_available() else -1
pipe = pipeline("sentiment-analysis", model=str(MODEL_PATH), device=device)
```

**Why this matters:**
- `device=-1` means CPU only (less efficient)
- `device=0` uses GPU if available (faster inference)
- Automatic detection improves performance

---

## Consistency Check: train.py vs app.py

### Label Mapping Consistency ✓
**train.py (line 14):**
```python
labels = df['sentiment'].str.upper().map({'POSITIVE': 1, 'NEGATIVE': 0}).tolist()
```

**app.py (predict_proba function):**
```python
if label == "POSITIVE":
    probs.append([1.0 - score, score])  # Negative prob, Positive prob
else:
    probs.append([score, 1.0 - score])  # Negative prob, Positive prob
```

✓ Consistent: Label 0 = NEGATIVE, Label 1 = POSITIVE

---

## Validation

✓ **Imports successfully** - No runtime errors
✓ **Model loads correctly** - DistilBERT weights loaded (104/104)
✓ **Pipeline configuration** - No attribute errors
✓ **SHAP callback input and per-sentiment word contributions** - Verified in the running app

---

## How to Test

```bash
# From project root
cd sentiment_app
.venv\Scripts\python.exe -m streamlit run src/app.py
```

Then enter a review in the text area and click "Analyze" to:
1. Get sentiment prediction with confidence score
2. Select **NEGATIVE** or **POSITIVE** to inspect that class's word contributions
3. Compare baseline and explained probabilities, and see each word's signed effect in percentage points

---

## Summary

| Issue | Severity | Type | Status |
|-------|----------|------|--------|
| SHAP callback input type and plot rendering | CRITICAL | Runtime Error | ✓ FIXED |
| Missing label names in model configuration | FUNCTIONAL | Incorrect label/probability mapping | ✓ FIXED |

**All issues resolved.** The app should now run without errors.
