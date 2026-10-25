# Spotify App Review Classification Using Machine Learning

Sentiment classification of **61,594 Google Play reviews of the Spotify app** (Jan–Jul 2022) into **negative, neutral and positive** classes using classical machine-learning models on TF-IDF features.

## Highlights

| Configuration | Test accuracy | Macro F1 | Neutral recall |
|---|---|---|---|
| Baseline: stop words removed, unigram TF-IDF + Logistic Regression | 78.72% | 58.37% | 4.6% |
| Baseline: RBF-kernel SVM (best original model) | 79.02% | 56.40% | 0.9% |
| **Proposed: keep stop words, (1,2)-gram TF-IDF + Logistic Regression** | **79.93%** | 60.13% | 6.4% |
| **Proposed: same + balanced class weights** | 75.39% | **65.19%** | **37.5%** |

- 5-fold cross-validation: proposed LR **80.12% ± 0.53** accuracy; balanced LR **66.15% ± 0.66** macro F1.
- Positive vs negative only (neutral excluded): **90.23%** accuracy.

**Key findings**

1. All baseline models almost ignore the neutral (3-star) class. Weighted metrics hide this, so macro F1 and per-class recall should be reported.
2. Removing NLTK stop words deletes sentiment-bearing words such as "not", "no" and "but". Keeping them and adding bigrams improves every metric.
3. Class weighting trades a few points of accuracy for much better neutral detection.

## Dataset

- File: `spotifyappreviews.csv` (also zipped in `archive (3).zip`)
- Source: Kaggle, Spotify app reviews (Google Play), 1 Jan – 9 Jul 2022
- Columns: `Time_submitted`, `Review`, `Rating` (1–5), `Total_thumbsup`
- Label mapping: 1–2★ → Negative (40.2%), 3★ → Neutral (11.2%), 4–5★ → Positive (48.6%)

## Repository contents

| File | Description |
|---|---|
| `SPOTIFY_APP_REVIEW_CLASSIFICATION_USING_MACHINE_LEARNING.ipynb` | Original Colab notebook |
| `spotify_app_review_classification_using_machine_learning.py` | Script export of the notebook |
| `spotifyappreviews.csv` | Dataset |
| `archive (3).zip` | Zipped copy of the dataset |

## Methods

**Baseline:** lowercase → NLTK tokenize → remove English stop words → `TfidfVectorizer()` → 7 classifiers: Multinomial NB, Logistic Regression, SVM (RBF), Random Forest, KNN, Gradient Boosting, MLP.

**Proposed:** lowercase → tokenize (keep all words) → `TfidfVectorizer(ngram_range=(1,2), min_df=2, sublinear_tf=True)` → Logistic Regression / Linear SVM / Complement NB, optionally `class_weight="balanced"`.

Split: 80/20 with `random_state=42`.

## Baseline results (test set)

| Classifier | Test accuracy | Weighted F1 | Macro F1 |
|---|---|---|---|
| Multinomial NB | 77.00% | 72.48% | 54.36% |
| Logistic Regression | 78.72% | 75.13% | 58.37% |
| Random Forest | 77.16% | 72.73% | 54.66% |
| K-Nearest Neighbors | 70.31% | 66.55% | 50.87% |
| Gradient Boosting | 74.27% | 70.17% | 53.27% |
| MLP (ANN) | 74.11% | 73.10% | 59.64% |
| SVM (RBF) | 79.02% | 74.58% | 56.40% |

## Getting started

```bash
pip install pandas numpy scikit-learn nltk matplotlib seaborn
python -c "import nltk; nltk.download('punkt'); nltk.download('punkt_tab'); nltk.download('stopwords')"
```

In the `.py` file, change the dataset path from `/content/sample_data/spotifyappreviews.csv` to your local path, then run:

```bash
python spotify_app_review_classification_using_machine_learning.py
```

Note: the SVM and MLP models can take a long time to train on CPU.

### Quick start with the improved model

```python
import pandas as pd
from sklearn.pipeline import make_pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

df = pd.read_csv("spotifyappreviews.csv")
y = df["Rating"].map(lambda r: "positive" if r > 3 else ("negative" if r < 3 else "neutral"))

model = make_pipeline(
    TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True),
    LogisticRegression(max_iter=2000, class_weight="balanced"),
)
model.fit(df["Review"], y)
print(model.predict(["Love the playlists!", "Great app but it keeps crashing lately", "Worst update ever"]))
```

## Known issues in the original code

- **Metrics:** only weighted precision/recall/F1 are reported, which hides near-zero neutral recall. Add macro F1 and `classification_report`.
- **Sentiment pie chart:** the best model is retrained on all data and predicts the same data, so the distribution is optimistic. Use held-out or cross-validated predictions.
- **Recommendation bar chart:** it uses `review_sentiment_counts[2]` (neutral) for "Recommended"; it should be `[1]` (positive).
- **Stop words:** removing NLTK stop words drops negations and "but", which lowers performance.

## Future work

- Fine-tune transformer models (e.g., BERT)
- Aspect-based sentiment (ads, shuffle, playback, podcasts)
- Manually annotate a sample to measure rating-label noise
- Track sentiment over time to flag problematic app updates

## Tech stack

Python · pandas · NumPy · scikit-learn · NLTK · Matplotlib · seaborn
