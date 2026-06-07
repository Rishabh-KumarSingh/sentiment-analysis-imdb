# 🎬 Sentiment Analysis — IMDB Movie Reviews

**Pinnacle Labs · Data Science Internship · Project 2**

Binary sentiment classification (Positive / Negative) on 50,000 IMDB movie reviews using TF-IDF vectorization and LinearSVC — achieving **~92.8% accuracy**.

---

## Project Structure

```
sentiment_analysis/
├── data/
│   └── IMDB Dataset.csv        ← place dataset here (see below)
├── models/
│   ├── best_model.pkl          ← saved after training
│   ├── tfidf_vectorizer.pkl    ← saved after training
│   └── metrics.json            ← all model metrics
├── notebooks/
│   └── EDA.ipynb               ← exploratory data analysis
├── src/
│   ├── preprocess.py           ← text cleaning pipeline
│   ├── train.py                ← training + model comparison
│   └── evaluate.py             ← evaluation & inference
├── app.py                      ← Streamlit web app
├── requirements.txt
└── README.md
```

---

## Dataset

**IMDB Dataset of 50K Movie Reviews** (Kaggle)
- 50,000 reviews — 25,000 positive · 25,000 negative
- Perfectly balanced binary classification task
- Download: https://www.kaggle.com/datasets/lakshmi25npathi/imdb-dataset-of-50k-movie-reviews
- Place as: `data/IMDB Dataset.csv`

---

## Setup

```bash
# 1. Clone the repo
git clone https://github.com/YOUR_USERNAME/sentiment-analysis.git
cd sentiment-analysis

# 2. Create virtual environment
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Download NLTK data (auto-runs on first import, or manually)
python -c "import nltk; nltk.download('stopwords'); nltk.download('wordnet')"

# 5. Place IMDB Dataset.csv in data/
```

---

## Usage

### Step 1 — Exploratory Data Analysis
```bash
jupyter notebook notebooks/EDA.ipynb
```

### Step 2 — Train Models
```bash
python src/train.py
```
Trains 4 models, compares them, saves the best one to `models/`.

### Step 3 — Evaluate / Predict
```bash
# Show stored training metrics
python src/evaluate.py

# Evaluate on a custom CSV
python src/evaluate.py --csv path/to/test.csv

# Predict a single review
python src/evaluate.py --text "This movie was absolutely brilliant!"
```

### Step 4 — Launch Web App
```bash
streamlit run app.py
```

---

## ML Pipeline

### Text Preprocessing (`src/preprocess.py`)
| Step | Action |
|------|--------|
| HTML strip | BeautifulSoup removes `<br/>`, tags |
| Lowercase | Normalize case |
| Contractions | "won't" → "will not" |
| Special chars | Remove punctuation & numbers |
| Stopwords | NLTK — **negations preserved** (not, never, isn't …) |
| Lemmatization | WordNetLemmatizer |

### TF-IDF Vectorizer
| Parameter | Value |
|-----------|-------|
| `max_features` | 100,000 |
| `ngram_range` | (1, 2) — unigrams + bigrams |
| `sublinear_tf` | True — log normalization |
| `min_df` | 2 |
| `max_df` | 0.95 |

### Models Compared

| Model | Accuracy | ROC-AUC |
|-------|----------|---------|
| **LinearSVC (Calibrated)** | **~92.8%** | **~0.979** |
| SGD Classifier | ~91.8% | ~0.975 |
| Logistic Regression | ~91.2% | ~0.972 |
| Multinomial Naive Bayes | ~88.4% | ~0.952 |



---

## Streamlit App Features

| Tab | Features |
|-----|----------|
| 🔍 Single Review | Text input · Confidence gauge · Top contributing words chart |
| 📂 Batch CSV | Upload CSV · Bulk predict · Sentiment distribution pie · Download results |
| 📊 Model Dashboard | Accuracy/AUC comparison · Confusion matrix · Classification report |

---

## Results

```
Best Model : LinearSVC (Calibrated)
Accuracy   : ~92.8%
ROC-AUC    : ~0.979

              precision    recall  f1-score   support
    Negative       0.93      0.93      0.93      5000
    Positive       0.93      0.93      0.93      5000
    accuracy                           0.93     10000
```

---

## Tech Stack

![Python](https://img.shields.io/badge/Python-3.10-blue)
![Scikit-learn](https://img.shields.io/badge/scikit--learn-1.5-orange)
![Streamlit](https://img.shields.io/badge/Streamlit-1.35-red)
![NLTK](https://img.shields.io/badge/NLTK-3.8-green)
![Plotly](https://img.shields.io/badge/Plotly-5.22-purple)

- **Python 3.10**
- **scikit-learn** — TF-IDF, LinearSVC, model evaluation
- **NLTK** — stopwords, WordNetLemmatizer
- **BeautifulSoup4** — HTML stripping
- **Streamlit** — web app
- **Plotly** — interactive charts
- **WordCloud** — word frequency visualization

---

## Author

**Your Name**
Data Science Intern — Pinnacle Labs
- GitHub: [https://github.com/Rishabh-KumarSingh]
- LinkedIn: [www.linkedin.com/in/rishabh-kumar-singh-b5354a251]

---


