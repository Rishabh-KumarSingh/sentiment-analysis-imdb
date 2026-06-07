"""
preprocess.py
-------------
Text cleaning pipeline for IMDB Sentiment Analysis.

Steps:
  1. HTML tag removal (BeautifulSoup)
  2. Lowercase conversion
  3. Special character & punctuation removal
  4. Stopword removal (NLTK) — negations preserved
  5. Lemmatization (WordNetLemmatizer)
"""

import re
import nltk
from bs4 import BeautifulSoup
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# ── Download required NLTK data (runs once) ───────────────────────────────
def download_nltk_data():
    resources = ["stopwords", "wordnet", "omw-1.4", "punkt"]
    for r in resources:
        try:
            nltk.data.find(f"corpora/{r}")
        except LookupError:
            nltk.download(r, quiet=True)

download_nltk_data()

# ── Negation words — must NOT be removed as stopwords ─────────────────────
NEGATION_WORDS = {
    "no", "not", "nor", "neither", "never", "nobody", "nothing",
    "nowhere", "isn't", "wasn't", "aren't", "weren't", "haven't",
    "hasn't", "hadn't", "won't", "wouldn't", "don't", "doesn't",
    "didn't", "can't", "couldn't", "shouldn't", "mustn't",
}

# Build stopword set keeping negations intact
_raw_stops   = set(stopwords.words("english"))
STOP_WORDS   = _raw_stops - NEGATION_WORDS

lemmatizer   = WordNetLemmatizer()


# ── Core cleaning function ─────────────────────────────────────────────────
def clean_text(text: str) -> str:
    """
    Full cleaning pipeline on a single review string.

    Parameters
    ----------
    text : str
        Raw review text (may contain HTML, special chars, etc.)

    Returns
    -------
    str
        Cleaned, lemmatized string ready for TF-IDF vectorization.
    """
    # 1. Remove HTML tags
    text = BeautifulSoup(text, "html.parser").get_text()

    # 2. Lowercase
    text = text.lower()

    # 3. Replace contractions (basic)
    contraction_map = {
        "won't": "will not", "can't": "cannot", "n't": " not",
        "'re": " are", "'s": " is", "'d": " would",
        "'ll": " will", "'ve": " have", "'m": " am",
    }
    for contraction, expansion in contraction_map.items():
        text = text.replace(contraction, expansion)

    # 4. Remove special characters — keep only alphabetic + spaces
    text = re.sub(r"[^a-z\s]", " ", text)

    # 5. Collapse multiple spaces
    text = re.sub(r"\s+", " ", text).strip()

    # 6. Tokenize
    tokens = text.split()

    # 7. Remove stopwords (negations preserved)
    tokens = [t for t in tokens if t not in STOP_WORDS]

    # 8. Lemmatize
    tokens = [lemmatizer.lemmatize(t) for t in tokens]

    # 9. Drop very short tokens
    tokens = [t for t in tokens if len(t) > 2]

    return " ".join(tokens)


def preprocess_series(series):
    """
    Apply clean_text to a pandas Series with a progress bar.

    Parameters
    ----------
    series : pd.Series
        Series of raw review strings.

    Returns
    -------
    pd.Series
        Cleaned series.
    """
    try:
        from tqdm import tqdm
        tqdm.pandas(desc="Cleaning reviews")
        return series.progress_apply(clean_text)
    except ImportError:
        print("Cleaning reviews...")
        return series.apply(clean_text)


# ── Quick smoke-test ───────────────────────────────────────────────────────
if __name__ == "__main__":
    sample = (
        "<br /><b>This movie</b> was absolutely NOT what I expected! "
        "The acting wasn't great, but the storyline couldn't be better. "
        "10/10 would recommend :)"
    )
    print("RAW :", sample)
    print("CLEAN:", clean_text(sample))
