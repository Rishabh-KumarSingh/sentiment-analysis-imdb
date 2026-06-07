# Data Directory

Place the IMDB dataset file here before running training.

**Required file:** `IMDB Dataset.csv`

**Download from:** https://www.kaggle.com/datasets/lakshmi25npathi/imdb-dataset-of-50k-movie-reviews

**Expected columns:**
| Column | Description |
|--------|-------------|
| `review` | Raw movie review text |
| `sentiment` | Label — `positive` or `negative` |

**Dataset stats:**
- 50,000 total reviews
- 25,000 positive · 25,000 negative (perfectly balanced)
- Train split: 40,000 · Test split: 10,000 (80/20)

Once placed, run:
```bash
python src/train.py
```
