# ⚾ MLB NRFI Predictions Dashboard (Fireball + Random Forest + Confidence + Cumulative Chart)

This project builds a daily updated Machine Learning pipeline and dashboard  
to predict **NRFI (No Run First Inning)** bets for Major League Baseball games.

It combines:
- Team batting and pitching statistics
- Custom Fireball Confidence Ratings
- Random Forest model trained on real outcomes
- Full Cumulative Win % tracking over the season

Built for automated daily refresh, GitHub update, and live Streamlit deployment!

---

## 📋 Project Structure

| File | Purpose |
|:-----|:--------|
| `get_scores_full.py` | Scrapes daily boxscores from ESPN |
| `build_stats.py` | Builds NRFI batting/pitching stats and predictions |
| `merge_stats.py` | Merges all game data |
| `build_nrfi_results.py` | Builds ground-truth actual NRFI results |
| `pipeline.py` | Runs full daily data refresh and GitHub push |
| `app.py` | Streamlit dashboard (live NRFI predictions and results) |
| `model_rf_real.pkl` | Frozen Random Forest model used for predictions |

---

## 🚀 How to Run Locally

1. Install Python packages:

```bash
pip install streamlit pandas beautifulsoup4 requests scikit-learn lightgbm joblib matplotlib
