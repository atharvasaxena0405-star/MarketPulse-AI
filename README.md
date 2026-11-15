# 📈 MarketPulse AI — Can News Sentiment Really Predict the Stock Market?

**Veritas AI International High School Hackathon 2026 · Track: AI + Finance · Project 4.2.4 "News Headlines and the Stock Market"**
**Team:** Atharva Saxena, West Forsyth High School, atharvasaxena0405@gmail.com

> **This project is a prototype and is not financial advice.**

## TL;DR (the honest answer)
We tested whether the sentiment of each day's top-25 world-news headlines predicts whether the Dow Jones Industrial Average (DJIA) goes up or down. **On 378 unseen test days (2015-2016) it does not** — no model beat the simple "always predict UP" guess. Financial-only headlines did not do better than general headlines either. We report that plainly, with confidence intervals, because that is what the data shows.

| Test accuracy, 2015-01-02 → 2016-07-01 (**378 days**) | Accuracy | 95% CI |
|---|---|---|
| Baseline A: always UP | **50.8%** | 45.8 – 55.8% |
| Baseline B: same as previous day | 45.8% | 40.8 – 50.8% |
| Model 1: Logistic Regression (VADER sentiment, all headlines) | 48.7% | 43.7 – 53.7% |
| Model 2: Random Forest (same features) — *preferred on validation* | 46.0% | 41.1 – 51.1% |

All numbers come from running [`MarketPulse_AI.ipynb`](MarketPulse_AI.ipynb) (saved with outputs) and are also stored in [`results.json`](results.json).

## Problem statement & target user
Headlines like "markets fear…" make it feel obvious that news moves stocks, and finance influencers sell "sentiment signals". **Target user:** a high-school student or first-time investor who has seen a claim that "AI can predict the market from the news" and wants to know whether it holds up. MarketPulse AI is a *reality-check tool*: it scores real historical headlines, makes a prediction, and shows — against baselines, on data the model never saw — how well that works.

## Dataset
**Daily News for Stock Market Prediction** (`Combined_News_DJIA.csv`) — [Kaggle, Aaron7sun](https://www.kaggle.com/datasets/aaron7sun/stocknews). 1,989 trading days (2008-08-08 → 2016-07-01); top-25 Reddit r/WorldNews headlines per day; `Label` = 1 if the DJIA closed up or unchanged, 0 if it fell. The notebook downloads a copy from the [Hugging Face mirror `abhimlv/ESMPT-Dataset`](https://huggingface.co/datasets/abhimlv/ESMPT-Dataset) (we verified it has the documented 1,989 rows and date range); if that link ever fails, download from Kaggle and place it in `data/`. Check the dataset's licence on Kaggle before reusing it.

## Methodology
1. **Cleaning:** parse dates, sort chronologically, strip the `b'…'` byte-string wrapper around every headline, handle 7 missing headline cells (0.014%) as empty.
2. **Split (chronological, never shuffled):** train 2008–2013 · validation 2014 · final fit 2008–2014 · **test 2015–2016**. An assertion in the notebook checks there is no date overlap.
3. **Sentiment:** every headline scored with **VADER** (NLTK) → `neg/neu/pos/compound`; aggregated per day into 10 features (mean, median, std, most-negative, most-positive, % negative, % positive, mean pos/neg/neu).
4. **Models:** Logistic Regression (simple, interpretable) and Random Forest (non-linear). Hyper-parameters and the preferred model are chosen **on the 2014 validation year only**; then the model is refit on 2008–2014 and tested once.
5. **Baselines:** always UP; previous trading day's direction.
6. **Statistics:** Wilson 95% confidence intervals and exact paired tests (McNemar) vs. Always-UP, because 378 days is a small sample (±5 points).
7. **Prediction time & look-ahead bias:** day-*D* headlines are Reddit posts from the same day the market move is measured, so they may contain post-move reactions. We therefore run the brief's **same-day** setup *and* a **strict pre-market** variant (day *D−1* headlines → day *D* direction).

## ⭐ Original contribution: financial vs. general news
A transparent keyword/regex classifier ([`marketpulse.py`](marketpulse.py)) tags headlines in six categories (Fed & rates, inflation & jobs, growth, banking & markets, earnings, trade & policy). It tagged 4,831 of 49,718 headlines (**9.7%**); 194 of 1,989 days had none. We trained comparable models on (A) all-headline sentiment and (B) financial-headline sentiment, and also asked: does negative news carry more signal, are high-sentiment days easier, does yesterday's direction help?

| Question | What we measured | Answer |
|---|---|---|
| Financial vs. general news? | 22 experiments (feature set × model × timing), test acc. 46.0–51.1% vs. Always UP 50.8% | **No consistent edge.** Best financial-only model 51.1% (LogReg), 0.3 points (1 day) above Always UP — well inside noise |
| Robustness check | Expanding window, one model per year 2010–2016 (1,636 days) | All-news 53.1%, financial-only 53.1%, market-history-only 53.2%, Always UP 53.9%. Paired test financial vs. all-news: p = 1.0. Strict pre-market financial-only: 50.1%, *significantly worse* than Always UP (p = 0.0005) |
| Does negative news carry more information? | Negative-tone vs. positive-tone features only (test) | Negative 48.1%, positive 51.1% — both inside the noise band; no evidence negative news is more informative |
| High-sentiment days easier? | Top-25% \|daily mean\| days (n = 61) vs. rest | Not more accurate: LR 47.5% vs. 48.9%, RF 44.3% vs. 46.4% (CIs ≈ ±12 points) |
| Does adding previous direction help? | With/without `prev_dir` features | Mixed and tiny: all-news LR 48.7% → 50.5%, RF 46.0% → 50.8%; market-history alone 49.2% / 48.9% |
| Any real signal anywhere? | Training-period correlations (22 features) | Best: % negative headlines vs. UP, r = −0.077 (p = 0.002; Bonferroni limit 0.0023). Explains <1% of variance and did not carry into the test period |

Of the 22 grid experiments, **5 scored above Always UP, none significantly**; one was significantly *worse*. With 22 comparisons you expect about one "p < 0.05" by luck, so a single lucky win would not count as evidence.

## Where it fails
* The preferred Random Forest predicts **UP on 79.9% of test days** (actual UP: 50.8%). Its output only ranges 0.40–0.69, i.e. it leans on the training base rate (54% UP) with small adjustments. Confusion matrix: 29 true DOWN caught, **157 DOWN days wrongly called UP**, 47 UP days called DOWN, 145 UP correct. Precision(UP) 48.0%, recall(UP) 75.5%, F1(UP) 58.7%.
* The four most confident wrong predictions (2016-01-13, 2015-08-21, 2016-02-01, 2015-03-25) were all days with slightly negative average sentiment (−0.07 to −0.13) driven by terror/geopolitics headlines, plus a few positive science/energy/environment stories; the model predicted UP (score 0.66) and the Dow fell. The only clearly market-related positive headline (Alphabet passing Apple as most valuable company, 2016-02-01) was followed by a DOWN day. See the notebook, Part 5.
* **Why headlines struggle:** public news is priced in within seconds; VADER measures tone, not economic impact (it scores "Stocks surge as Fed cuts rates" **negative** because of "cuts", and "Unemployment falls to lowest level" at −0.67); world news is mostly not about markets; keyword matching is crude ("market" in a farmers' market); 378 days is a small test sample; 2008–2014 recovery may not resemble 2015–16.

## How to run
**Google Colab (easiest):** upload `MarketPulse_AI.ipynb` → *Runtime → Run all*. It installs packages, writes `marketpulse.py`, downloads the data, and runs everything in a few minutes (CPU only).

**Locally:**
```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
jupyter nbconvert --to notebook --execute --inplace MarketPulse_AI.ipynb   # or open in Jupyter and Run All
python app.py                                                               # Gradio demo at http://127.0.0.1:7860
```
`app.py` needs `models/marketpulse_model.joblib`, which the notebook creates (included in this repo). If NLTK cannot download `vader_lexicon` on macOS because of an SSL certificate error, run `export SSL_CERT_FILE=$(python -c "import certifi;print(certifi.where())")` first (this is how we ran it).

## Files
| File | Purpose |
|---|---|
| `MarketPulse_AI.ipynb` | Full pipeline, executed, outputs and charts saved |
| `marketpulse.py` | Cleaning, VADER, financial-news filter, feature code (shared with the app) |
| `app.py` | Gradio demo: enter headlines → sentiment summary, raw model score, limitations |
| `results.json`, `figures/` | Every reported number and chart, written by the notebook |
| `HACKATHON_WRITEUP.md`, `DEMO_SCRIPT.md` | Submission write-up and 2-minute video script |
| `build_notebook.py` | Script that generates the notebook (for maintainers) |

## Limitations (summary)
Single market (DJIA), single news source (Reddit r/WorldNews top-25, which is crowd-voted, not a newswire), one sentiment model (VADER; we did not run FinBERT), a keyword filter we spot-checked but did not hand-label for precision/recall, 378 test days, possible same-day look-ahead in the labels, and daily direction only (no magnitude, no trading costs). **Nothing here justifies trading on news sentiment; what it supports is a caution against believing such claims without a baseline.** The demo's score is a raw model output, not a calibrated probability.

## What we'd do next
Run FinBERT and compare with VADER; use timestamps (real pre-market news) instead of Reddit daily tops; predict returns/volatility rather than direction; test individual stocks via yfinance or FRED indicators.

## Credits & citations
* **Dataset:** Aaron7sun, *Daily News for Stock Market Prediction*, Kaggle — https://www.kaggle.com/datasets/aaron7sun/stocknews (Reddit r/WorldNews headlines; DJIA from Yahoo Finance). Download mirror: https://huggingface.co/datasets/abhimlv/ESMPT-Dataset
* **Sentiment model:** VADER — Hutto, C.J. & Gilbert, E.E. (2014), *VADER: A Parsimonious Rule-based Model for Sentiment Analysis of Social Media Text*, ICWSM — via NLTK (https://www.nltk.org)
* **Libraries:** pandas, NumPy, scikit-learn, SciPy, matplotlib, joblib, [Gradio](https://www.gradio.app), Jupyter/nbconvert
* **Hackathon brief:** Veritas AI International High School Hackathon 2026 Project Briefs, §4.2.4
* **AI tool:** Claude (Anthropic) via Claude Code, used as a coding and writing assistant. The team is responsible for understanding and explaining every part.

> **This project is a prototype and is not financial advice.**
