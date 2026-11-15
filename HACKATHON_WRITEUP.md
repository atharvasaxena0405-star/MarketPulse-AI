# MarketPulse AI — Can News Sentiment Really Predict the Stock Market?
**Track:** Finance
**Team:** _[Name 1 (school, email)], [Name 2 (school, email)], [Name 3 (school, email)]_

> This project is a prototype and is not financial advice.

## The problem
Claims that "AI can predict the stock market from the news" are everywhere, and first-time investors and students are the ones most likely to believe them. Most such claims are never checked against a simple baseline on data the model has not seen. We wanted to test one honestly: does the sentiment of each day's top world-news headlines tell you anything about whether the Dow Jones (DJIA) rises or falls?

## What we built
**In:** the day's news headlines (25 per day in the dataset; any number in the demo). **Out:** a sentiment summary, a predicted market direction (UP / DOWN) and a raw model score. A Gradio demo (`app.py`) lets a user paste headlines and click "Analyze News"; it is labelled an experimental historical model, not a trading signal. The target user is a student or new investor who wants a reality check on "news predicts stocks" claims.

## What we added
We started from suggested project 4.2.4 and added our own question: **do financial headlines predict better than general world news?** We built a transparent keyword classifier (six categories, e.g. Fed & rates, banking & markets, earnings) and trained comparable models on all-headline vs. financial-headline sentiment. We also tested a strict pre-market version (yesterday's headlines → today's move) to guard against look-ahead bias, checked negative news, high-sentiment days and yesterday's direction, and repeated the test for every year 2010–2016, with confidence intervals and significance tests.

## How it works
**Data:** *Daily News for Stock Market Prediction* (Kaggle): 1,989 trading days, 2008-08-08 → 2016-07-01, top-25 Reddit r/WorldNews headlines per day, label 1 = DJIA up/flat. Cleaned and sorted by date, then **split by date, never shuffled**: train 2008–2013, validation 2014, test 2015–2016 (378 days). **Model:** each headline gets a VADER sentiment score (pretrained); we aggregate to 10 daily features (mean, median, std, most negative/positive, % negative/positive, …). Logistic Regression and Random Forest are trained on these; settings and the preferred model are chosen on the 2014 validation year only, then refit on 2008–2014 and tested once.

## Results
Test accuracy on 378 unseen days (2015–2016), 95% CI about ±5 points:
**Always UP (baseline 1): 50.8% · Previous-day direction (baseline 2): 45.8% · Logistic Regression: 48.7% · Random Forest (preferred on validation): 46.0%.** The ML models did **not** beat the simple baseline. Financial-only features were no better than all-news features (best of 22 experiments: 51.1%, one day better than Always UP; 5 of 22 scored above Always UP, none significantly; one was significantly worse). A 2010–2016 yearly walk-forward check (1,636 days) agreed: all-news and financial-only 53.1% vs. 53.9% for Always UP.
**Where it fails:** the Random Forest predicted UP on 79.9% of days (actual 50.8%) — mostly echoing the base rate — wrongly calling 157 down days UP. Its most confident mistakes (e.g. 2016-01-13, 2015-08-21) were terror/geopolitics days with a few positive science stories. Why we think headlines fail: public news is already priced in; VADER reads tone, not economic impact (it scores "Stocks surge as Fed cuts rates" negative); most world news isn't about markets; small test sample; 2015–16 differs from the training years.
**What this would and wouldn't justify:** an investor should *not* trade on these results — the honest finding is "news sentiment as measured here adds no usable predictive information."

## What we'd do next
Replace VADER with FinBERT and timestamped pre-market news from a financial source; test individual stocks (yfinance) or economic indicators (FRED), and predict volatility rather than direction.

## Sources
* Dataset: Aaron7sun, *Daily News for Stock Market Prediction*, Kaggle — https://www.kaggle.com/datasets/aaron7sun/stocknews (mirror used: https://huggingface.co/datasets/abhimlv/ESMPT-Dataset)
* VADER (Hutto & Gilbert, 2014) via NLTK — https://www.nltk.org
* pandas, NumPy, scikit-learn, SciPy, matplotlib, joblib, Gradio
* Veritas AI International High School Hackathon 2026 Project Briefs, §4.2.4
* AI tool: Claude (Anthropic), via Claude Code, as coding/writing assistant
* Code: `MarketPulse_AI.ipynb`, `marketpulse.py`, `app.py` — _[link to Colab/GitHub]_ · Demo video: _[link]_
