# MarketPulse AI — 2-Minute Demo Script
Screen-record the Gradio app (`python app.py`) and the notebook's chart `figures/03_model_accuracy_comparison.png`. Speak over it; no editing needed. Roughly 280 spoken words in total.

## 0:00–0:15 · The problem (show: title slide or the app header)
"Everyone's heard that AI can predict the stock market from the news. We're [names]. We built MarketPulse AI to *test* that claim honestly: can the sentiment of the day's top world headlines predict whether the Dow Jones goes up or down?"

## 0:15–1:15 · Working demo (show: the Gradio app)
1. *(0:15)* "Here's the app. I paste in five headlines — a Fed interest-rate story, a stock-market sell-off, an unemployment report, a bombing, a science breakthrough." Click **Analyze News**.
2. *(0:30)* "It scores every headline with VADER, a pretrained sentiment model, and summarises the day: overall tone, percent negative, and how many headlines are financial."
3. *(0:42)* "Then the Random Forest outputs a direction — here it says [read the screen] — with a raw model score. We label this score *not* a calibrated probability."
4. *(0:52)* "Notice the warning: experimental historical model, not a trading signal. And it tells you the key fact — on data it never saw, this model did **not** beat a simple guess."
5. *(1:02)* "It's also honest about its own blind spots. Try 'Stocks surge as Fed cuts rates' — VADER reads the word 'cuts' as negative. Sentiment is not the same as meaning." *(Optional: type that headline; read the score.)*

## 1:15–1:45 · Technical approach and baselines (show: accuracy chart, then financial-vs-general chart)
"We trained on 2008 to 2014 and tested on 2015 to 2016 — split by date, never shuffled — 378 unseen days. Baseline one, always predict UP, scores **50.8%**. Baseline two, copy yesterday, **45.8%**. Our Logistic Regression scored **48.7%** and the Random Forest **46.0%**. So: no edge. Our original question was whether *financial* headlines work better than general news. We built a keyword filter, tested 22 combinations — financial news was no better, and none beat the baseline significantly."

## 1:45–2:00 · Limitations and next steps (show: README limitations)
"Limits: one market, Reddit headlines instead of timestamped news, a crude keyword filter, and only 378 test days. Next, we'd try FinBERT and real pre-market news. The takeaway: always compare to a baseline — here, the honest answer to 'can news sentiment predict the market?' is 'not from this data.' This is a prototype and not financial advice."

---
**Presenter tips:** Say the numbers exactly as printed in the notebook/README (50.8%, 45.8%, 48.7%, 46.0%). If a judge asks "isn't 'no result' a failed project?" — the brief says to report the honest answer "even if it's 'not really'", and we tested for look-ahead bias, confidence intervals and multiple comparisons so the null result is trustworthy.
