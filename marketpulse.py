"""MarketPulse AI - shared helpers (data cleaning, VADER sentiment, financial-news filter, features).

Used by both the notebook (MarketPulse_AI.ipynb) and the Gradio demo (app.py)
so the demo uses exactly the same feature code as the experiments.
"""
import re
import numpy as np
import pandas as pd
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer

HEADLINE_COLS = [f"Top{i}" for i in range(1, 26)]
TRAIN_END = "2014-12-31"   # training: 2008-08-08 .. 2014-12-31
TEST_START = "2015-01-01"  # test:     2015-01-01 .. 2016-07-01
VAL_START = "2014-01-01"   # validation (inside training): 2014 only

# --------------------------------------------------------------------------- #
# 1. Cleaning
# --------------------------------------------------------------------------- #
def clean_headline(text):
    """The CSV stores headlines as Python byte-string reprs, e.g.  b'Some headline'  or  b"It's ...".
    Strip that prefix and the wrapping quotes, then tidy whitespace."""
    if not isinstance(text, str):
        return ""
    t = text.strip()
    if len(t) >= 3 and t[0] == "b" and t[1] in "'\"" and t[-1] == t[1]:
        t = t[2:-1]
    t = t.replace("\\'", "'").replace('\\"', '"').replace("\\n", " ")
    return re.sub(r"\s+", " ", t).strip()


def load_news(path):
    """Load CSV, parse dates, sort chronologically, clean all headline cells.
    Missing headlines (a handful of NaN cells) become '' and are skipped later."""
    df = pd.read_csv(path)
    df["Date"] = pd.to_datetime(df["Date"])
    df = df.sort_values("Date").reset_index(drop=True)   # chronological, NEVER shuffled
    for c in HEADLINE_COLS:
        df[c] = df[c].map(clean_headline)
    return df


def headlines_of(row):
    return [row[c] for c in HEADLINE_COLS if row[c]]

# --------------------------------------------------------------------------- #
# 2. VADER sentiment
# --------------------------------------------------------------------------- #
_sia = None
def get_vader():
    global _sia
    if _sia is None:
        try:
            nltk.data.find("sentiment/vader_lexicon.zip")
        except LookupError:
            nltk.download("vader_lexicon", quiet=True)
        _sia = SentimentIntensityAnalyzer()
    return _sia


def score_headlines(headlines):
    """Return DataFrame with VADER neg/neu/pos/compound for each headline."""
    sia = get_vader()
    rows = [sia.polarity_scores(h) for h in headlines]
    return pd.DataFrame(rows, columns=["neg", "neu", "pos", "compound"])

# --------------------------------------------------------------------------- #
# 3. Financial-news keyword classifier (our original contribution)
# --------------------------------------------------------------------------- #
# Word-boundary regexes. Deliberately transparent so students can explain every rule.
FIN_CATEGORIES = {
    "fed_rates": r"\b(federal reserve|the fed|fed rate|interest rates?|rate (cut|hike|rise)s?|central bank|ecb|bank of (england|japan|china)|bernanke|yellen|draghi|monetary policy|quantitative easing|bond yields?|treasur(y|ies))\b",
    "inflation_jobs": r"\b(inflation|deflation|unemployment|jobless|jobs report|payrolls?|wages?|cost of living|consumer prices?|cpi)\b",
    "growth": r"\b(gdp|recession|economic growth|economy|economic|slowdown|downturn|stimulus|austerity|recovery|depression)\b",
    "banks_markets": r"\b(banks?|banking|stock markets?|stocks?|shares|wall street|dow jones|nasdaq|nikkei|ftse|s&p|hedge funds?|investors?|bailout|bankrupt(cy)?|debt|default|credit|lehman|financial|finance|currency|euro|dollar|yuan|sterling|forex|bitcoin|imf|world bank|market|markets)\b",
    "earnings_corp": r"\b(earnings|profits?|revenues?|quarterly|ipo|merger|takeover|acquisition|acquires?|shareholders?|ceo|corporate|corporation|company|companies|bonus(es)?|oil prices?|crude|opec|gold prices?)\b",
    "trade_policy": r"\b(trade|tariffs?|sanctions?|exports?|imports?|embargo|subsid(y|ies)|taxes|tax|taxation|budget|deficit|fiscal|wto|trade deal|currency war)\b",
}
_FIN_RE = {k: re.compile(v, re.I) for k, v in FIN_CATEGORIES.items()}


def fin_categories(headline):
    """Which financial categories (if any) a headline matches."""
    return [k for k, rx in _FIN_RE.items() if rx.search(headline)]


def is_financial(headline):
    return any(rx.search(headline) for rx in _FIN_RE.values())

# --------------------------------------------------------------------------- #
# 4. Daily feature engineering
# --------------------------------------------------------------------------- #
POS_T, NEG_T = 0.05, -0.05   # standard VADER thresholds on the compound score
AGG_NAMES = ["mean", "median", "std", "min", "max", "pct_neg", "pct_pos", "mean_pos", "mean_neg", "mean_neu"]


def aggregate(scores):
    """Turn per-headline VADER scores into one row of daily features (dict, no prefix).
    Empty input -> neutral zeros (documented imputation)."""
    if scores is None or len(scores) == 0:
        return {k: 0.0 for k in AGG_NAMES}
    c = scores["compound"]
    return {
        "mean": c.mean(), "median": c.median(),
        "std": c.std(ddof=0),
        "min": c.min(), "max": c.max(),
        "pct_neg": (c <= NEG_T).mean(), "pct_pos": (c >= POS_T).mean(),
        "mean_pos": scores["pos"].mean(), "mean_neg": scores["neg"].mean(), "mean_neu": scores["neu"].mean(),
    }


def daily_features_from_headlines(headlines):
    """Features for ONE day's list of headlines (used by the demo and by the dataset builder)."""
    sc = score_headlines(headlines) if len(headlines) else pd.DataFrame(columns=["neg", "neu", "pos", "compound"])
    fin_mask = np.array([is_financial(h) for h in headlines], dtype=bool) if len(headlines) else np.array([], dtype=bool)
    out = {f"all_{k}": v for k, v in aggregate(sc).items()}
    out.update({f"fin_{k}": v for k, v in aggregate(sc[fin_mask] if len(sc) else sc).items()})
    out["fin_count"] = float(fin_mask.sum())
    out["fin_frac"] = float(fin_mask.mean()) if len(fin_mask) else 0.0
    out["n_headlines"] = float(len(headlines))
    return out


def build_feature_table(df):
    """One row per trading day; keeps Date and Label. Pure per-day function => no cross-day leakage."""
    feats = [daily_features_from_headlines(headlines_of(r)) for _, r in df.iterrows()]
    out = pd.concat([df[["Date", "Label"]].reset_index(drop=True), pd.DataFrame(feats)], axis=1)
    return out


def add_market_history(ft):
    """Previous-day direction features. Label(D-1) is known before D's open, so these are legal inputs."""
    ft = ft.copy()
    ft["prev_dir"] = ft["Label"].shift(1)
    ft["prev_up_rate5"] = ft["Label"].shift(1).rolling(5, min_periods=1).mean()
    return ft


def lag_news(ft, cols):
    """Strict pre-market variant: use YESTERDAY's headline features to predict TODAY's label."""
    ft = ft.copy()
    ft[cols] = ft[cols].shift(1)
    return ft


def feature_columns(group):
    """group in {'all','fin','both'}"""
    all_cols = [f"all_{k}" for k in AGG_NAMES]
    fin_cols = [f"fin_{k}" for k in AGG_NAMES] + ["fin_count", "fin_frac"]
    return {"all": all_cols, "fin": fin_cols, "both": all_cols + fin_cols}[group]
