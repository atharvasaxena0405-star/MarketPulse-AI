"""MarketPulse AI - experimental Gradio demo. Run:  python app.py
Uses the SAME feature code (marketpulse.py) and the model saved by MarketPulse_AI.ipynb (models/marketpulse_model.joblib).
This is a prototype and is not financial advice."""
import os
import joblib
import pandas as pd
import gradio as gr
import marketpulse as mp

HERE = os.path.dirname(os.path.abspath(__file__))
BUNDLE = joblib.load(os.path.join(HERE, "models", "marketpulse_model.joblib"))
MODEL, META = BUNDLE["model"], BUNDLE["meta"]
COLS = META["feature_cols"]

LIMITS = f"""
### ⚠️ Read this first - experimental historical model, NOT a trading signal
* This is a **prototype and is not financial advice.** Do not trade or invest based on it.
* It was trained on Reddit world-news headlines from **2008-2014** and tested on 2015-2016. On those unseen test days it was right **{META['test_accuracy']:.1%}** of the time, while a "always predict UP" guess was right **{META['always_up_test_accuracy']:.1%}**. **It did not beat the simple baseline.**
* The score shown is the model's raw output for "market rose or stayed flat", **not a calibrated or guaranteed probability** and not a measure of confidence.
* It only reads headline *tone* (VADER sentiment). It knows nothing about prices, earnings or events, and a handful of headlines you type is very different from the 25 daily headlines it was trained on.
"""

EXAMPLE = """Fed signals interest rates will stay low as economy recovers
Global stock markets tumble as banking fears spread
Unemployment falls to lowest level in seven years
Suicide bomber kills 15 at market in capital
Scientists announce breakthrough in cancer treatment"""


def analyze(text):
    heads = [mp.clean_headline(h) or h.strip() for h in (text or "").splitlines() if h.strip()]
    if not heads:
        return "Please enter at least one headline (one per line).", {}, "", LIMITS
    sc = mp.score_headlines(heads)
    feats = mp.daily_features_from_headlines(heads)
    p_up = float(MODEL.predict_proba(pd.DataFrame([feats])[COLS])[0, 1])
    label = "UP (market rose or stayed flat)" if p_up >= 0.5 else "DOWN (market fell)"
    fin = [h for h in heads if mp.is_financial(h)]
    tone = "negative" if feats["all_mean"] <= mp.NEG_T else "positive" if feats["all_mean"] >= mp.POS_T else "neutral"
    summary = (f"**Headlines analysed:** {len(heads)}  \n**Overall tone:** {tone} (mean compound {feats['all_mean']:+.2f})  \n"
               f"**Negative / positive headlines:** {feats['all_pct_neg']:.0%} / {feats['all_pct_pos']:.0%}  \n"
               f"**Most negative:** {sc.compound.min():+.2f} · **Most positive:** {sc.compound.max():+.2f}  \n"
               f"**Financial headlines detected:** {len(fin)} of {len(heads)}")
    if len(heads) < 10:
        summary += f"  \n_Note: you gave {len(heads)} headlines; the model was trained on days with ~25._"
    table = "| headline | VADER compound | financial? |\n|---|---|---|\n" + "\n".join(
        f"| {h[:90]} | {c:+.2f} | {'yes' if mp.is_financial(h) else ''} |" for h, c in zip(heads, sc.compound))
    verdict = f"## Experimental model output: {label}"
    return verdict + "\n\n" + summary, {"UP (raw model score)": p_up, "DOWN (raw model score)": 1 - p_up}, table, LIMITS


with gr.Blocks(title="MarketPulse AI") as demo:
    gr.Markdown("# 📈 MarketPulse AI\n**Can news sentiment predict the Dow Jones?** An experimental historical model - *not a live trading signal.*")
    with gr.Row():
        with gr.Column():
            box = gr.Textbox(label="Enter news headlines (one per line)", lines=9, value=EXAMPLE)
            btn = gr.Button("Analyze News", variant="primary")
        with gr.Column():
            out_md = gr.Markdown()
            out_lbl = gr.Label(label="Raw model score (NOT a calibrated probability)")
    with gr.Accordion("Per-headline sentiment", open=False):
        out_tbl = gr.Markdown()
    out_lim = gr.Markdown(LIMITS)
    btn.click(analyze, box, [out_md, out_lbl, out_tbl, out_lim])

if __name__ == "__main__":
    demo.launch()
