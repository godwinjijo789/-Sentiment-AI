import hashlib
import io
import os
os.environ["USE_TF"] = "0"
import time
from datetime import datetime

import pandas as pd
import psutil
import streamlit as st

st.set_page_config(page_title="Sentiment AI Dashboard", page_icon="🧠", layout="wide")
MODEL_ID = os.getenv("SENTIMENT_MODEL_ID", "distilbert/distilbert-base-uncased-finetuned-sst-2-english")

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=Space+Grotesk:wght@500;600;700&display=swap');
html, body, [class*="css"] {font-family:'DM Sans',sans-serif;}
.stApp {
    --app-background: light-dark(#f8fafc,#0b1020);
    --body-color: light-dark(#1f2937,#f4f7ff);
    --header-background: light-dark(rgba(248,250,252,.78),rgba(11,16,32,.7));
    --sidebar-start: light-dark(#f1f5fb,#111831);
    --sidebar-end: light-dark(#e9eef7,#0c1225);
    --sidebar-border: light-dark(rgba(31,41,55,.10),rgba(255,255,255,.08));
    --hero-start: light-dark(rgba(79,70,229,.12),rgba(79,70,229,.32));
    --hero-end: light-dark(rgba(15,184,184,.08),rgba(15,184,184,.13));
    --hero-border: light-dark(rgba(31,41,55,.12),rgba(255,255,255,.12));
    --accent-color: light-dark(#0f766e,#8de8e1);
    --heading-color: light-dark(#172033,#ffffff);
    --muted-color: light-dark(#4b5563,#c1cce5);
    --card-background: light-dark(rgba(255,255,255,.82),rgba(255,255,255,.055));
    --card-border: light-dark(rgba(31,41,55,.12),rgba(255,255,255,.10));
    --label-color: light-dark(#475569,#aab6d2);
    --pill-background: light-dark(rgba(31,41,55,.07),rgba(255,255,255,.08));
    --pill-color: light-dark(#334155,#d9e4ff);
    --metric-background: light-dark(rgba(255,255,255,.72),rgba(255,255,255,.045));
    --footer-color: light-dark(#64748b,#7785a8);
    background:radial-gradient(circle at 85% 0%,light-dark(rgba(91,92,246,.09),rgba(91,92,246,.18)),transparent 28%),radial-gradient(circle at 0% 20%,light-dark(rgba(27,190,190,.07),rgba(27,190,190,.10)),transparent 25%),var(--app-background);
    color:var(--body-color);
}
[data-testid="stHeader"] {background:var(--header-background)}
[data-testid="stSidebar"] {background:linear-gradient(180deg,var(--sidebar-start),var(--sidebar-end));border-right:1px solid var(--sidebar-border)}
.hero {padding:28px 30px;border-radius:24px;background:linear-gradient(120deg,var(--hero-start),var(--hero-end));border:1px solid var(--hero-border);margin-bottom:22px;}
.eyebrow {color:var(--accent-color);font-size:.78rem;font-weight:700;text-transform:uppercase;letter-spacing:.16em}
.hero h1,.result-title {font-family:'Space Grotesk',sans-serif;color:var(--heading-color)}
.hero h1 {font-size:clamp(2rem,4vw,3.2rem);line-height:1.05;margin:10px 0}
.hero p,.muted,.result-detail {color:var(--muted-color)}
.metric-card {padding:18px 20px;border-radius:18px;min-height:112px;background:var(--card-background);border:1px solid var(--card-border)}
.metric-label {color:var(--label-color);font-size:.8rem;margin-bottom:8px}
.metric-value {color:var(--heading-color);font-family:'Space Grotesk',sans-serif;font-size:1.7rem;font-weight:700}
.metric-note {color:var(--accent-color);font-size:.76rem;margin-top:5px}
.section-title {font-family:'Space Grotesk',sans-serif;font-size:1.35rem;font-weight:700;color:var(--heading-color);margin:8px 0 4px}
.result-box {border-radius:20px;padding:22px;margin-top:14px;background:var(--card-background);border:1px solid var(--card-border)}
.result-title {margin:8px 0}
.result-detail {margin:0}
.result-score {font-size:2.5rem;margin:3px 0 8px;color:var(--heading-color)}
.positive {border-left:5px solid #35d39a}.negative {border-left:5px solid #ff6b85}
.small-pill {display:inline-block;padding:5px 10px;border-radius:999px;background:var(--pill-background);color:var(--pill-color);font-size:.75rem;border:1px solid var(--card-border)}
div.stButton>button[kind="primary"],button[kind="primaryFormSubmit"] {border:0;border-radius:12px;min-height:2.9rem;background:linear-gradient(90deg,#6558f5,#258cda);color:white;font-weight:700}
[data-testid="stMetric"] {background:var(--metric-background);padding:14px;border:1px solid var(--card-border);border-radius:16px}
.app-footer {text-align:center;color:var(--footer-color);font-size:.8rem;padding:6px 0 18px}
hr {border-color:var(--card-border)}
</style>
""", unsafe_allow_html=True)

@st.cache_resource(show_spinner=False)
def load_model(model_id):
    import torch
    from transformers import pipeline
    device = 0 if torch.cuda.is_available() else -1
    return pipeline("sentiment-analysis", model=model_id, device=device), ("GPU" if device == 0 else "CPU")

def ram_stats():
    m = psutil.virtual_memory()
    return m.percent, m.used / 1024**3, m.total / 1024**3

def percentage(count, total):
    return count / total * 100 if total else None

def format_percentage(value):
    return f"{value:.2f}%" if value is not None else "N/A"

if "history" not in st.session_state: st.session_state.history = []
if "last_result" not in st.session_state: st.session_state.last_result = None

with st.sidebar:
    st.markdown("## 🧠 Sentiment AI")
    st.markdown('<span class="small-pill">REVIEW INTELLIGENCE</span>', unsafe_allow_html=True)
    page = st.radio("NAVIGATION", ["Dashboard", "Batch data analysis", "About the model"])
    st.divider()
    st.caption("Model is loaded once and cached in RAM.")
    st.caption("Public model — HF token not required.")

st.markdown("""<div class="hero"><div class="eyebrow">AI-powered text analytics</div><h1>Understand the feeling<br>behind every review.</h1><p>Analyse text with a transformer-based sentiment model. View predictions, confidence scores, system RAM, and session insights.</p></div>""", unsafe_allow_html=True)

model, device_name, model_error = None, "Not loaded", None
with st.spinner("Loading sentiment model into memory… first run may take a little longer"):
    try: model, device_name = load_model(MODEL_ID)
    except Exception as e: model_error = str(e)
ram_pct, ram_used, ram_total = ram_stats()
history = st.session_state.history
pos_pct = (sum(x["label"] == "POSITIVE" for x in history) / len(history) * 100) if history else 0
cols = st.columns(4)
metrics = [("MODEL STATUS", "Ready" if model else "Needs setup", device_name), ("SYSTEM RAM USED", f"{ram_pct:.0f}%", f"{ram_used:.1f} / {ram_total:.1f} GB"), ("REVIEWS ANALYSED", str(len(history)), "This session"), ("POSITIVE SHARE", f"{pos_pct:.0f}%", "Of this session's analyses")]
for col, (label, value, note) in zip(cols, metrics):
    with col: st.markdown(f'<div class="metric-card"><div class="metric-label">{label}</div><div class="metric-value">{value}</div><div class="metric-note">{note}</div></div>', unsafe_allow_html=True)
if model_error:
    st.error("Could not load the model. Check the package installation and restart the app.")
    st.code(model_error)
st.write("")

if page == "Dashboard":
    left, right = st.columns([1.5, 1], gap="large")
    with left:
        st.markdown('<div class="section-title">✍️ Analyse a review</div><div class="muted">Type or paste a movie review, product review, comment, or short text.</div>', unsafe_allow_html=True)
        with st.form("analysis_form"):
            review = st.text_area("Text to analyse", placeholder="Example: The story was engaging, the acting was excellent, and I enjoyed every minute.", height=190, max_chars=5000)
            submitted = st.form_submit_button("✨ Analyse sentiment", type="primary", use_container_width=True)
        if submitted:
            if not review.strip(): st.warning("Please type some text first.")
            elif model is None: st.error("The model is not loaded. See the error above.")
            else:
                with st.spinner("Analysing text…"):
                    start = time.perf_counter()
                    pred = model(review.strip(), truncation=True, max_length=512)[0]
                    duration = time.perf_counter() - start
                item = {"time": datetime.now().strftime("%H:%M:%S"), "review": review.strip(), "label": pred["label"].upper(), "score": float(pred["score"]), "seconds": duration}
                st.session_state.history.insert(0, item)
                st.session_state.history = st.session_state.history[:100]
                st.session_state.last_result = item
                st.rerun()
        result = st.session_state.last_result
        if result:
            good = result["label"] == "POSITIVE"
            st.markdown(f"""<div class="result-box {'positive' if good else 'negative'}"><div class="eyebrow">Latest result</div><h2 class="result-title">{'😊 Positive sentiment' if good else '🙁 Negative sentiment'}</h2><p class="result-detail">Model confidence score</p><h1 class="result-score">{result['score']*100:.2f}%</h1><span class="small-pill">Inference time: {result['seconds']:.3f}s</span></div>""", unsafe_allow_html=True)
            st.caption("Confidence is the model's score for its predicted class, not a guarantee of correctness.")
    with right:
        st.markdown('<div class="section-title">📊 Session insights</div>', unsafe_allow_html=True)
        if history:
            counts = pd.Series([x["label"] for x in history]).value_counts().reindex(["POSITIVE", "NEGATIVE"], fill_value=0)
            st.bar_chart(counts, color="#6f79ff")
            st.markdown("**Recent analyses**")
            for item in history[:5]:
                preview = item["review"].replace("\n", " ")
                if len(preview) > 78: preview = preview[:75] + "..."
                icon = "🟢" if item["label"] == "POSITIVE" else "🔴"
                st.markdown(f"{icon} **{item['label']} · {item['score']*100:.1f}%**  \n{preview}")
        else:
            st.info("Your chart and recent results will appear here after the first analysis.")
            st.markdown("**Sample text ideas**")
            st.code("The movie was brilliant and emotionally engaging.")
            st.code("The plot was boring and the acting was weak.")

elif page == "Batch data analysis":
    st.markdown('<div class="section-title">📂 Batch CSV analysis</div>', unsafe_allow_html=True)
    st.write("Upload a CSV with a text column named `review`, `text`, `sentence`, or `comment`. For gender rates, select a gender/sex column with values such as Female/Male or F/M.")
    file = st.file_uploader("Choose CSV file", type=["csv"])
    if file:
        try:
            file_bytes = file.getvalue()
            file_id = hashlib.sha256(file_bytes).hexdigest()
            df = pd.read_csv(io.BytesIO(file_bytes))
            st.write(f"Rows found: **{len(df):,}**")
            st.dataframe(df, height="content", width="stretch", hide_index=True, alt="All rows in the uploaded CSV")

            options = [c for c in df.columns if str(c).strip().casefold() in {"review", "text", "sentence", "comment"}]
            if not options:
                st.error("No recognised text column found. Rename it to review, text, sentence, or comment.")
            else:
                text_col = st.selectbox("Text column", options, key=f"batch_text_column_{file_id[:12]}")
                gender_options = [None, *df.columns]
                default_gender_index = next(
                    (i + 1 for i, column in enumerate(df.columns) if str(column).strip().casefold() in {"gender", "sex"}),
                    0,
                )
                gender_col = st.selectbox(
                    "Gender column (optional)",
                    gender_options,
                    index=default_gender_index,
                    format_func=lambda column: "Not provided" if column is None else str(column),
                    key=f"batch_gender_column_{file_id[:12]}",
                )
                if df.empty:
                    st.info("This CSV contains column headers but no data rows.")

                if st.button("Analyse CSV", type="primary", disabled=model is None or df.empty):
                    output = df.copy(); labels = []; scores = []
                    progress = st.progress(0); status = st.empty()
                    texts = output[text_col].fillna("").astype(str).tolist()
                    for i, txt in enumerate(texts):
                        if txt.strip():
                            p = model(txt[:10000], truncation=True, max_length=512)[0]
                            labels.append(p["label"].upper()); scores.append(round(float(p["score"]), 4))
                        else: labels.append("EMPTY"); scores.append(0.0)
                        progress.progress((i + 1) / max(1, len(texts)))
                        status.write(f"Processed {i+1:,} / {len(texts):,}")
                    output["sentiment"] = labels; output["confidence"] = scores
                    classified = output["sentiment"].isin(["POSITIVE", "NEGATIVE"])
                    positive_count = int((output["sentiment"] == "POSITIVE").sum())
                    negative_count = int((output["sentiment"] == "NEGATIVE").sum())
                    classified_count = positive_count + negative_count
                    gender_stats = {}
                    if gender_col is not None:
                        gender_values = output[gender_col].astype("string").str.strip().str.casefold()
                        for name, aliases in {
                            "Female": {"female", "f", "woman", "women"},
                            "Male": {"male", "m", "man", "men"},
                        }.items():
                            matching_gender = gender_values.isin(aliases).fillna(False)
                            gender_total = int((matching_gender & classified).sum())
                            gender_positive = int((matching_gender & (output["sentiment"] == "POSITIVE")).sum())
                            gender_negative = int((matching_gender & (output["sentiment"] == "NEGATIVE")).sum())
                            gender_stats[name] = {
                                "total": gender_total,
                                "positive": gender_positive,
                                "negative": gender_negative,
                                "positive_pct": percentage(gender_positive, gender_total),
                                "negative_pct": percentage(gender_negative, gender_total),
                            }

                    summary = {
                        "total": len(output),
                        "classified": classified_count,
                        "positive": positive_count,
                        "negative": negative_count,
                        "positive_pct": percentage(positive_count, classified_count),
                        "negative_pct": percentage(negative_count, classified_count),
                        "gender_col": gender_col,
                        "gender": gender_stats,
                    }
                    st.session_state.batch_analysis = {
                        "file_id": file_id,
                        "text_col": text_col,
                        "gender_col": gender_col,
                        "output": output,
                        "summary": summary,
                    }
                    progress.empty()
                    status.empty()

                analysis = st.session_state.get("batch_analysis")
                if analysis and (
                    analysis["file_id"] == file_id
                    and analysis["text_col"] == text_col
                    and analysis["gender_col"] == gender_col
                ):
                    output = analysis["output"]
                    summary = analysis["summary"]
                    st.success("Analysis complete.")
                    st.subheader("Sentiment summary")
                    positive_delta = (
                        f"{format_percentage(summary['positive_pct'])} of classified reviews"
                        if summary["positive_pct"] is not None
                        else "No classified reviews"
                    )
                    negative_delta = (
                        f"{format_percentage(summary['negative_pct'])} of classified reviews"
                        if summary["negative_pct"] is not None
                        else "No classified reviews"
                    )
                    positive_col, negative_col = st.columns(2)
                    with positive_col:
                        st.metric(
                            "Positive reviews",
                            f"{summary['positive']:,}",
                            positive_delta,
                            delta_color="off",
                            border=True,
                        )
                    with negative_col:
                        st.metric(
                            "Negative reviews",
                            f"{summary['negative']:,}",
                            negative_delta,
                            delta_color="off",
                            border=True,
                        )
                    st.caption(
                        f"Percentages use {summary['classified']:,} non-empty reviews with a positive or negative prediction."
                    )
                    chart_data = pd.DataFrame(
                        {"Sentiment": ["Positive", "Negative"], "Reviews": [summary["positive"], summary["negative"]]}
                    )
                    st.bar_chart(chart_data, x="Sentiment", y="Reviews", alt="Positive and negative review counts")

                    st.subheader("Sentiment by gender")
                    if gender_col is None:
                        st.info("Select a gender/sex column to calculate female and male percentages.")
                    else:
                        gender_columns = st.columns(4)
                        for column, gender, sentiment, label in [
                            (gender_columns[0], "Female", "positive", "Female positive"),
                            (gender_columns[1], "Female", "negative", "Female negative"),
                            (gender_columns[2], "Male", "positive", "Male positive"),
                            (gender_columns[3], "Male", "negative", "Male negative"),
                        ]:
                            stats = summary["gender"][gender]
                            with column:
                                denominator_note = (
                                    f"{stats[sentiment]:,} of {stats['total']:,} {gender.lower()} reviews"
                                    if stats["total"]
                                    else f"No classified {gender.lower()} reviews"
                                )
                                st.metric(
                                    label,
                                    format_percentage(stats[f"{sentiment}_pct"]),
                                    denominator_note,
                                    delta_color="off",
                                    border=True,
                                )

                    st.subheader("Analysed rows")
                    st.dataframe(
                        output,
                        height="content",
                        width="stretch",
                        hide_index=True,
                        alt="CSV rows with sentiment predictions and confidence scores",
                    )

                    summary_rows = [
                        ("Total rows", f"{summary['total']:,}"),
                        ("Classified reviews", f"{summary['classified']:,}"),
                        ("Positive reviews", f"{summary['positive']:,}"),
                        ("Positive percentage", format_percentage(summary["positive_pct"])),
                        ("Negative reviews", f"{summary['negative']:,}"),
                        ("Negative percentage", format_percentage(summary["negative_pct"])),
                    ]
                    if gender_col is None:
                        unavailable = "N/A - no gender column selected"
                        summary_rows.extend(
                            [
                                ("Female positive percentage", unavailable),
                                ("Female negative percentage", unavailable),
                                ("Male positive percentage", unavailable),
                                ("Male negative percentage", unavailable),
                            ]
                        )
                    else:
                        for gender in ("Female", "Male"):
                            stats = summary["gender"][gender]
                            summary_rows.extend(
                                [
                                    (f"{gender} positive reviews", f"{stats['positive']:,}"),
                                    (f"{gender} positive percentage", format_percentage(stats["positive_pct"])),
                                    (f"{gender} negative reviews", f"{stats['negative']:,}"),
                                    (f"{gender} negative percentage", format_percentage(stats["negative_pct"])),
                                ]
                            )

                    metric_column = "summary_metric"
                    value_column = "summary_value"
                    while metric_column in output.columns or value_column in output.columns:
                        metric_column = f"_{metric_column}"
                        value_column = f"_{value_column}"
                    export = output.copy()
                    export[metric_column] = ""
                    export[value_column] = ""
                    export_rows = []
                    for metric, value in summary_rows:
                        row = {column: "" for column in output.columns}
                        row[metric_column] = metric
                        row[value_column] = value
                        export_rows.append(row)
                    export = pd.concat([export, pd.DataFrame(export_rows, columns=export.columns)], ignore_index=True)
                    st.download_button(
                        "⬇️ Download results CSV",
                        export.to_csv(index=False).encode("utf-8-sig"),
                        "sentiment_analysis_results.csv",
                        "text/csv",
                        width="stretch",
                    )
        except Exception as e:
            st.session_state.pop("batch_analysis", None)
            st.error(f"Could not process this CSV: {e}")
    else:
        st.session_state.pop("batch_analysis", None)
else:
    st.markdown('<div class="section-title">🧩 About this model</div>', unsafe_allow_html=True)
    st.write("This app uses a pretrained DistilBERT classifier for English positive/negative sentiment.")
    st.markdown(f"- **Model:** `{MODEL_ID}`\n- **Task:** Binary sentiment classification\n- **Output:** POSITIVE or NEGATIVE and a confidence score\n- **Runtime:** GPU when available, otherwise CPU\n- **Memory:** Model pipeline is cached in RAM\n- **Limitations:** Sarcasm, mixed opinions, very short text, and non-English inputs may be harder to classify.")
    st.info("For a formal project report, evaluate on a labelled held-out dataset and report accuracy, precision, recall, F1-score, and a confusion matrix.")

st.markdown("---")
st.markdown('<div class="app-footer">SENTIMENT AI · Deep Learning Project · Streamlit + DistilBERT</div>', unsafe_allow_html=True)
