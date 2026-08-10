import streamlit as st
import pandas as pd
import numpy as np
import pickle
import shap
import lime
import lime.lime_tabular
import matplotlib.pyplot as plt
import spacy
import re
import os
import warnings
warnings.filterwarnings("ignore")
# Get the directory where app.py is located
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

import os
import gdown

# ── Download models from Google Drive if not present ──────────
MODEL_IDS = {
    "models/lr_model.pkl":  "1K-SuX5qJFK_25ePPrO4T2XP8Zniuajnf",
    "models/svm_model.pkl": "1fA5MzO3C7-ss20BujWDHL96T3zro7B4l",
    "models/rf_model.pkl":  "1zHj1WtaQrdC8ko1tBReY9FAUfd8Tzdue",
    "models/scaler.pkl":    "1iJM4GRUCfSUSJFNFEAe-Y0eYrLXh3i8X",
    "models/tfidf.pkl":     "1V2850vY-TnAcjf1nQJJV34dlWh_41bcS",
}

os.makedirs("models", exist_ok=True)

for path, file_id in MODEL_IDS.items():
    if not os.path.exists(path):
        url = f"https://drive.google.com/uc?id={file_id}"
        print(f"Downloading {path}...")
        gdown.download(url, path, quiet=False)

def get_path(*parts):
    """Build absolute path relative to app.py location"""
    return os.path.join(BASE_DIR, *parts)

# ── Page Config ─────────────────────────────────────────────
st.set_page_config(
    page_title="Financial Ambiguity Detector",
    page_icon="🔍",
    layout="wide"
)

# ── Custom CSS ───────────────────────────────────────────────
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(135deg, #1E2761, #2E75B6);
        padding: 2rem;
        border-radius: 12px;
        color: white;
        margin-bottom: 2rem;
    }
    .result-card {
        padding: 1.5rem;
        border-radius: 10px;
        margin: 1rem 0;
    }
    .ambiguous-card {
        background: #FFF3CD;
        border-left: 5px solid #F59E0B;
    }
    .clear-card {
        background: #D4EDDA;
        border-left: 5px solid #28A745;
    }
</style>
""", unsafe_allow_html=True)

# ── Lexicons ─────────────────────────────────────────────────
MODAL_VERBS = [
    "may", "might", "could", "would", "should",
    "can", "ought", "shall"
]
VAGUE_QUANTIFIERS = [
    "some", "certain", "various", "several", "a number of",
    "many", "few", "limited", "significant", "meaningful",
    "substantial", "considerable", "modest", "marginal", "notable"
]
TEMPORAL_VAGUE = [
    "in due course", "over time", "in the near term",
    "going forward", "in the coming periods",
    "over the medium term", "at some point", "in the future",
    "eventually", "soon", "in the coming months",
    "near term", "medium term", "longer term", "short term"
]
CONDITIONAL_FRAMING = [
    "subject to", "assuming", "contingent upon", "provided that",
    "to the extent that", "dependent on", "barring", "absent",
    "in the absence of", "conditional on", "depending on"
]
HEDGE_WORDS = [
    "approximately", "roughly", "around", "about",
    "generally", "typically", "usually", "often",
    "largely", "broadly", "possibly", "potentially",
    "likely", "expected", "anticipated", "believed",
    "estimated", "appears", "seems", "suggests", "indicates"
]

# ── Helper — Find CSV ────────────────────────────────────────
def find_csv():
    paths = [
        get_path("data", "processed", "combined_clean.csv"),
        get_path("data", "Processed", "combined_clean.csv"),
    ]
    for p in paths:
        if os.path.exists(p):
            return p
    return None

# ── Load Models ──────────────────────────────────────────────
@st.cache_resource
def load_models():
    with open(get_path("models", "lr_model.pkl"),  "rb") as f: lr    = pickle.load(f)
    with open(get_path("models", "svm_model.pkl"), "rb") as f: svm   = pickle.load(f)
    with open(get_path("models", "rf_model.pkl"),  "rb") as f: rf    = pickle.load(f)
    with open(get_path("models", "scaler.pkl"),    "rb") as f: sc    = pickle.load(f)
    with open(get_path("models", "tfidf.pkl"),     "rb") as f: tfidf = pickle.load(f)
    return lr, svm, rf, sc, tfidf

@st.cache_resource
def load_nlp():
    try:
        return spacy.load("en_core_web_sm")
    except OSError:
        from spacy.cli import download as spacy_download
        spacy_download("en_core_web_sm")
        return spacy.load("en_core_web_sm")

# ── Feature Extraction ───────────────────────────────────────
def extract_features(text, nlp):
    doc   = nlp(str(text))
    words = [t.text.lower() for t in doc]
    if not words:
        return None

    sent_length = len(words)

    def get_depth(token):
        d = 0
        while token.head != token:
            token = token.head
            d += 1
        return d

    tree_depth   = max((get_depth(t) for t in doc), default=0)
    func_pos     = {"DT","IN","CC","PRP","PRP$","WDT","WP","WRB","MD","TO"}
    func_count   = sum(1 for t in doc if t.pos_ in func_pos)
    func_ratio   = func_count / len(words) if words else 0
    punct_count  = sum(1 for t in doc if t.is_punct)
    word_lengths = [len(t.text) for t in doc if not t.is_punct and not t.is_space]
    avg_word_len = sum(word_lengths)/len(word_lengths) if word_lengths else 0
    neg_count    = sum(1 for t in doc if t.dep_ == "neg")
    ent_count    = len(doc.ents)
    clause_count = sum(1 for t in doc if t.pos_ == "CCONJ" or t.text == ",")
    verb_count   = sum(1 for t in doc if t.pos_ == "VERB")
    adj_count    = sum(1 for t in doc if t.pos_ == "ADJ")
    adv_count    = sum(1 for t in doc if t.pos_ == "ADV")
    noun_count   = sum(1 for t in doc if t.pos_ == "NOUN")
    dep_distances = [abs(t.i - t.head.i) for t in doc if t.dep_ != "ROOT"]
    avg_dep_dist  = sum(dep_distances)/len(dep_distances) if dep_distances else 0
    ends_question = int(str(text).strip().endswith("?"))
    ends_period   = int(str(text).strip().endswith("."))

    return {
        "sent_length"   : sent_length,
        "tree_depth"    : tree_depth,
        "func_ratio"    : round(func_ratio, 4),
        "punct_count"   : punct_count,
        "avg_word_len"  : round(avg_word_len, 4),
        "neg_count"     : neg_count,
        "ent_count"     : ent_count,
        "clause_count"  : clause_count,
        "verb_count"    : verb_count,
        "adj_count"     : adj_count,
        "adv_count"     : adv_count,
        "noun_count"    : noun_count,
        "avg_dep_dist"  : round(avg_dep_dist, 4),
        "ends_question" : ends_question,
        "ends_period"   : ends_period
    }

def clean_text(text):
    text = str(text).strip()
    text = re.sub(r'\s+', ' ', text)
    text = text.encode('ascii', 'ignore').decode('ascii')
    text = re.sub(r'http\S+|www\S+', '', text)
    text = text.lower().strip()
    return text

def detect_categories(text, nlp):
    doc        = nlp(str(text))
    text_lower = text.lower()
    found      = {}

    modals = [t.text for t in doc if t.tag_ == "MD" and t.text.lower() in MODAL_VERBS]
    if modals:
        found["Modal Hedging"] = modals

    passive = [t.text for t in doc if t.dep_ in ("nsubjpass", "auxpass")]
    if passive:
        found["Passive Voice"] = passive

    vague = [p for p in VAGUE_QUANTIFIERS if p in text_lower]
    if vague:
        found["Vague Quantifiers"] = vague

    temporal = [p for p in TEMPORAL_VAGUE if p in text_lower]
    if temporal:
        found["Temporal Vagueness"] = temporal

    conditional = [p for p in CONDITIONAL_FRAMING if p in text_lower]
    if conditional:
        found["Conditional Framing"] = conditional

    return found

def predict_sentence(text, lr, svm, rf, scaler, tfidf, nlp):
    cleaned  = clean_text(text)
    features = extract_features(cleaned, nlp)
    if not features:
        return None, None, None

    feat_df   = pd.DataFrame([features])
    tfidf_vec = tfidf.transform([cleaned])
    tfidf_df  = pd.DataFrame(
        tfidf_vec.toarray(),
        columns=[f"tfidf_{f}" for f in tfidf.get_feature_names_out()]
    )

    X            = pd.concat([feat_df, tfidf_df], axis=1)
    feature_names = list(X.columns)
    X_scaled     = scaler.transform(X)

    results = {}
    for name, model in [
        ("Logistic Regression", lr),
        ("SVM", svm),
        ("Random Forest", rf)
    ]:
        pred  = model.predict(X_scaled)[0]
        proba = model.predict_proba(X_scaled)[0][1]
        results[name] = {"prediction": int(pred), "confidence": float(proba)}

    return results, X_scaled, feature_names

# ── Build Background Data for SHAP and LIME ──────────────────
@st.cache_data
def load_background(_tfidf, _scaler, _nlp, n=150):
    csv_path = find_csv()
    if csv_path is None:
        return None, None

    df       = pd.read_csv(csv_path)
    sample   = df["sentence_clean"].dropna().sample(
        min(n, len(df)), random_state=42
    ).tolist()

    feat_rows = []
    valid_texts = []
    for text in sample:
        f = extract_features(str(text), _nlp)
        if f:
            feat_rows.append(f)
            valid_texts.append(text)

    if not feat_rows:
        return None, None

    feat_df    = pd.DataFrame(feat_rows)
    tfidf_mat  = _tfidf.transform(valid_texts)
    tfidf_df   = pd.DataFrame(
        tfidf_mat.toarray(),
        columns=[f"tfidf_{c}" for c in _tfidf.get_feature_names_out()]
    )
    combined   = pd.concat([
        feat_df.reset_index(drop=True),
        tfidf_df.reset_index(drop=True)
    ], axis=1)

    return combined, _scaler.transform(combined)

# ── Main App ─────────────────────────────────────────────────
def main():

    # Header
    st.markdown("""
    <div class="main-header">
        <h1 style="margin:0; font-size:2rem;">🔍 Financial Ambiguity Detector</h1>
        <p style="margin:0.5rem 0 0 0; opacity:0.85;">
            Interpretable NLP for Strategic Ambiguity Detection in Financial Text<br>
        </p>
    </div>
    """, unsafe_allow_html=True)

    # Load models
    try:
        lr, svm, rf, scaler, tfidf = load_models()
        nlp                        = load_nlp()
    except Exception as e:
        st.error(f"Could not load models: {e}")
        st.info("Make sure all pkl files are in the models/ folder")
        return

    # Load background data
    bg_df, bg_scaled = load_background(tfidf, scaler, nlp)

    # Sidebar
    with st.sidebar:
        st.markdown("### About This Project")
        st.markdown("""
        This tool detects **strategic linguistic ambiguity**
        in financial text — deliberately vague language used
        to avoid firm commitments without stating anything false.

        **5 Ambiguity Categories:**
        - 🔵 Modal Hedging
        - 🟣 Passive Voice
        - 🟢 Vague Quantifiers
        - 🟠 Temporal Vagueness
        - 🔴 Conditional Framing

        **Models:**
        - Logistic Regression
        - Support Vector Machine
        - Random Forest

        **Explainability:**
        - SHAP feature attributions
        - LIME local explanations
        """)

        st.markdown("### Example Sentences")
        examples = [
            "Revenues may see some improvement in certain segments going forward.",
            "The results were impacted by several factors subject to market conditions.",
            "We anticipate meaningful progress over the coming periods assuming stability.",
            "Net profit increased by 12% to EUR 45 million this quarter.",
            "The company will open three new stores in Dublin next year."
        ]
        for ex in examples:
            if st.button(f"📝 {ex[:50]}...", key=ex):
                st.session_state["input_text"] = ex

    # Input
    st.markdown("### Enter a Financial Sentence")
    default = st.session_state.get(
        "input_text",
        "Revenues may see some improvement in certain segments going forward, subject to broader market conditions."
    )

    input_text = st.text_area(
        "Type or paste a sentence from a financial document:",
        value=default,
        height=100
    )

    col1, col2 = st.columns([3, 1])
    with col1:
        analyse_btn = st.button(
            "🔍 Analyse Sentence", type="primary", use_container_width=True
        )
    with col2:
        clear_btn = st.button("🗑️ Clear", use_container_width=True)

    if clear_btn:
        st.session_state["input_text"] = ""
        st.rerun()

    if analyse_btn and input_text.strip():
        with st.spinner("Analysing sentence..."):
            results, X_scaled, feat_names = predict_sentence(
                input_text, lr, svm, rf, scaler, tfidf, nlp
            )

        if results is None:
            st.error("Could not process sentence. Please try again.")
            return

        categories = detect_categories(input_text, nlp)

        # ── Overall Verdict ──────────────────────────────────
        votes    = sum(1 for r in results.values() if r["prediction"] == 1)
        avg_conf = np.mean([r["confidence"] for r in results.values()])
        is_amb   = votes >= 2

        if is_amb:
            st.markdown(f"""
            <div class="result-card ambiguous-card">
                <h2 style="color:#856404; margin:0;">⚠️ STRATEGICALLY AMBIGUOUS</h2>
                <p style="margin:0.5rem 0 0 0; color:#856404;">
                    {votes}/3 models flagged this sentence as ambiguous
                    with average confidence of {avg_conf*100:.1f}%
                </p>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div class="result-card clear-card">
                <h2 style="color:#155724; margin:0;">✅ NOT AMBIGUOUS</h2>
                <p style="margin:0.5rem 0 0 0; color:#155724;">
                    {3-votes}/3 models classified this as a clear statement
                    with average confidence of {(1-avg_conf)*100:.1f}%
                </p>
            </div>
            """, unsafe_allow_html=True)

        # ── Category Badges ───────────────────────────────────
        st.markdown("### 🏷️ Ambiguity Categories Detected")
        cat_colors = {
            "Modal Hedging"      : "#1565C0",
            "Passive Voice"      : "#6A1B9A",
            "Vague Quantifiers"  : "#00695C",
            "Temporal Vagueness" : "#E65100",
            "Conditional Framing": "#B71C1C"
        }
        if categories:
            for cat, markers in categories.items():
                color       = cat_colors.get(cat, "#333333")
                markers_str = ", ".join(markers)
                st.markdown(f"""
                <span style="
                    background:{color}20;
                    border:1px solid {color};
                    color:{color};
                    padding:0.4rem 1rem;
                    border-radius:20px;
                    font-weight:bold;
                    margin:0.3rem;
                    display:inline-block;
                ">{cat}: <em>{markers_str}</em></span>
                """, unsafe_allow_html=True)
        else:
            st.info("No specific ambiguity markers detected")

        st.markdown("<br>", unsafe_allow_html=True)

        # ── Model Predictions ─────────────────────────────────
        st.markdown("### 📊 Model Predictions")
        cols = st.columns(3)
        model_colors = {
            "Logistic Regression": "#1E2761",
            "SVM"                : "#2E75B6",
            "Random Forest"      : "#0D9488"
        }
        for col, (mname, res) in zip(cols, results.items()):
            pred  = res["prediction"]
            conf  = res["confidence"]
            color = model_colors[mname]
            label = "AMBIGUOUS" if pred == 1 else "NOT AMBIGUOUS"
            icon  = "⚠️" if pred == 1 else "✅"
            with col:
                st.markdown(f"""
                <div style="
                    background:white;
                    border:2px solid {color};
                    border-radius:10px;
                    padding:1.2rem;
                    text-align:center;
                ">
                    <p style="color:{color};font-weight:bold;
                              font-size:0.9rem;margin:0;">{mname}</p>
                    <h3 style="margin:0.5rem 0;
                               color:{'#856404' if pred==1 else '#155724'};">
                        {icon} {label}
                    </h3>
                    <p style="margin:0;color:#555;">
                        Confidence: <strong>{conf*100:.1f}%</strong>
                    </p>
                </div>
                """, unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        # ── SHAP Explanation ──────────────────────────────────
        st.markdown("### 🔬 SHAP Explanation (Logistic Regression)")
        st.caption("Which features pushed the model toward or away from ambiguous")

        if bg_scaled is not None:
            try:
                # Align background columns with input columns
                bg_aligned = bg_df.copy()
                for col in feat_names:
                    if col not in bg_aligned.columns:
                        bg_aligned[col] = 0
                missing   = [c for c in feat_names if c not in bg_aligned.columns]
                for c in missing:
                    bg_aligned[c] = 0
                bg_aligned    = bg_aligned[feat_names]
                bg_scaled_ali = scaler.transform(bg_aligned)

                # Compute SHAP
                explainer  = shap.LinearExplainer(
                    lr, bg_scaled_ali,
                    feature_perturbation="interventional"
                )
                shap_vals  = explainer.shap_values(X_scaled)

                # Handle array shape
                if isinstance(shap_vals, list):
                    shap_row = shap_vals[0][0]
                elif shap_vals.ndim == 2:
                    shap_row = shap_vals[0]
                else:
                    shap_row = shap_vals

                # Build feature-value pairs
                pairs = list(zip(feat_names, shap_row))
                pairs.sort(key=lambda x: abs(x[1]), reverse=True)
                top12 = pairs[:12]

                names  = [str(f).replace("tfidf_", "")[:25] for f, _ in top12]
                values = [float(v) for _, v in top12]
                colors = ["#E53E3E" if v > 0 else "#2E75B6" for v in values]

                fig, ax = plt.subplots(figsize=(9, 5))
                ax.barh(names[::-1], values[::-1], color=colors[::-1],
                        edgecolor="white", height=0.6)
                ax.axvline(x=0, color="black", linewidth=1)
                ax.set_xlabel("SHAP Value (positive = toward ambiguous)")
                ax.set_title("Top 12 Feature Contributions — SHAP")
                ax.spines["top"].set_visible(False)
                ax.spines["right"].set_visible(False)
                plt.tight_layout()
                st.pyplot(fig)
                plt.close()
                st.caption(
                    "🔴 Red = pushes toward AMBIGUOUS  |  "
                    "🔵 Blue = pushes toward NOT AMBIGUOUS"
                )

            except Exception as e:
                st.warning(f"SHAP could not be generated: {e}")
        else:
            st.warning("Background data not found — SHAP unavailable")

        # ── LIME Explanation ──────────────────────────────────
        st.markdown("### 🧩 LIME Explanation (Logistic Regression)")
        st.caption("Which words/features locally influenced this specific prediction")

        if bg_scaled is not None:
            try:
                # Align background for LIME
                bg_aligned = bg_df.copy()
                for col in feat_names:
                    if col not in bg_aligned.columns:
                        bg_aligned[col] = 0
                bg_aligned    = bg_aligned[feat_names]
                bg_scaled_ali = scaler.transform(bg_aligned)

                lime_exp = lime.lime_tabular.LimeTabularExplainer(
                    training_data         = bg_scaled_ali,
                    feature_names         = feat_names,
                    class_names           = ["Not Ambiguous", "Ambiguous"],
                    mode                  = "classification",
                    discretize_continuous = True,
                    random_state          = 42
                )

                exp = lime_exp.explain_instance(
                    data_row     = X_scaled[0],
                    predict_fn   = lr.predict_proba,
                    num_features = 12,
                    num_samples  = 500,
                    labels       = (1,)
                )

                lime_list   = exp.as_list(label=1)
                lime_names  = [
                    str(f[0]).replace("tfidf_", "")[:25]
                    for f in lime_list
                ]
                lime_values = [float(f[1]) for f in lime_list]
                lime_colors = [
                    "#E53E3E" if v > 0 else "#2E75B6"
                    for v in lime_values
                ]

                fig2, ax2 = plt.subplots(figsize=(9, 5))
                ax2.barh(
                    lime_names[::-1], lime_values[::-1],
                    color=lime_colors[::-1],
                    edgecolor="white", height=0.6
                )
                ax2.axvline(x=0, color="black", linewidth=1)
                ax2.set_xlabel("LIME Weight (positive = toward ambiguous)")
                ax2.set_title("Local Feature Importance — LIME")
                ax2.spines["top"].set_visible(False)
                ax2.spines["right"].set_visible(False)
                plt.tight_layout()
                st.pyplot(fig2)
                plt.close()
                st.caption(
                    "🔴 Red = pushes toward AMBIGUOUS  |  "
                    "🔵 Blue = pushes toward NOT AMBIGUOUS"
                )

            except Exception as e:
                st.warning(f"LIME could not be generated: {e}")
        else:
            st.warning("Background data not found — LIME unavailable")

        # ── Confidence Chart ──────────────────────────────────
        st.markdown("### 📈 Confidence Scores Across Models")
        mnames = list(results.keys())
        confs  = [results[m]["confidence"] * 100 for m in mnames]
        bcolors = ["#E53E3E" if c >= 50 else "#2E75B6" for c in confs]

        fig3, ax3 = plt.subplots(figsize=(8, 3))
        bars = ax3.bar(mnames, confs, color=bcolors,
                       edgecolor="white", linewidth=1.5)
        ax3.axhline(y=50, color="gray", linestyle="--",
                    linewidth=1, label="Decision boundary (50%)")
        ax3.set_ylabel("Confidence — Ambiguous (%)")
        ax3.set_ylim(0, 110)
        ax3.spines["top"].set_visible(False)
        ax3.spines["right"].set_visible(False)
        for bar, conf in zip(bars, confs):
            ax3.text(
                bar.get_x() + bar.get_width()/2,
                bar.get_height() + 1,
                f"{conf:.1f}%",
                ha="center", va="bottom", fontweight="bold"
            )
        plt.tight_layout()
        st.pyplot(fig3)
        plt.close()

    elif analyse_btn and not input_text.strip():
        st.warning("Please enter a sentence to analyse")

    # ── Footer ────────────────────────────────────────────────
    st.markdown("---")
    st.markdown("""
    <div style="text-align:center;color:#888;font-size:0.85rem;">
        Interpretable NLP for Strategic Ambiguity Detection in Financial Disclosures<br>
    </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()
