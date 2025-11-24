import streamlit as st
import streamlit as st
from sentence_transformers import SentenceTransformer
from textblob import TextBlob
from sklearn.neighbors import KNeighborsClassifier
import json
import spacy
import pandas as pd
import re
import io
import tempfile
import os
import joblib

# Minimal, stable Streamlit app with keyword highlighting and scoring.

@st.cache_resource
def load_models():
    nlp = spacy.load("en_core_web_sm")
    bert_model = SentenceTransformer("all-MiniLM-L6-v2")
    return nlp, bert_model
import streamlit as st
from sentence_transformers import SentenceTransformer
from textblob import TextBlob
from sklearn.neighbors import KNeighborsClassifier
import json
import spacy
import numpy as np
import pandas as pd
import re
import io
import tempfile
import os
import joblib

# Minimal, stable Streamlit app with keyword highlighting and scoring.

@st.cache_resource
def load_models():
    nlp = spacy.load("en_core_web_sm")
    bert_model = SentenceTransformer("all-MiniLM-L6-v2")
    return nlp, bert_model

nlp, bert_model = load_models()

# load keyword lists
with open("keywords.json", "r", encoding="utf-8") as f:
    keyword_data = json.load(f)

ACTION_VERBS = keyword_data.get("ACTION_VERBS", [])
SOFT_SKILLS = keyword_data.get("SOFT_SKILLS", [])
HARD_SKILLS = keyword_data.get("HARD_SKILLS", [])
RECRUITER_KEYWORDS = keyword_data.get("RECRUITER_KEYWORDS", [])


@st.cache_resource
def load_or_train_knn():
    model_path = "knn_model.pkl"
    csv_path = "self_promotion_dataset.csv"
    if os.path.exists(model_path):
        try:
            return joblib.load(model_path)
        except Exception:
            pass
    if not os.path.exists(csv_path):
        class Dummy:
            def predict_proba(self, X):
                return np.zeros((len(X), 2))
            def predict(self, X):
                return np.zeros(len(X), dtype=int)
        return Dummy()
    df = pd.read_csv(csv_path)
    if df.empty or "sentence" not in df.columns or "label" not in df.columns:
        return None
    try:
        X = bert_model.encode(df["sentence"].tolist())
        y = df["label"].astype(int).values
        knn = KNeighborsClassifier(n_neighbors=5)
        knn.fit(X, y)
        joblib.dump(knn, model_path)
        return knn
    except Exception:
        return None


knn_model = load_or_train_knn()


def has_metric(sentence: str) -> bool:
    return bool(re.search(r"(\b\d+%|\b\d{1,3}\b|\$\d+[kKmM]?)", sentence))


def sentiment_score(sentence: str) -> float:
    try:
        return float(TextBlob(sentence).sentiment.polarity)
    except Exception:
        return 0.0


def self_promotion_score(sentence: str) -> float:
    try:
        vec = bert_model.encode([sentence])
        if knn_model is None:
            return 0.0
        if hasattr(knn_model, "predict_proba"):
            probs = knn_model.predict_proba(vec)
            if probs.shape[1] > 1:
                return float(probs[0][1])
            return float(probs[0][0])
        return float(knn_model.predict(vec)[0])
    except Exception:
        return 0.0


def smart_split(text: str):
    text = text.replace("•", "\n• ").replace("– ", "- ").replace("—", " - ")
    lines = [l.strip() for l in text.splitlines() if l.strip()]
    segments = []
    buffer = []
    for line in lines:
        if line.startswith(('-', '*', '•')):
            buffer.append(re.sub(r'^[-*•\s]+', '', line))
            continue
        if buffer:
            segments.append(' • '.join(buffer))
            buffer = []
        segments.append(line)
    if buffer:
        segments.append(' • '.join(buffer))
    out = []
    for seg in segments:
        if seg and len(seg) < 60 and out:
            out[-1] = out[-1] + ' ' + seg
        else:
            out.append(seg)
    return out


def _escape_for_regex(s: str) -> str:
    return re.escape(s)


def count_keywords(text: str):
    total = len(text.split()) or 1
    def pct(count):
        return int(min(100, round(100.0 * count / total)))
    def count_list(lst):
        c = 0
        for k in sorted([x for x in lst if x], key=lambda x: -len(x)):
            try:
                pat = re.compile(rf"(?<!\w){_escape_for_regex(k)}(?!\w)", flags=re.IGNORECASE)
                c += len(pat.findall(text))
            except re.error:
                continue
        return c
    h = count_list(HARD_SKILLS)
    s = count_list(SOFT_SKILLS)
    r = count_list(RECRUITER_KEYWORDS)
    return pct(h), pct(s), pct(r)


def highlight_keywords(text: str) -> str:
    # avoid highlighting emails/urls
    excluded = []
    for m in re.finditer(r"\b[\w\.-]+@[\w\.-]+\.[a-zA-Z]{2,}\b|https?://\S+|www\.\S+", text):
        excluded.append((m.start(), m.end()))

    def overlaps(span):
        a0, a1 = span
        for b0, b1 in excluded:
            if not (a1 <= b0 or a0 >= b1):
                return True
        return False

    replacements = []
    occupied = []

    # build list of (label, keyword) and sort longest-first
    all_kws = []
    for k in RECRUITER_KEYWORDS:
        if k:
            all_kws.append(('RECRUITER', k))
    for k in SOFT_SKILLS:
        if k:
            all_kws.append(('SOFT', k))
    for k in HARD_SKILLS:
        if k:
            all_kws.append(('HARD', k))
    all_kws = sorted(all_kws, key=lambda x: -len(x[1]))

    for label, kw in all_kws:
        try:
            pat = re.compile(rf"(?<!\w){_escape_for_regex(kw)}(?!\w)", flags=re.IGNORECASE)
        except re.error:
            continue
        for m in pat.finditer(text):
            s_char, e_char = m.start(), m.end()
            if overlaps((s_char, e_char)):
                continue
            if any(not (e_char <= o0 or s_char >= o1) for o0, o1 in occupied):
                continue
            matched = text[s_char:e_char]
            color = '#ff9800' if label == 'RECRUITER' else '#2196f3' if label == 'SOFT' else '#4caf50'
            html = f"<span style='background-color:{color}; color:white; padding:2px 4px; border-radius:3px;'>{matched}</span>"
            replacements.append((s_char, e_char, html))
            occupied.append((s_char, e_char))

    out = text
    for s, e, html in sorted(replacements, key=lambda x: x[0], reverse=True):
        out = out[:s] + html + out[e:]
    return out


def analyze_self_promotion(text: str):
    # simple sentence scoring using KNN/BERT + metric boost
    sents = list(nlp(text).sents)
    results = []
    scores = []
    for sent in sents:
        sent_txt = sent.text.strip()
        if not sent_txt:
            continue
        score = self_promotion_score(sent_txt)
        if has_metric(sent_txt):
            score = min(1.0, score + 0.15)
        scores.append(score)
        results.append((sent_txt, score))
    avg = float(np.mean(scores)) if scores else 0.0
    return results, avg


# --- Streamlit UI ---
st.set_page_config(page_title="SkillHighlight Analyzer", layout="wide")
st.title("SkillHighlight: Resume Self-Promotion Analyzer")
st.write("Upload a resume or paste text to analyze your self-promotion and skills balance.")

uploaded = st.file_uploader("Upload your resume (PDF, DOCX, TXT):", type=["pdf", "docx", "txt"])
text_input = st.text_area("Or paste resume text here:")

text = ""
if uploaded:
    try:
        data = uploaded.read()
        name = uploaded.name.lower()
        if name.endswith('.txt'):
            try:
                text = data.decode('utf-8')
            except Exception:
                text = data.decode('latin-1', errors='ignore')
        else:
            try:
                from pdfminer.high_level import extract_text as extract_pdf_text
            except Exception:
                extract_pdf_text = None
            try:
                import docx2txt
            except Exception:
                docx2txt = None
            if name.endswith('.pdf') and extract_pdf_text:
                text = extract_pdf_text(io.BytesIO(data))
            elif name.endswith('.docx') and docx2txt:
                with tempfile.NamedTemporaryFile(delete=False, suffix='.docx') as tmp:
                    tmp.write(data)
                    tmp_path = tmp.name
                try:
                    text = docx2txt.process(tmp_path)
                finally:
                    try:
                        os.remove(tmp_path)
                    except Exception:
                        pass
    except Exception as e:
        st.error(f"Error extracting text from uploaded file: {e}")
        text = ""
elif text_input.strip():
    text = text_input.strip()

if text:
    hard_pct, soft_pct, rec_pct = count_keywords(text)
    results, avg_score = analyze_self_promotion(text)

    col1, col2 = st.columns([2, 1])
    with col1:
        st.subheader("Extracted / Input Text")
        st.markdown(f"<div style='line-height:1.6; font-size:16px;'>{highlight_keywords(text)}</div>", unsafe_allow_html=True)
    with col2:
        st.subheader("Keyword Composition")
        st.markdown(f"""
            <div style='margin-bottom:10px;'><b>Hard Skills:</b> {hard_pct}%
            <div style='height:10px; background-color:#ccc; border-radius:6px;'>
                <div style='width:{hard_pct}%; background-color:#4caf50; height:10px; border-radius:6px;'></div></div></div>
            <div style='margin-bottom:10px;'><b>Soft Skills:</b> {soft_pct}%
            <div style='height:10px; background-color:#ccc; border-radius:6px;'>
                <div style='width:{soft_pct}%; background-color:#2196f3; height:10px; border-radius:6px;'></div></div></div>
            <div style='margin-bottom:10px;'><b>Recruiter Keywords:</b> {rec_pct}%
            <div style='height:10px; background-color:#ccc; border-radius:6px;'>
                <div style='width:{rec_pct}%; background-color:#ff9800; height:10px; border-radius:6px;'></div></div></div>
        """, unsafe_allow_html=True)

    st.subheader(f"Overall Self-Promotion Score: {avg_score:.2f}")
    if avg_score > 0.8:
        st.success("Excellent self-promotion! Your resume effectively showcases achievements.")
    elif avg_score > 0.5:
        st.info("Good self-promotion. You can further strengthen it with more measurable results or leadership verbs.")
    else:
        st.warning("Weak self-promotion. Add stronger action verbs, metrics, and self-achievement phrasing.")

    st.subheader("Sentence Analysis (sorted by score)")
    for sent, score in sorted(results, key=lambda x: x[1], reverse=True):
        color = "#2e7d32" if score > 0.8 else "#f9a825" if score > 0.5 else "#c62828"
        st.markdown(
            f"<div style='background-color:{color}; color:white; padding:10px; border-radius:8px; margin:6px 0;'>{sent}"
            f"<span style='float:right; font-size:13px;'>Score: {score:.2f}</span></div>",
            unsafe_allow_html=True
        )
else:
    st.info("Upload a file or paste resume text to begin analysis.")
            root = span.root
            if root.pos_ != "VERB":
                allow = False
            else:
                has_obj = any([child.dep_ in ("dobj", "pobj", "obj") for child in root.children])
                if not has_obj:
                    allow = False
                else:
                    # use bright red for action verbs
                    color = "#e53935"
        elif label == "RECRUITER":
            # Always allow, but decrease priority if sentence is a bullet list without verbs
            if not meta.get("has_verb", True) and (sent is not None and (sent.text.strip().startswith(('-', '*', '•')) or len(sent.text.strip()) < 80)):
                # lower priority: enforce stricter per-key cap
                if cnt >= 1:
                    allow = False

        if not allow:
            continue

        # Passes heuristics: schedule replacement
        html = f"<span style='background-color:{color}; color:white; padding:2px 4px; border-radius:3px;'>{text_matched}</span>"
        replacements.append((s_char, e_char, html, label, meta.get("is_summary", False)))
        occupied.append((s_char, e_char))
        per_key_counts[(label, key_norm)] = cnt + 1

    # Sort replacements: prioritize summary-contained SOFT highlights then RECRUITER then SOFT then HARD then ACTION
    order_priority = {"RECRUITER": 3, "SOFT": 2, "HARD": 1, "ACTION": 4}
    replacements.sort(key=lambda x: (not x[4], order_priority.get(x[3], 0), x[0]), reverse=True)

    out = text
    for s, e, html, _, _ in sorted(replacements, key=lambda x: x[0], reverse=True):
        out = out[:s] + html + out[e:]
    return out

# -------------------------
# Streamlit UI
# The remainder of the file wires everything into a Streamlit app:
# 1. Accept an uploaded file or pasted text
# 2. Extract text from common file types (PDF, DOCX, TXT)
# 3. Compute keyword composition and self-promotion scores
# 4. Render highlighted input and per-segment score boxes
# -------------------------

# --- Streamlit UI ---
st.set_page_config(page_title="SkillHighlight Analyzer", layout="wide")
st.title("SkillHighlight: Resume Self-Promotion Analyzer")
st.write("Upload a resume or paste text to analyze your self-promotion and skills balance.")

uploaded = st.file_uploader("Upload your resume (PDF, DOCX, TXT):", type=["pdf", "docx", "txt"])
text_input = st.text_area("Or paste resume text here:")

text = ""
if uploaded:
    # Robust handling for uploaded files. Streamlit's UploadedFile provides
    # file bytes; pdfminer can read from a BytesIO, and docx2txt expects a
    # filename so we write a temp file for DOCX processing.
    try:
        data = uploaded.read()
        name = uploaded.name.lower()

        try:
            from pdfminer.high_level import extract_text as extract_pdf_text
        except Exception:
            extract_pdf_text = None

        try:
            import docx2txt
        except Exception:
            docx2txt = None

        if name.endswith(".pdf"):
            if extract_pdf_text is None:
                st.error("PDF extraction library not available (pdfminer).")
            else:
                text = extract_pdf_text(io.BytesIO(data))
        elif name.endswith(".docx"):
            if docx2txt is None:
                st.error("DOCX extraction library not available (docx2txt).")
            else:
                # write to temp file because docx2txt.process expects a path
                with tempfile.NamedTemporaryFile(delete=False, suffix=".docx") as tmp:
                    tmp.write(data)
                    tmp_path = tmp.name
                try:
                    text = docx2txt.process(tmp_path)
                finally:
                    try:
                        os.remove(tmp_path)
                    except Exception:
                        pass
        elif name.endswith(".txt"):
            try:
                text = data.decode("utf-8")
            except Exception:
                text = data.decode("latin-1", errors="ignore")
    except Exception as e:
        st.error(f"Error extracting text from uploaded file: {e}")
        text = ""
elif text_input.strip():
    text = text_input.strip()

if text:
    hard_pct, soft_pct, rec_pct = count_keywords(text)
    results, avg_score = analyze_self_promotion(text)

    col1, col2 = st.columns([2, 1])

    with col1:
        st.subheader("Extracted / Input Text")
        st.markdown(f"<div style='line-height:1.6; font-size:16px;'>{highlight_keywords(text)}</div>", unsafe_allow_html=True)

    with col2:
        st.subheader("Keyword Composition")
        st.markdown(f"""
            <div style='margin-bottom:10px;'><b>Hard Skills:</b> {hard_pct}%
            <div style='height:10px; background-color:#ccc; border-radius:6px;'>
                <div style='width:{hard_pct}%; background-color:#4caf50; height:10px; border-radius:6px;'></div></div></div>
            <div style='margin-bottom:10px;'><b>Soft Skills:</b> {soft_pct}%
            <div style='height:10px; background-color:#ccc; border-radius:6px;'>
                <div style='width:{soft_pct}%; background-color:#2196f3; height:10px; border-radius:6px;'></div></div></div>
            <div style='margin-bottom:10px;'><b>Recruiter Keywords:</b> {rec_pct}%
            <div style='height:10px; background-color:#ccc; border-radius:6px;'>
                <div style='width:{rec_pct}%; background-color:#ff9800; height:10px; border-radius:6px;'></div></div></div>
        """, unsafe_allow_html=True)

    st.subheader(f"Overall Self-Promotion Score: {avg_score:.2f}")
    if avg_score > 0.8:
        st.success("Excellent self-promotion! Your resume effectively showcases achievements.")
    elif avg_score > 0.5:
        st.info("Good self-promotion. You can further strengthen it with more measurable results or leadership verbs.")
    else:
        st.warning("Weak self-promotion. Add stronger action verbs, metrics, and self-achievement phrasing.")

    st.subheader("Sentence Analysis (sorted by score)")
    for sent, score in sorted(results, key=lambda x: x[1], reverse=True):
        color = "#2e7d32" if score > 0.8 else "#f9a825" if score > 0.5 else "#c62828"
        st.markdown(
            f"<div style='background-color:{color}; color:white; padding:10px; border-radius:8px; margin:6px 0;'>{sent}"
            f"<span style='float:right; font-size:13px;'>Score: {score:.2f}</span></div>",
            unsafe_allow_html=True
        )

else:
    st.info("Upload a file or paste resume text to begin analysis.")
