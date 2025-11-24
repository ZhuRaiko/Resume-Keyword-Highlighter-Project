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

# Configuration toggles
# Set TOKEN_ALIGNED to True to highlight whole spaCy tokens (safer for punctuation edges)
TOKEN_ALIGNED = True
# Relax strict heuristics if you want more permissive matches
RELAX_HARD = False
RELAX_ACTION = False
# Relax soft-skill sentiment suppression and recruiter verbless downgrades
RELAX_SOFT = True
RELAX_RECRUITER = True
# Sentiment threshold for suppressing soft skills
SOFT_NEG_THRESHOLD = -0.1


# Clean, minimal Streamlit app (copy of main.py baseline) saved as main_clean.py
# Use this file to validate behavior before replacing main.py.

@st.cache_resource
def load_models():
    nlp = spacy.load("en_core_web_sm")
    bert_model = SentenceTransformer("all-MiniLM-L6-v2")
    return nlp, bert_model

nlp, bert_model = load_models()

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


def analyze_self_promotion(text: str):
    """Return list of (sentence, score) and average score for the document."""
    sents = []
    try:
        doc = nlp(text)
        for sent in doc.sents:
            stxt = sent.text.strip()
            if not stxt:
                continue
            base = self_promotion_score(stxt)
            bonus = 0.1 if has_metric(stxt) else 0.0
            # simple sentiment boost for positive sentences
            try:
                pol = TextBlob(stxt).sentiment.polarity
            except Exception:
                pol = 0.0
            pol_boost = 0.05 if pol > 0.1 else 0.0
            score = min(1.0, base + bonus + pol_boost)
            sents.append((stxt, score))
    except Exception:
        # fallback: split on periods
        parts = [p.strip() for p in re.split(r"(?<=[.!?])\s+", text) if p.strip()]
        for p in parts:
            sents.append((p, min(1.0, self_promotion_score(p))))
    avg = float(np.mean([s for _, s in sents])) if sents else 0.0
    return sents, avg


def _escape_for_regex(s: str) -> str:
    return re.escape(s)


def count_keywords(text: str):
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

    total_matches = h + s + r
    if total_matches > 0:
        # Show composition of matched keywords across categories (more intuitive for the UI bars)
        def comp(count):
            return int(round(100.0 * count / total_matches))
        return comp(h), comp(s), comp(r)
    # fallback: percent of total words (original behavior)
    total = len(text.split()) or 1
    def pct(count):
        return int(min(100, round(100.0 * count / total)))
    return pct(h), pct(s), pct(r)


def highlight_keywords(text: str) -> str:
    # find excluded spans (emails, urls)
    excluded = []
    for m in re.finditer(r"\b[\w\.-]+@[\w\.-]+\.[a-zA-Z]{2,}\b|https?://\S+|www\.\S+", text):
        excluded.append((m.start(), m.end()))
    def overlaps(span):
        a0, a1 = span
        for b0, b1 in excluded:
            if not (a1 <= b0 or a0 >= b1):
                return True
        return False

    # Build normalized maps for keywords using spaCy (lemmas) and joined no-punct forms
    all_keywords = []
    for k in RECRUITER_KEYWORDS:
        if k:
            all_keywords.append(("RECRUITER", k))
    for k in SOFT_SKILLS:
        if k:
            all_keywords.append(("SOFT", k))
    for k in HARD_SKILLS:
        if k:
            all_keywords.append(("HARD", k))
    for k in ACTION_VERBS:
        if k:
            all_keywords.append(("ACTION", k))

    # remove duplicates by normalized text
    seen = set()
    uniq_keywords = []
    for label, kw in all_keywords:
        kn = kw.strip()
        key = (label, kn.lower())
        if key in seen:
            continue
        seen.add(key)
        uniq_keywords.append((label, kn))

    # process keywords via nlp.pipe for speed
    map_by_tuple = {}   # tuple(lemmas) -> (label, display_kw)
    map_by_joined = {}  # joined_no_punct -> (label, display_kw)
    max_kw_len = 1
    if uniq_keywords:
        docs = list(nlp.pipe([kw for _, kw in uniq_keywords]))
        for (label, kw), kdoc in zip(uniq_keywords, docs):
            toks = [t for t in kdoc if not t.is_space]
            lem_tup = tuple(t.lemma_.lower() for t in toks) if toks else (kw.lower(),)
            joined = ''.join(re.sub(r"\W+", "", t.text.lower()) for t in toks)
            # skip keywords that normalize to empty (only punctuation) to avoid matching ':' or spaces
            if not joined and not any(re.search(r"\w", lt) for lt in lem_tup):
                continue
            # use fallback lower-key if joined empty but lem_tup has content
            joined_key = joined if joined else ''.join(re.sub(r"\W+", "", kw.lower()))
            map_by_tuple[lem_tup] = (label, kw)
            if joined_key:
                map_by_joined[joined_key] = (label, kw)
            if len(lem_tup) > max_kw_len:
                max_kw_len = len(lem_tup)

    # single spaCy parse of the document
    doc = nlp(text)
    tokens = [tok for tok in doc]
    n = len(tokens)

    # prepare sentence-level sentiment cache and verbless/bullet detection
    sents = list(doc.sents)
    sent_polarity = {}
    sent_has_verb = {}
    for i, s in enumerate(sents):
        try:
            sent_polarity[i] = TextBlob(s.text).sentiment.polarity
        except Exception:
            sent_polarity[i] = 0.0
        sent_has_verb[i] = any(t.pos_ == 'VERB' for t in s)
    sent_index_by_token = {}
    for idx, tok in enumerate(tokens):
        # find which sentence this token belongs to (by start char)
        for si, s in enumerate(sents):
            if tok.idx >= s.start_char and tok.idx < s.end_char:
                sent_index_by_token[idx] = si
                break

    occupied_tokens = set()
    replacements = []

    i = 0
    while i < n:
        if i in occupied_tokens:
            i += 1
            continue
        matched = False
        # try longest-first windows
        for L in range(min(max_kw_len, n - i), 0, -1):
            j = i + L - 1
            window = tokens[i:j+1]
            # build lemma tuple and joined no-punct
            lem_tup = tuple(t.lemma_.lower() for t in window)
            joined = ''.join(re.sub(r"\W+", "", t.text.lower()) for t in window)

            meta = None
            if lem_tup in map_by_tuple:
                meta = map_by_tuple[lem_tup]
            elif joined in map_by_joined:
                meta = map_by_joined[joined]

            if not meta:
                continue

            label, display_kw = meta

            # compute span chars (original token span)
            s_char = window[0].idx
            e_char = window[-1].idx + len(window[-1].text)

            # extract matched text and trim leading/trailing characters
            matched_text = text[s_char:e_char]

            # find first/last alphanumeric positions and expand to include contiguous allowed symbols
            allowed_symbols = set(['+', '#', '.', '-'])
            first_alnum = None
            last_alnum = None
            for idx, ch in enumerate(matched_text):
                if ch.isalnum():
                    first_alnum = idx
                    break
            for idx in range(len(matched_text)-1, -1, -1):
                if matched_text[idx].isalnum():
                    last_alnum = idx
                    break
            if first_alnum is None or last_alnum is None:
                continue
            # expand left from first_alnum to include contiguous allowed symbols
            first_i = first_alnum
            while first_i > 0 and matched_text[first_i-1] in allowed_symbols:
                first_i -= 1
            # expand right from last_alnum to include contiguous allowed symbols
            last_i = last_alnum
            while last_i+1 < len(matched_text) and matched_text[last_i+1] in allowed_symbols:
                last_i += 1

            # clamp runs of identical symbols at edges to sensible maxima
            # allowed counts per symbol: '+'->2, '#'->1, '.'->1, '-'->1
            allowed_counts = {'+': 2, '#': 1, '.': 1, '-': 1}
            # clamp right side
            if last_i > last_alnum:
                run_char = matched_text[last_alnum+1]
                if run_char in allowed_counts:
                    # count contiguous run length after last_alnum
                    run_len = 0
                    p = last_alnum + 1
                    while p < len(matched_text) and matched_text[p] == run_char:
                        run_len += 1
                        p += 1
                    max_allowed = allowed_counts.get(run_char, 1)
                    if run_len > max_allowed:
                        last_i = last_alnum + max_allowed
            # clamp left side
            if first_i < first_alnum:
                run_char = matched_text[first_i]
                if run_char in allowed_counts:
                    # count contiguous run length before first_alnum (from first_i up to first_alnum-1)
                    run_len = 0
                    p = first_i
                    while p < first_alnum and matched_text[p] == run_char:
                        run_len += 1
                        p += 1
                    max_allowed = allowed_counts.get(run_char, 1)
                    if run_len > max_allowed:
                        # trim extras from the left by advancing first_i
                        first_i = first_alnum - max_allowed

            new_s = s_char + first_i
            new_e = s_char + last_i + 1
            # ensure trimmed span still valid
            if new_s >= new_e:
                continue
            # update span to trimmed version for overlap checks
            if overlaps((new_s, new_e)):
                continue
            if any(k in occupied_tokens for k in range(i, j+1)):
                continue

            # heuristics per category
            sent_idx = sent_index_by_token.get(i, None)
            # HARD: require noun/proper noun or direct object present in window
            if label == 'HARD':
                ok = any(t.pos_ in ('NOUN', 'PROPN') for t in window) or any(t.dep_ == 'dobj' for t in window)
                if not ok:
                    continue
                color = '#4caf50'
            # SOFT: skip if negative sentiment in sentence (unless RELAX_SOFT)
            elif label == 'SOFT':
                # If user requested a relaxed soft-skill policy, skip negative suppression entirely
                if RELAX_SOFT:
                    color = '#1976d2'
                else:
                    neg = False
                    boost = False
                    if sent_idx is not None:
                        pol = sent_polarity.get(sent_idx, 0.0)
                        if pol < SOFT_NEG_THRESHOLD:
                            neg = True
                        # boost if in first 3 sentences (summary)
                        if sent_idx < 3:
                            boost = True
                    if neg:
                        continue
                    color = '#1976d2' if not boost else '#0d47a1'
            # ACTION: verb with a direct object
            elif label == 'ACTION':
                has_verb_obj = False
                for t in window:
                    if t.pos_ == 'VERB' and any(c.dep_ == 'dobj' for c in t.children):
                        has_verb_obj = True
                        break
                if not has_verb_obj:
                    continue
                color = '#c62828'
            # RECRUITER: downgrade color if sentence is verbless/bullet
            else:
                if not RELAX_RECRUITER and not sent_has_verb.get(sent_idx, True):
                    downgraded = True
                else:
                    downgraded = False
                color = '#ff9800' if not downgraded else '#ffdebc'

            # use trimmed matched text for display
            matched_text = text[new_s:new_e]
            # build html
            html = f"<span style='background-color:{color}; color:white; padding:2px 4px; border-radius:3px;'>{matched_text}</span>"
            replacements.append((new_s, new_e, html))
            for k in range(i, j+1):
                occupied_tokens.add(k)
            matched = True
            break
        if not matched:
            i += 1
        else:
            i += 1

    out = text
    for s, e, html in sorted(replacements, key=lambda x: x[0], reverse=True):
        out = out[:s] + html + out[e:]
    return out


# Streamlit UI (use this file to validate before replacing main.py)
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
