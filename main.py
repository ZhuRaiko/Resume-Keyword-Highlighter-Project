import streamlit as st
from sentence_transformers import SentenceTransformer
from textblob import TextBlob
from sklearn.neighbors import KNeighborsClassifier
import spacy
import numpy as np
import pandas as pd
import re
import os
import joblib
import torch
import torch.nn as nn   # >>> ADDED for BiLSTM

# --- Load models once ---
@st.cache_resource
def load_models():
    nlp = spacy.load("en_core_web_sm")
    bert_model = SentenceTransformer("all-MiniLM-L6-v2")
    return nlp, bert_model

nlp, bert_model = load_models()

# --- Keyword lists ---
ACTION_VERBS = [
    "achieved", "implemented", "developed", "led", "optimized", "created", "initiated",
    "designed", "coordinated", "delivered", "resolved", "mentored", "awarded",
    "recognized", "improved", "boosted", "generated", "increased", "spearheaded",
    "supervised", "analyzed", "engineered", "executed", "enhanced", "facilitated",
    "launched", "managed", "negotiated", "organized", "oversaw", "produced",
    "streamlined", "trained", "upgraded", "supported", "evaluated", "consulted"
]

SOFT_SKILLS = [
    "communication", "teamwork", "adaptability", "leadership", "creativity",
    "collaboration", "problem-solving", "critical thinking", "time management",
    "empathy", "conflict resolution", "decision making", "emotional intelligence",
    "organization", "negotiation", "flexibility", "work ethic", "initiative",
    "attention to detail", "accountability", "presentation", "multitasking"
]

HARD_SKILLS = [
    "python", "c++", "java", "javascript", "typescript", "react", "node.js", "html", "css", "sql", "docker",
    "kubernetes", "aws", "azure", "gcp", "git", "linux",
    "machine learning", "deep learning", "data analysis", "data visualization",
    "pandas", "numpy", "tensorflow", "pytorch", "statistics", "data engineering",
    "project management", "agile", "scrum", "budgeting", "forecasting",
    "risk management", "business analysis", "customer relationship management",
    "photoshop", "illustrator", "seo", "content marketing", "user experience", "figma", "canva"
]

RECRUITER_KEYWORDS = [
    "certified", "results-driven", "innovation", "initiative", "performance",
    "strategic", "deliverables", "optimization", "goal-oriented", "motivated",
    "analytical", "proactive", "efficient", "collaborative", "leadership",
    "cross-functional", "stakeholder", "scalable", "impact", "ROI"
]

# --- Load or train KNN model ---
@st.cache_resource
def load_or_train_knn():
    model_path = "knn_model.pkl"
    csv_path = "self_promotion_dataset.csv"

    if os.path.exists(model_path):
        knn = joblib.load(model_path)
        return knn

    if not os.path.exists(csv_path):
        st.error("Missing 'self_promotion_dataset.csv' — please add it to your project folder.")
        st.stop()

    df = pd.read_csv(csv_path)
    df = df.dropna(subset=["sentence", "label"])
    X_train = bert_model.encode(df["sentence"].tolist())
    y_train = df["label"].astype(int).values

    knn = KNeighborsClassifier(n_neighbors=5)
    knn.fit(X_train, y_train)
    joblib.dump(knn, model_path)
    return knn

knn_model = load_or_train_knn()

# >>> ADDED for BiLSTM
class BiLSTM(nn.Module):
    def __init__(self, hidden=128):
        super().__init__()
        self.embedding = nn.Embedding(30522, 128)
        self.lstm = nn.LSTM(128, hidden, batch_first=True, bidirectional=True)
        self.dropout = nn.Dropout(0.3)
        self.fc = nn.Linear(hidden * 2, 1)

    def forward(self, input_ids):
        x = self.embedding(input_ids)
        x, _ = self.lstm(x)
        x = self.dropout(x[:, -1, :])
        return torch.sigmoid(self.fc(x))

@st.cache_resource
def load_bilstm():
    model = BiLSTM()
    if os.path.exists("bilstm_selfpromo.pt"):
        model.load_state_dict(torch.load("bilstm_selfpromo.pt", map_location="cpu"))
    model.eval()
    return model

# tokenizer reused from sentence-transformer (wordpiece)
from transformers import BertTokenizer
bilstm_tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
bilstm_model = load_bilstm()

def bilstm_self_promotion_score(sentence):
    tokens = bilstm_tokenizer(sentence, return_tensors="pt", truncation=True, padding=True)
    with torch.no_grad():
        prob = bilstm_model(tokens["input_ids"]).item()
    return float(prob)
# <<< END OF ADDED

# --- Helper functions ---
def has_metric(sentence):
    return bool(re.search(r"(\b\d+%|\b\d{1,3}\b|\$\d+[kKmM]?)", sentence))

def sentiment_score(sentence):
    return TextBlob(sentence).sentiment.polarity

def self_promotion_score(sentence):
    vec = bert_model.encode([sentence])
    if hasattr(knn_model, "predict_proba"):
        prob = knn_model.predict_proba(vec)[0][1]
    else:
        prob = knn_model.predict(vec)[0]
    return float(prob)

def smart_split(text):
    # Improved smart_split to reduce UI clutter when displaying many tiny segments:
    # - normalize bullets and dashes
    # - group consecutive bullets into a single 'bullet block' segment
    # - skip contact lines (email/phone/urls) which are noise for scoring/display
    # - merge very short fragments with previous segment to avoid many tiny boxes
    text = text.replace("•", "\n• ").replace("– ", "- ").replace("—", " - ")
    lines = [l.strip() for l in text.splitlines() if l.strip()]

    segments = []
    buffer = ""
    bullet_block = []

    contact_re = re.compile(r"\b[\w\.-]+@[\w\.-]+\.[a-zA-Z]{2,}\b|https?://|www\.|\b\+?\d[\d\-\s\(\)]{6,}\d\b")
    header_re = re.compile(r"^(experience|career|summary|education|skills|projects|page)\b", re.IGNORECASE)

    def flush_buffer():
        nonlocal buffer
        if buffer:
            segments.append(buffer.strip())
            buffer = ""

    def flush_bullets():
        nonlocal bullet_block
        if bullet_block:
            # join bullets into a single segment to avoid dozens of tiny boxes
            segments.append(" • ".join(bullet_block))
            bullet_block = []

    for line in lines:
        # bullet lines -> collect
        if line.startswith(("-", "*", "•")):
            # remove leading bullet chars/spaces
            b = re.sub(r"^[-*•\s]+", "", line)
            bullet_block.append(b)
            continue

        # encountered non-bullet: flush any collected bullets first
        flush_bullets()

        # section headers / page titles -> act as strong separators
        if header_re.match(line):
            flush_buffer()
            segments.append(line)
            continue

        # skip contact lines (email, phone, urls) entirely to reduce clutter
        if contact_re.search(line):
            # do not add to buffer; continue
            continue

        # short standalone lines (likely headings or short labels)
        if len(line) < 45 and not line.endswith("."):
            flush_buffer()
            segments.append(line)
            continue

        # normal text -> accumulate until a period ends the block
        buffer += (" " + line)
        if line.endswith("."):
            flush_buffer()

    # final flushes
    flush_bullets()
    flush_buffer()

    # Post-process: merge very short segments with previous to avoid tiny boxes
    final_segments = []
    for seg in segments:
        s = seg.strip()
        if not s:
            continue
        # if segment is very short, merge with previous if possible
        if len(s) < 25 and final_segments and not header_re.match(final_segments[-1]):
            final_segments[-1] = final_segments[-1] + " " + s
        else:
            final_segments.append(s)

    return final_segments

def analyze_self_promotion(text):
    # Use smart_split to get resume-aware sentence/segment boundaries
    segments = smart_split(text)
    results = []
    for seg in segments:
        seg_text = seg.strip()
        if not seg_text:
            continue
        knn_score = self_promotion_score(seg_text)
        bilstm_score = bilstm_self_promotion_score(seg_text)   # >>> added ensemble
        score = (knn_score + bilstm_score) / 2                  # >>> hybrid without changing UI
        results.append((seg_text, score))
    avg_score = np.mean([s for _, s in results]) if results else 0
    return results, avg_score

def count_keywords(text):
    text_lower = text.lower()
    soft = sum(1 for w in SOFT_SKILLS if w in text_lower)
    hard = sum(1 for w in HARD_SKILLS if w in text_lower)
    rec = sum(1 for w in RECRUITER_KEYWORDS if w in text_lower)
    total = soft + hard + rec
    if total == 0:
        return 0, 0, 0
    return round(hard / total * 100, 1), round(soft / total * 100, 1), round(rec / total * 100, 1)

def highlight_keywords(text):
    # Build safe regex for each keyword. Use \b for pure word tokens, and
    # use lookaround boundaries for keywords that include non-word characters
    # (e.g. C++, C#, node.js) so they match correctly.
    def pattern_for(keyword: str) -> str:
        esc = re.escape(keyword)
        # if keyword only contains word characters, 
        # we can safely use \b boundaries for nicer matches
        if re.match(r"^[A-Za-z0-9_]+$", keyword):
            return rf"\b({esc})\b"
        # otherwise use lookarounds that assert non-word chars around the token
        return rf"(?<!\w)({esc})(?!\w)"

    for w in sorted(SOFT_SKILLS, key=len, reverse=True):
        pat = pattern_for(w)
        text = re.sub(pat, r"<span style='background-color:#2196f3; color:white; padding:2px 4px; border-radius:3px;'>\1</span>", text, flags=re.IGNORECASE)
    for w in sorted(HARD_SKILLS, key=len, reverse=True):
        pat = pattern_for(w)
        text = re.sub(pat, r"<span style='background-color:#4caf50; color:white; padding:2px 4px; border-radius:3px;'>\1</span>", text, flags=re.IGNORECASE)
    for w in sorted(RECRUITER_KEYWORDS, key=len, reverse=True):
        pat = pattern_for(w)
        text = re.sub(pat, r"<span style='background-color:#ff9800; color:white; padding:2px 4px; border-radius:3px;'>\1</span>", text, flags=re.IGNORECASE)
    return text

# --- Streamlit UI ---
st.set_page_config(page_title="SkillHighlight Analyzer", layout="wide")
st.title("SkillHighlight: Resume Self-Promotion Analyzer")
st.write("Upload a resume or paste text to analyze your self-promotion and skills balance.")

uploaded = st.file_uploader("Upload your resume (PDF, DOCX, TXT):", type=["pdf", "docx", "txt"])
text_input = st.text_area("Or paste resume text here:")

text = ""
if uploaded:
    import docx2txt
    from pdfminer.high_level import extract_text
    if uploaded.name.endswith(".pdf"):
        text = extract_text(uploaded)
    elif uploaded.name.endswith(".docx"):
        text = docx2txt.process(uploaded)
    elif uploaded.name.endswith(".txt"):
        text = uploaded.read().decode("utf-8")
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
