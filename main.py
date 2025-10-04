import streamlit as st
from sentence_transformers import SentenceTransformer
from textblob import TextBlob
from sklearn.linear_model import LogisticRegression
import spacy
import numpy as np
import re

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
    "python", "java", "javascript", "typescript", "react", "node.js", "html", "css", "sql", "docker",
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
    "cross-functional", "stakeholder", "scalable", "impact", "roi"
]

SELF_PROMO_EXAMPLES = [
    "Led a cross-functional team to deliver key business initiatives ahead of schedule.",
    "Recognized for outstanding leadership and consistent achievement of targets.",
    "Awarded for excellence in innovation and customer satisfaction.",
    "Increased operational efficiency by 30% through process optimization.",
    "Spearheaded a company-wide training program improving team productivity.",
    "Developed and deployed a scalable solution that reduced costs by 25%.",
    "Implemented new strategies that enhanced client retention rates.",
    "Created and managed high-impact marketing campaigns across digital channels.",
    "Drove a 40% growth in revenue by improving sales processes and customer engagement.",
    "Streamlined workflows, resulting in faster project delivery and improved quality."
]

# --- Add weak examples for contrast ---
WEAK_EXAMPLES = [
    "Responsible for handling tasks as assigned.",
    "Worked with a team to complete daily duties.",
    "Assisted in project development and provided support.",
    "Participated in meetings and discussions with colleagues.",
    "Helped with documentation and data entry.",
    "Performed regular maintenance checks as required.",
    "Handled customer inquiries and escalated issues.",
    "Supported team operations through assigned responsibilities.",
    "Maintained files and updated records as needed.",
    "Completed assigned work within deadlines."
]

# --- Train logistic regression classifier ---
@st.cache_resource
def train_selfpromo_model():
    strong_emb = bert_model.encode(SELF_PROMO_EXAMPLES)
    weak_emb = bert_model.encode(WEAK_EXAMPLES)
    X_train = np.vstack([strong_emb, weak_emb])
    y_train = np.array([1] * len(strong_emb) + [0] * len(weak_emb))
    clf = LogisticRegression(max_iter=1000)
    clf.fit(X_train, y_train)
    return clf

promo_model = train_selfpromo_model()

# --- Helper functions ---
def has_metric(sentence):
    return bool(re.search(r"(\b\d+%|\b\d{1,3}\b|\$\d+[kKmM]?)", sentence))

def sentiment_score(sentence):
    return TextBlob(sentence).sentiment.polarity

def uses_pronoun(sentence):
    return any(p in sentence.lower() for p in [" i ", " my ", " me "])

# --- Improved self-promotion scoring ---
def self_promotion_score(sentence):
    s = sentence.lower()
    base_score = 0.0

    if any(v in s for v in ACTION_VERBS):
        base_score += 0.4
    if has_metric(sentence):
        base_score += 0.3
    if any(skill in s for skill in HARD_SKILLS):
        base_score += 0.2
    if any(word in s for word in RECRUITER_KEYWORDS):
        base_score += 0.1
    if sentiment_score(sentence) > 0.05:
        base_score += 0.1
    if uses_pronoun(sentence):
        base_score -= 0.1
    if len(sentence.split()) > 12:
        base_score += 0.1

    base_score = max(0, min(1, base_score))

    # --- Learned model prediction ---
    vec = bert_model.encode([sentence])
    learned_prob = promo_model.predict_proba(vec)[0, 1]

    combined = (base_score * 0.4) + (learned_prob * 0.6)
    return max(0, min(1, combined))

def analyze_self_promotion(text):
    doc = nlp(text)
    results = [(sent.text, self_promotion_score(sent.text)) for sent in doc.sents]
    avg_score = np.mean([r[1] for r in results]) if results else 0
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
    for w in sorted(SOFT_SKILLS, key=len, reverse=True):
        text = re.sub(rf"\b({w})\b", r"<span style='background-color:#2196f3; color:white; padding:2px 4px; border-radius:3px;'>\1</span>", text, flags=re.IGNORECASE)
    for w in sorted(HARD_SKILLS, key=len, reverse=True):
        text = re.sub(rf"\b({w})\b", r"<span style='background-color:#4caf50; color:white; padding:2px 4px; border-radius:3px;'>\1</span>", text, flags=re.IGNORECASE)
    for w in sorted(RECRUITER_KEYWORDS, key=len, reverse=True):
        text = re.sub(rf"\b({w})\b", r"<span style='background-color:#ff9800; color:white; padding:2px 4px; border-radius:3px;'>\1</span>", text, flags=re.IGNORECASE)
    return text

# --- Streamlit UI ---
st.set_page_config(page_title="SkillHighlight Analyzer", layout="wide")

st.title("SkillHighlight: Resume Self-Promotion Analyzer")
st.write("Upload a resume file or paste text below to see the analysis.")

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
        st.subheader("Extracted / Input Text:")
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

    st.subheader(f"Self-Promotion Score: {avg_score:.2f}")
    if avg_score > 0.8:
        st.success("Excellent self-promotion! Your resume communicates achievements strongly.")
    elif avg_score > 0.5:
        st.info("Good self-promotion. You can enhance it by adding more action verbs or measurable results.")
    else:
        st.warning("Weak self-promotion detected. Try using action verbs and quantifiable outcomes.")

    st.subheader("Sentence Analysis (sorted by score):")
    for sent, score in sorted(results, key=lambda x: x[1], reverse=True):
        color = "#2e7d32" if score > 0.8 else "#f9a825" if score > 0.5 else "#c62828"
        st.markdown(
            f"<div style='background-color:{color}; color:white; padding:10px; border-radius:8px; margin:6px 0;'>{sent} "
            f"<span style='float:right; font-size:13px;'>Score: {score:.2f}</span></div>",
            unsafe_allow_html=True
        )

else:
    st.info("Upload a file or paste text to begin analysis.")
