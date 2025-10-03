import streamlit as st
from sentence_transformers import SentenceTransformer
from textblob import TextBlob
import spacy
import re

# --- Load models ---
nlp = spacy.load("en_core_web_sm")
bert_model = SentenceTransformer("all-MiniLM-L6-v2")

# --- Action verbs list ---
ACTION_VERBS = [
    "achieved", "implemented", "developed", "led", "optimized", "created",
    "initiated", "designed", "coordinated", "delivered", "resolved", "enhanced"
]

# --- Functions ---
def extract_text(file):
    if file.name.endswith(".txt"):
        return file.read().decode("utf-8")
    elif file.name.endswith(".docx"):
        import docx2txt
        return docx2txt.process(file)
    elif file.name.endswith(".pdf"):
        from pdfminer.high_level import extract_text
        return extract_text(file)
    else:
        return ""

def has_metric(sentence):
    return bool(re.search(r"\b\d+%|\b\d{1,3}\b", sentence))

def sentiment_score(sentence):
    return TextBlob(sentence).sentiment.polarity

def uses_pronoun(sentence):
    return any(p in sentence.lower() for p in [" i ", " my ", " me "])

def self_promotion_score(sentence):
    score = 0
    if any(v in sentence.lower() for v in ACTION_VERBS):
        score += 0.4
    if has_metric(sentence):
        score += 0.3
    if sentiment_score(sentence) > 0.1:
        score += 0.2
    if uses_pronoun(sentence):
        score -= 0.2
    return max(0, min(1, score))

def analyze_self_promotion(text):
    doc = nlp(text)
    results = []
    for sent in doc.sents:
        score = self_promotion_score(sent.text)
        results.append((sent.text, score))
    avg_score = sum([r[1] for r in results]) / len(results)
    return results, avg_score

# --- Streamlit UI ---
st.title("🧠 SkillHighlight: Resume Keyword & Self-Promotion Analyzer")

uploaded = st.file_uploader("Upload your resume (PDF, DOCX, or TXT)", type=["pdf", "docx", "txt"])

if uploaded:
    text = extract_text(uploaded)
    st.subheader("Extracted Text:")
    st.write(text[:500] + "..." if len(text) > 500 else text)
    
    with st.spinner("Analyzing self-promotion..."):
        results, avg_score = analyze_self_promotion(text)
    
    st.subheader(f"💬 Self-Promotion Score: {avg_score:.2f}")
    if avg_score > 0.8:
        st.success("Excellent self-promotion! Your resume communicates achievements strongly.")
    elif avg_score > 0.5:
        st.info("Good self-promotion. You can enhance clarity by using more action verbs or measurable results.")
    else:
        st.warning("Weak self-promotion detected. Try using action verbs and quantifiable outcomes.")
    
    # Highlight sentences
    st.subheader("Sentence Analysis:")
    for sent, score in results:
        color = "lightgreen" if score > 0.8 else "lightyellow" if score > 0.5 else "#ffcccc"
        st.markdown(f"<div style='background-color:{color}; padding:8px; border-radius:6px; margin:4px 0;'>{sent}</div>", unsafe_allow_html=True)
