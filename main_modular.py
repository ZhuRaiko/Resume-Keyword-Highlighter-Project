"""
Modular Resume Keyword Highlighter
Clean, maintainable architecture with separated concerns
"""
import streamlit as st
import json
from modules.extractor import extract_from_file, normalize_resume_text
from modules.embeddings import load_models
from modules.scoring import load_knn_model, analyze_sentences
from modules.counters import count_keywords
from modules.highlight import highlight_keywords


# Configuration
st.set_page_config(page_title="SkillHighlight Analyzer", layout="wide")

# Load keywords once
@st.cache_data
def load_keywords():
    """Load keywords from JSON file"""
    with open("data/keywords.json", "r", encoding="utf-8") as f:
        data = json.load(f)
    return (
        data.get("ACTION_VERBS", []),
        data.get("SOFT_SKILLS", []),
        data.get("HARD_SKILLS", []),
        data.get("RECRUITER_KEYWORDS", [])
    )


# Main UI
def main():
    # Load models and keywords
    nlp, bert_model = load_models()
    knn_model = load_knn_model(bert_model)
    action_verbs, soft_skills, hard_skills, recruiter_keywords = load_keywords()
    
    # Sidebar configuration
    with st.sidebar:
        st.header("⚙️ Configuration")
        
        st.subheader("Matching Settings")
        token_aligned = st.checkbox("Token-aligned mode", value=True, 
                                     help="Highlight whole tokens only (safer for punctuation)")
        
        st.subheader("Heuristic Relaxation")
        st.caption("Enable to make highlighting more permissive")
        relax_hard = st.checkbox("Relax HARD skills", value=False,
                                  help="Allow hard skills without strict noun/object requirements")
        relax_action = st.checkbox("Relax ACTION verbs", value=False,
                                    help="Allow action verbs without direct object requirement")
        relax_soft = st.checkbox("Relax SOFT skills", value=True,
                                  help="Skip negative sentiment suppression for soft skills")
        relax_recruiter = st.checkbox("Relax RECRUITER keywords", value=True,
                                       help="Allow recruiter keywords in verbless bullets")
        
        st.subheader("Sentiment Threshold")
        soft_neg_threshold = st.slider("Soft skill sentiment threshold", 
                                         min_value=-1.0, max_value=0.0, value=-0.1, step=0.05,
                                         help="Suppress soft skills below this sentiment score")
        
        st.subheader("Rendering")
        render_legacy = st.checkbox("Legacy HTML rendering", value=False,
                                     help="Use the older, simpler highlight spans without padding/rounded corners")
        
        st.divider()
        st.caption("💡 Tip: Adjust settings above to customize highlighting behavior")
    
    # Main content
    st.title("SkillHighlight: Resume Self-Promotion Analyzer")
    st.write("Upload a resume or paste text to analyze your self-promotion and skills balance.")
    
    # Input section
    uploaded = st.file_uploader("Upload your resume (PDF, DOCX, TXT):", type=["pdf", "docx", "txt"])
    text_input = st.text_area("Or paste resume text here:")
    
    # Extract text
    text = ""
    if uploaded:
        try:
            text = extract_from_file(uploaded)
        except Exception as e:
            st.error(str(e))
            return
    elif text_input.strip():
        text = text_input.strip()
    
    if not text:
        st.info("Upload a file or paste resume text to begin analysis.")
        return
    
    # Normalize text
    text = normalize_resume_text(text)
    
    # Initialize disabled classifiers
    if 'disabled_classifiers' not in st.session_state:
        st.session_state.disabled_classifiers = set()
    disabled_bars = st.session_state.disabled_classifiers
    
    # Count keywords
    hard_pct, soft_pct, rec_pct, action_pct = count_keywords(
        text, hard_skills, soft_skills, recruiter_keywords, action_verbs
    )
    
    # Analyze sentences
    try:
        results, avg_score = analyze_sentences(nlp, knn_model, bert_model, text, action_verbs)
    except Exception as e:
        st.error(f"Analysis failed: {e}")
        return
    
    # Display results
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("Extracted / Input Text")
        try:
            highlighted_text = highlight_keywords(
                nlp, text, hard_skills, soft_skills, recruiter_keywords, action_verbs,
                disabled_labels=disabled_bars, token_aligned=token_aligned,
                relax_hard=relax_hard, relax_action=relax_action,
                relax_soft=relax_soft, relax_recruiter=relax_recruiter,
                soft_neg_threshold=soft_neg_threshold, render_legacy=render_legacy
            )
            st.html(f"<div style='line-height:1.6; font-size:16px; white-space:pre-wrap; font-family: monospace;'>{highlighted_text}</div>")
        except Exception as e:
            st.error(f"Highlighting failed: {e}")
            st.text(text)
    
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
            <div style='margin-bottom:10px;'><b>Action Verbs:</b> {action_pct}%
            <div style='height:10px; background-color:#ccc; border-radius:6px;'>
                <div style='width:{action_pct}%; background-color:#c62828; height:10px; border-radius:6px;'></div></div></div>
        """, unsafe_allow_html=True)
    
    # Self-promotion score
    st.subheader(f"Overall Self-Promotion Score: {avg_score:.2f}")
    if avg_score > 0.8:
        st.success("Excellent self-promotion! Your resume effectively showcases achievements.")
    elif avg_score > 0.5:
        st.info("Good self-promotion. You can further strengthen it with more measurable results or leadership verbs.")
    else:
        st.warning("Weak self-promotion. Add stronger action verbs, metrics, and self-achievement phrasing.")
    
    # Sentence analysis
    st.subheader("Sentence Analysis (sorted by score)")
    
    to_show = sorted(results, key=lambda x: x[1], reverse=True)
    for sent, score in to_show:
        color = "#2e7d32" if score > 0.8 else "#f9a825" if score > 0.5 else "#c62828"
        st.markdown(
            f"<div style='background-color:{color}; color:white; padding:10px; border-radius:8px; margin:6px 0;'>{sent}"
            f"<span style='float:right; font-size:13px;'>Score: {score:.2f}</span></div>",
            unsafe_allow_html=True
        )


if __name__ == "__main__":
    main()
