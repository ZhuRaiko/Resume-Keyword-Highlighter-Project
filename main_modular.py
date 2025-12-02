"""
Modular Resume Keyword Highlighter
Clean, maintainable architecture with separated concerns
"""
import streamlit as st
import json
from streamlit_toggle import st_toggle_switch
from modules.extractor import extract_from_file, normalize_resume_text
from modules.embeddings import load_models
from modules.scoring import load_knn_model, analyze_sentences
from modules.counters import count_keywords
from modules.highlight import highlight_keywords


# Configuration
st.set_page_config(page_title="SkillHighlight Analyzer", layout="wide")

# Load and encode fonts
import base64

def load_font_base64(font_path):
    """Load font file and convert to base64"""
    with open(font_path, "rb") as f:
        return base64.b64encode(f.read()).decode()

# Load Morganite fonts
morganite_black = load_font_base64("fonts/Morganite-Black.ttf")
morganite_bold = load_font_base64("fonts/Morganite-Bold.ttf")
morganite_extrabold = load_font_base64("fonts/Morganite-ExtraBold.ttf")
morganite_semibold = load_font_base64("fonts/Morganite-SemiBold.ttf")
morganite_medium = load_font_base64("fonts/Morganite-Medium.ttf")

# Load Deliware fonts
deliware_black = load_font_base64("fonts/deliware-black.otf")
deliware_bold = load_font_base64("fonts/deliware-bold.otf")

# Set dark background theme and custom fonts
st.markdown(f"""
    <style>
    /* Load Morganite fonts via base64 */
    @font-face {{
        font-family: 'Morganite';
        src: url(data:font/truetype;charset=utf-8;base64,{morganite_black}) format('truetype');
        font-weight: 900;
        font-style: normal;
    }}
    @font-face {{
        font-family: 'Morganite';
        src: url(data:font/truetype;charset=utf-8;base64,{morganite_bold}) format('truetype');
        font-weight: 700;
        font-style: normal;
    }}
    @font-face {{
        font-family: 'Morganite';
        src: url(data:font/truetype;charset=utf-8;base64,{morganite_extrabold}) format('truetype');
        font-weight: 800;
        font-style: normal;
    }}
    @font-face {{
        font-family: 'Morganite';
        src: url(data:font/truetype;charset=utf-8;base64,{morganite_semibold}) format('truetype');
        font-weight: 600;
        font-style: normal;
    }}
    @font-face {{
        font-family: 'Morganite';
        src: url(data:font/truetype;charset=utf-8;base64,{morganite_medium}) format('truetype');
        font-weight: 500;
        font-style: normal;
    }}
    
    /* Load Deliware fonts */
    @font-face {{
        font-family: 'Deliware';
        src: url(data:font/opentype;charset=utf-8;base64,{deliware_black}) format('opentype');
        font-weight: 900;
        font-style: normal;
    }}
    @font-face {{
        font-family: 'Deliware';
        src: url(data:font/opentype;charset=utf-8;base64,{deliware_bold}) format('opentype');
        font-weight: 700;
        font-style: normal;
    }}
    
    /* Force dark background */
    .stApp {{
        background-color: #0e1117;
    }}
    
    /* Main content area - add padding for breathing room */
    .main {{
        background-color: #0e1117;
    }}
    
    .main .block-container {{
        padding: 3rem 5rem !important;
        max-width: 95% !important;
    }}
    
    /* Sidebar */
    [data-testid="stSidebar"] {{
        background-color: #262730;
    }}
    
    /* Text colors */
    .stApp, .main, p, span, div {{
        color: #fafafa;
    }}
    
    /* Headers with Morganite font */
    h1 {{
        font-family: 'Morganite', sans-serif !important;
        font-weight: 900 !important;
        letter-spacing: 0.02em !important;
        color: #fafafa !important;
    }}
    
    h2 {{
        font-family: 'Morganite', sans-serif !important;
        font-weight: 800 !important;
        font-size: 3.5rem !important;
        letter-spacing: 0.02em !important;
        line-height: 1.15 !important;
        color: #fafafa !important;
    }}
    
    h3 {{
        font-family: 'Morganite', sans-serif !important;
        font-weight: 700 !important;
        font-size: 2rem !important;
        letter-spacing: 0.01em !important;
        line-height: 1.25 !important;
        color: #fafafa !important;
    }}
    
    h4, h5, h6 {{
        font-family: 'Morganite', sans-serif !important;
        font-weight: 600 !important;
        font-size: 1.5rem !important;
        letter-spacing: 0.01em !important;
        line-height: 1.3 !important;
        color: #fafafa !important;
    }}
    </style>
""", unsafe_allow_html=True)

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
        st.header("Configuration")

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
        st.caption("Tip: Adjust settings above to customize highlighting behavior")

    # Main content - Brutalist title with Morganite Black
    st.markdown("""
        <h1 style='font-family: "Morganite", sans-serif; font-weight: 900; font-size: 5.5rem; 
             letter-spacing: 0.02em; line-height: 1.05; margin-bottom: 0.5rem; 
             text-transform: none; font-stretch: expanded;'>
            SkillHighlight: Resume Self-Promotion Analyzer
        </h1>
    """, unsafe_allow_html=True)
    st.write("Upload a resume or paste text to analyze your self-promotion and skills balance.")

    # Input section
    uploaded = st.file_uploader("Upload your resume (PDF, DOCX, TXT):", type=["pdf", "docx", "txt"])
    text_input = st.text_area("Or paste resume text here:")

    # Extract text and track file type
    text = ""
    file_type = None
    if uploaded:
        try:
            text = extract_from_file(uploaded)
            file_type = uploaded.name.lower().split('.')[-1] if '.' in uploaded.name else None
        except Exception as e:
            st.error(str(e))
            return
    elif text_input.strip():
        text = text_input.strip()
        file_type = "text_input"

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

    # Analyze sentences with file type context
    try:
        results, avg_score = analyze_sentences(nlp, knn_model, bert_model, text, action_verbs, file_type)
    except Exception as e:
        st.error(f"Analysis failed: {e}")
        return

    # Display results
    col1, col2 = st.columns([2, 1])

    with col1:
        st.markdown("<h2 style='margin-top: 2rem; margin-bottom: 1.5rem;'>Extracted / Input Text</h2>", unsafe_allow_html=True)
        try:
            highlighted_text = highlight_keywords(
                nlp, text, hard_skills, soft_skills, recruiter_keywords, action_verbs,
                disabled_labels=disabled_bars, token_aligned=token_aligned,
                relax_hard=relax_hard, relax_action=relax_action,
                relax_soft=relax_soft, relax_recruiter=relax_recruiter,
                soft_neg_threshold=soft_neg_threshold, render_legacy=render_legacy
            )
            st.html(f"""
                <div style='line-height:1.6; font-size:16px; white-space:pre-wrap; 
                     font-family: monospace; height: 500px; overflow-y: auto; 
                     padding: 16px; border: 1px solid #444; border-radius: 8px;
                     background: #1e1e1e;'>
                    {highlighted_text}
                </div>
            """)
        except Exception as e:
            st.error(f"Highlighting failed: {e}")
            st.text(text)

    with col2:
        st.markdown("<h2 style='margin-top: 2rem; margin-bottom: 1.5rem;'>Keyword Analysis</h2>", unsafe_allow_html=True)
        
        # Custom CSS for toggle labels
        st.markdown("""
            <style>
            /* Clean toggle labels with Deliware font */
            [data-testid="stHorizontalBlock"] label {
                font-size: 16px !important;
                font-weight: 700 !important;
                font-family: 'Deliware', sans-serif !important;
                letter-spacing: 0.03em !important;
            }
            [data-testid="stHorizontalBlock"] {
                margin-bottom: 8px !important;
            }
            </style>
        """, unsafe_allow_html=True)
        
        # Hard Skills
        hard_enabled = 'HARD' not in disabled_bars
        cols = st.columns([0.15, 0.85])
        with cols[0]:
            hard_toggle = st_toggle_switch(
                label="",
                key="hard_toggle",
                default_value=hard_enabled,
                label_after=False,
                inactive_color="#555",
                active_color="#26a69a",
                track_color="#26a69a"
            )
        with cols[1]:
            st.markdown("<span style='font-family: Deliware, sans-serif; font-size: 24px; font-weight: 700; letter-spacing: 0.03em;'>Hard Skills</span>", unsafe_allow_html=True)
        if hard_toggle != hard_enabled:
            if hard_toggle:
                st.session_state.disabled_classifiers.discard('HARD')
            else:
                st.session_state.disabled_classifiers.add('HARD')
            st.rerun()
        
        bar_color = "#26a69a" if hard_toggle else "#555"
        st.markdown(f"""
            <div style='height:12px; background-color:#2a2a2a; border-radius:2px; margin-bottom:2px;'>
                <div style='width:{hard_pct}%; background-color:{bar_color}; height:12px; border-radius:2px;'></div>
            </div>
            <p style='font-size:11px; margin-bottom:20px; font-family: monospace; color: #999;'>{hard_pct}%</p>
        """, unsafe_allow_html=True)
        
        # Soft Skills
        soft_enabled = 'SOFT' not in disabled_bars
        cols = st.columns([0.15, 0.85])
        with cols[0]:
            soft_toggle = st_toggle_switch(
                label="",
                key="soft_toggle",
                default_value=soft_enabled,
                label_after=False,
                inactive_color="#555",
                active_color="#7e57c2",
                track_color="#7e57c2"
            )
        with cols[1]:
            st.markdown("<span style='font-family: Deliware, sans-serif; font-size: 24px; font-weight: 700; letter-spacing: 0.03em;'>Soft Skills</span>", unsafe_allow_html=True)
        if soft_toggle != soft_enabled:
            if soft_toggle:
                st.session_state.disabled_classifiers.discard('SOFT')
            else:
                st.session_state.disabled_classifiers.add('SOFT')
            st.rerun()
        
        bar_color = "#7e57c2" if soft_toggle else "#555"
        st.markdown(f"""
            <div style='height:12px; background-color:#2a2a2a; border-radius:2px; margin-bottom:2px;'>
                <div style='width:{soft_pct}%; background-color:{bar_color}; height:12px; border-radius:2px;'></div>
            </div>
            <p style='font-size:11px; margin-bottom:20px; font-family: monospace; color: #999;'>{soft_pct}%</p>
        """, unsafe_allow_html=True)
        
        # Recruiter Keywords
        rec_enabled = 'RECRUITER' not in disabled_bars
        cols = st.columns([0.15, 0.85])
        with cols[0]:
            rec_toggle = st_toggle_switch(
                label="",
                key="rec_toggle",
                default_value=rec_enabled,
                label_after=False,
                inactive_color="#555",
                active_color="#ff9f43",
                track_color="#ff9f43"
            )
        with cols[1]:
            st.markdown("<span style='font-family: Deliware, sans-serif; font-size: 24px; font-weight: 700; letter-spacing: 0.03em;'>Recruiter Keywords</span>", unsafe_allow_html=True)
        if rec_toggle != rec_enabled:
            if rec_toggle:
                st.session_state.disabled_classifiers.discard('RECRUITER')
            else:
                st.session_state.disabled_classifiers.add('RECRUITER')
            st.rerun()
        
        bar_color = "#ff9f43" if rec_toggle else "#555"
        st.markdown(f"""
            <div style='height:12px; background-color:#2a2a2a; border-radius:2px; margin-bottom:2px;'>
                <div style='width:{rec_pct}%; background-color:{bar_color}; height:12px; border-radius:2px;'></div>
            </div>
            <p style='font-size:11px; margin-bottom:20px; font-family: monospace; color: #999;'>{rec_pct}%</p>
        """, unsafe_allow_html=True)
        
        # Action Verbs
        act_enabled = 'ACTION' not in disabled_bars
        cols = st.columns([0.15, 0.85])
        with cols[0]:
            act_toggle = st_toggle_switch(
                label="",
                key="act_toggle",
                default_value=act_enabled,
                label_after=False,
                inactive_color="#555",
                active_color="#ef5350",
                track_color="#ef5350"
            )
        with cols[1]:
            st.markdown("<span style='font-family: Deliware, sans-serif; font-size: 24px; font-weight: 700; letter-spacing: 0.03em;'>Action Verbs</span>", unsafe_allow_html=True)
        if act_toggle != act_enabled:
            if act_toggle:
                st.session_state.disabled_classifiers.discard('ACTION')
            else:
                st.session_state.disabled_classifiers.add('ACTION')
            st.rerun()
        
        bar_color = "#ef5350" if act_toggle else "#555"
        st.markdown(f"""
            <div style='height:12px; background-color:#2a2a2a; border-radius:2px; margin-bottom:2px;'>
                <div style='width:{action_pct}%; background-color:{bar_color}; height:12px; border-radius:2px;'></div>
            </div>
            <p style='font-size:11px; margin-bottom:20px; font-family: monospace; color: #999;'>{action_pct}%</p>
        """, unsafe_allow_html=True)

    # Self-promotion score
    st.markdown("---")
    
    # Dynamic gradient based on score
    if avg_score > 0.8:
        # Excellent: Vibrant Green gradient
        gradient = "linear-gradient(135deg, #1b5e20 0%, #66bb6a 100%)"
    elif avg_score > 0.5:
        # Good: Yellow to Orange gradient
        gradient = "linear-gradient(135deg, #f57f17 0%, #ffa726 100%)"
    else:
        # Weak: Red to Pink gradient
        gradient = "linear-gradient(135deg, #b71c1c 0%, #ef5350 100%)"
    
    st.markdown(f"""
        <div style='background: {gradient}; 
             padding: 24px; border-radius: 12px; text-align: center; margin: 20px 0;'>
            <h2 style='color: white; margin: 0; font-size: 2.5rem;'>Self-Promotion Score: {avg_score:.2f}</h2>
        </div>
    """, unsafe_allow_html=True)
    
    if avg_score > 0.8:
        st.success("Excellent self-promotion! Your resume effectively showcases achievements.")
    elif avg_score > 0.5:
        st.info("Good self-promotion. Consider adding more measurable results or leadership verbs.")
    else:
        st.warning("Weak self-promotion. Add stronger action verbs, metrics, and achievements.")

    # Sentence analysis
    st.markdown("<h2 style='margin-top: 2rem; margin-bottom: 1.5rem;'>Sentence Analysis</h2>", unsafe_allow_html=True)

    to_show = sorted(results, key=lambda x: x[1], reverse=True)
    for sent, score in to_show:
        if score > 0.8:
            bg_color = "rgba(40, 167, 69, 0.15)"  # Green with low opacity
            border_color = "#28a745"
            badge_color = "#d4edda"
            badge_text = "#155724"
            badge = "Strong"
        elif score > 0.5:
            bg_color = "rgba(255, 193, 7, 0.15)"  # Yellow with low opacity
            border_color = "#ffc107"
            badge_color = "#fff3cd"
            badge_text = "#856404"
            badge = "Moderate"
        else:
            bg_color = "rgba(220, 53, 69, 0.15)"  # Red with low opacity
            border_color = "#dc3545"
            badge_color = "#f8d7da"
            badge_text = "#721c24"
            badge = "Weak"
        
        st.markdown(f"""
            <div style='padding: 12px 16px; border-radius: 8px; margin: 8px 0; 
                 border: 1px solid {border_color}; background: {bg_color};'>
                <div style='display: flex; justify-content: space-between; align-items: start;'>
                    <p style='margin: 0; flex: 1; color: #e0e0e0;'>{sent}</p>
                    <span style='background: {badge_color}; color: {badge_text}; 
                           padding: 4px 12px; border-radius: 12px; font-size: 12px; 
                           font-weight: 600; margin-left: 12px; white-space: nowrap;'>
                        {badge} ({score:.2f})
                    </span>
                </div>
            </div>
        """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()
