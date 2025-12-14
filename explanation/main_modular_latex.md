% main_modular_latex.md — readable LaTeX layout with highlighted comments

\section*{main_modular.py}

\subsection*{Overview}
\textbf{Purpose:} Streamlit UI that loads models, keywords, and resume inputs,
runs analysis pipelines, and renders highlighted text and per-sentence scores.

\textbf{Role in system:} Acts as the interactive entry point that wires the
extractor, embedder, classifier, counters, and highlight modules into a
user-facing analyzer.

\subsection*{Notes on LaTeX view}
The explanatory text above is plain black. The full source is embedded in a
`lstlisting` block below; include the `listings` and `xcolor` packages and set
`commentstyle=\color{green}` to render inline comments in green while keeping
the overview and headers in black for readability.

\begin{verbatim}
\usepackage{xcolor}
\usepackage{listings}
\lstset{commentstyle=\color{green}, basicstyle=\ttfamily\small, breaklines=true}
\end{verbatim}

\subsection*{Source — full module}
\begin{lstlisting}[language=Python, basicstyle=\ttfamily\small, breaklines=true, frame=single, commentstyle=\color{green}]
"""
Modular Resume Keyword Highlighter

What this module does:
    - Provide a Streamlit-based UI that loads models, keywords, and user
      inputs, runs analysis pipelines, and renders highlighted resume text
      and per-sentence self-promotion scores.

Why this module is necessary in the overall system:
    - Serves as the application entry point and glue layer that connects
      the extractor, embedding, scoring, counter, and highlighting modules
      into an interactive web UI for users to analyze resumes.

How this module connects to other parts of a larger NLP / ML pipeline:
    - Calls `modules.extractor` to convert uploaded files into plain text.
    - Loads BERT and spaCy models via `modules.embeddings.load_models()`.
    - Loads a KNN classifier via `modules.scoring.load_knn_model()` and
      computes per-sentence scores with `modules.scoring.analyze_sentences()`.
    - Uses `modules.counters.count_keywords()` for summary statistics.
    - Uses `modules.highlight.highlight_keywords()` to produce HTML output
      that visually marks matched keywords and phrases.
"""

import streamlit as st
import json
from streamlit_toggle import st_toggle_switch
from modules.extractor import extract_from_file, normalize_resume_text
from modules.embeddings import load_models
from modules.scoring import load_knn_model, analyze_sentences
from modules.counters import count_keywords
from modules.highlight import highlight_keywords


# What this function does:
#   - Read a font file and return its base64-encoded representation.
# Why this function exists:
#   - The Streamlit app embeds custom fonts via base64 in inline CSS; this
#     helper centralizes the binary-to-base64 conversion.
# Inputs expected:
#   - font_path (str): Path to the font file to open and encode.
# Returns / side effects:
#   - Returns a base64 string of the font file contents. Side effect: reads
#     the file from disk; no network activity.
# How it contributes to the larger NLP / ML system:
#   - Purely presentational: ensures consistent typography for the UI where
#     analysis results are displayed and inspected.
def load_font_base64(font_path):
    """Load font file and convert to base64"""
    with open(font_path, "rb") as f:
        return base64.b64encode(f.read()).decode()


# What this function does:
#   - Load structured keyword lists from `data/keywords.json` and return
#     them as separate lists for action verbs, soft skills, hard skills, and
#     recruiter keywords.
# Why this function exists:
#   - Encapsulate keyword file reading and provide caching so repeated UI
#     interactions don't re-read the file from disk.
# Inputs expected:
#   - No inputs; reads the fixed JSON path `data/keywords.json`.
# Returns / side effects:
#   - Returns a 4-tuple of lists: (ACTION_VERBS, SOFT_SKILLS, HARD_SKILLS, RECRUITER_KEYWORDS).
#   - Side effect: uses Streamlit caching via `@st.cache_data` to persist
#     the loaded lists in memory across user interactions.
# How it contributes to the larger NLP / ML system:
#   - Supplies the vocabulary used by `count_keywords()` and
#     `highlight_keywords()` to detect and display matches in resumes.
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


# What this function does:
#   - Coordinate the Streamlit UI lifecycle: set up page config, load models
#     and keywords, accept user input (file upload or pasted text), run
#     analysis pipelines, and render both highlighted text and numeric
#     summaries to the user.
# Why this function exists:
#   - Consolidate application wiring and presentation logic so the
#     project's analysis modules are exposed through a single interactive
#     entry point suitable for demonstration and manual inspection.
# Inputs expected:
#   - No function arguments: uses Streamlit state and widgets for input.
# Returns / side effects:
#   - Renders UI components and may call `st.error`, `st.info`, and other
#     Streamlit functions. Returns None; side effects are purely UI-oriented.
# How it contributes to the larger NLP / ML system:
#   - Orchestrates the flow of data through extractor -> embedder -> scorer ->
#     highlighter, providing human-facing outputs for model validation and
#     iterative improvements.
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
            """
            )
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
\end{lstlisting}
