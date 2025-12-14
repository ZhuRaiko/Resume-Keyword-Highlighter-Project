% counters_latex.md — readable LaTeX layout with highlighted comments

\section*{counters.py}

\subsection*{Overview}
\textbf{Purpose:} Count keyword occurrences and compute category percentages used for scoring and UI summaries.

\textbf{Role in system:} Provides simple numeric summaries of matched keywords that feed into scoring, display, and evaluation components.

\subsection*{Notes on this LaTeX view}
The module-level explanatory text above is in plain (black) text. The source is provided in a `lstlisting` block below; include `listings` and `xcolor` packages and set `commentstyle=\color{green}` to render code comments in green while keeping the LaTeX overview black.

\begin{verbatim}
\usepackage{xcolor}
\usepackage{listings}
\lstset{commentstyle=\color{green}, basicstyle=\ttfamily\small, breaklines=true}
\end{verbatim}

\subsection*{Source — full module}
\begin{lstlisting}[language=Python, basicstyle=\ttfamily\small, breaklines=true, frame=single, commentstyle=\color{green}]
################################################################################
# Module: counters.py
#
# What this module does:
#   - Provides simple keyword counting utilities and percentage calculations
#     across predefined keyword categories (hard skills, soft skills,
#     recruiter keywords, action verbs).
#   - Exposes `escape_regex()` to safely escape terms for regex matching and
#     `count_keywords()` to count occurrences and compute category percentages.
#
# Why this module is necessary in the overall system:
#   - Downstream scoring and reporting components need numeric summaries of
#     matched keywords to compute composition, percentages, and simple metrics
#     used in UI and evaluation reports.
#
# How this module connects to other parts of the NLP / ML pipeline:
#   - Called after keyword extraction/highlighting to summarize matches.
#   - Its output (category counts or percentages) is used by `scoring.py`,
#     the frontend display, and evaluation scripts to report keyword coverage.
#
"""Keyword counting and percentage calculation"""
import re


# Function: escape_regex
# What this function does:
#   - Escapes any special regex characters in the provided string so it can be
#     used safely inside a regular expression pattern.
# Why this function exists:
#   - Keyword terms may contain characters that have regex meaning; escaping
#     prevents unintended pattern matches or regex errors when compiling.
# Inputs expected:
#   - s (str): a keyword or phrase to be escaped for regex use.
# Returns / side effects:
#   - Returns a string with regex metacharacters escaped. No side effects.
# How it contributes to the larger NLP / ML system:
#   - Ensures robust, safe pattern construction when counting keyword matches
def escape_regex(s: str) -> str:
    """Escape special regex characters"""
    return re.escape(s)


# Function: count_keywords
# What this function does:
#   - Counts occurrences of keywords in four categories (hard, soft,
#     recruiter, action) within a text string and returns either category
#     composition percentages (if any matches exist) or fallback percentages
#     relative to total words.
# Why this function exists:
#   - Provides a compact summary of how matched keywords are distributed
#     across categories so downstream components can display or score them.
# Inputs expected:
#   - text (str): input text where keyword matches will be searched.
#   - hard_skills, soft_skills, recruiter_keywords, action_verbs (list): lists
#     of keyword strings to search for in `text`.
# Returns / side effects:
#   - Returns a 4-tuple of integers representing percentage values for each
#     category in the order (hard, soft, recruiter, action).
#   - No external side effects (pure computation on inputs).
# How it contributes to the larger NLP / ML system:
#   - Used by score aggregation and UI layers to show category composition and
#     to provide numeric features for evaluation or simple heuristics.
def count_keywords(text: str, hard_skills: list, soft_skills: list, 
                  recruiter_keywords: list, action_verbs: list):
    """Count keywords in each category and return percentages"""
    def count_list(lst):
        c = 0
        for k in sorted([x for x in lst if x], key=lambda x: -len(x)):
            try:
                pat = re.compile(rf"(?<!\w){escape_regex(k)}(?!\w)", flags=re.IGNORECASE)
                c += len(pat.findall(text))
            except re.error:
                continue
        return c

    h = count_list(hard_skills)
    s = count_list(soft_skills)
    r = count_list(recruiter_keywords)
    a = count_list(action_verbs)

    total_matches = h + s + r + a
    if total_matches > 0:
        # Show composition of matched keywords across categories
        def comp(count):
            return int(round(100.0 * count / total_matches))
        return comp(h), comp(s), comp(r), comp(a)
    
    # fallback: percent of total words
    total = len(text.split()) or 1
    def pct(count):
        return int(min(100, round(100.0 * count / total)))
    return pct(h), pct(s), pct(r), pct(a)
\end{lstlisting}
