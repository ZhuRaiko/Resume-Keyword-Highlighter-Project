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
import streamlit.components.v1 as components

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
    
    # Achievement and impact indicators
    achievement_words = {'achieved', 'delivered', 'improved', 'increased', 'reduced', 'led', 'managed', 
                        'developed', 'created', 'launched', 'implemented', 'drove', 'exceeded', 'optimized',
                        'streamlined', 'established', 'built', 'designed', 'spearheaded', 'pioneered'}
    impact_words = {'resulting', 'leading to', 'by', 'saved', 'generated', 'boosted', 'enhanced'}
    
    def detect_achievement_pattern(stxt_lower):
        """Detect strong achievement patterns like 'Achieved X by doing Y'"""
        has_achievement = any(w in stxt_lower for w in achievement_words)
        has_impact = any(w in stxt_lower for w in impact_words)
        return has_achievement and has_impact
    
    def is_bullet_point(stxt):
        """Check if sentence starts with bullet/list marker"""
        return bool(re.match(r'^\s*[-•●○▪▫◦⦿⦾]\s+|^\s*\d+[\.)]\s+', stxt))
    
    sents = []
    try:
        doc = nlp(text)
        for sent in doc.sents:
            stxt = sent.text.strip()
            if not stxt:
                continue
            
            stxt_lower = stxt.lower()
            
            # Base KNN score
            base = self_promotion_score(stxt)
            
            # Metric bonus (stronger weight)
            metric_bonus = 0.15 if has_metric(stxt) else 0.0
            
            # Achievement pattern bonus
            achievement_bonus = 0.12 if detect_achievement_pattern(stxt_lower) else 0.0
            
            # Bullet point bonus (resume bullets are typically achievements)
            bullet_bonus = 0.08 if is_bullet_point(stxt) else 0.0
            
            # Sentiment boost for positive sentences
            try:
                pol = TextBlob(stxt).sentiment.polarity
            except Exception:
                pol = 0.0
            pol_boost = 0.06 if pol > 0.15 else 0.0
            
            # Action verb bonus (starts with strong action verb)
            action_verb_bonus = 0.0
            if ACTION_VERBS:
                first_word = stxt_lower.split()[0] if stxt_lower else ''
                if any(first_word.startswith(v.lower()) for v in ACTION_VERBS[:50]):  # check first 50
                    action_verb_bonus = 0.08
            
            # Length penalty (very short sentences unlikely to be achievements)
            length_penalty = -0.1 if len(stxt.split()) < 5 else 0.0
            
            # Calculate final score with all bonuses
            score = base + metric_bonus + achievement_bonus + bullet_bonus + pol_boost + action_verb_bonus + length_penalty
            score = min(1.0, max(0.0, score))  # clamp to [0, 1]
            
            sents.append((stxt, score))
    except Exception:
        # fallback: split on newlines, periods and bullets
        parts = [p.strip() for p in re.split(r"\n+|(?<=[.!?])\s+|(?<=^)\s*[-•●]\s+", text, flags=re.MULTILINE) if p.strip()]
        for p in parts:
            base_score = self_promotion_score(p)
            metric_bonus = 0.15 if has_metric(p) else 0.0
            sents.append((p, min(1.0, base_score + metric_bonus)))
    
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
    a = count_list(ACTION_VERBS)

    total_matches = h + s + r + a
    if total_matches > 0:
        # Show composition of matched keywords across categories (more intuitive for the UI bars)
        def comp(count):
            return int(round(100.0 * count / total_matches))
        return comp(h), comp(s), comp(r), comp(a)
    # fallback: percent of total words (original behavior)
    total = len(text.split()) or 1
    def pct(count):
        return int(min(100, round(100.0 * count / total)))
    return pct(h), pct(s), pct(r), pct(a)


def highlight_keywords(text: str, disabled_labels=None) -> str:
    if disabled_labels is None:
        disabled_labels = set()
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

    # Context validation: check if matched tokens are used as skills/attributes (not incidental)
    def is_valid_context(window, label):
        """Validate that the matched window is used in appropriate context for the given label."""
        
        # HARD skills: Check for date/time context patterns first (applies to all window sizes)
        if label == 'HARD':
            # Check if any token in the window is near a year
            for tok in window:
                nearby_tokens = []
                for offset in range(-3, 4):  # Check 3 tokens before and after
                    idx = tok.i + offset
                    if 0 <= idx < len(tok.doc) and idx != tok.i:
                        nearby_tokens.append(tok.doc[idx])
                
                # If there's a year (4-digit number) nearby, likely a date context
                has_year_nearby = any(t.text.isdigit() and len(t.text) == 4 and 1900 <= int(t.text) <= 2100 
                                     for t in nearby_tokens if t.text.isdigit())
                if has_year_nearby:
                    # Check if token is a common non-technical word that could be in dates
                    non_tech_in_dates = {'spring', 'summer', 'fall', 'winter', 'may', 'march', 'august'}
                    if tok.text.lower() in non_tech_in_dates:
                        return False
        
        # Single-token checks for all labels
        if len(window) == 1:
            tok = window[0]
            # Skip if token is modifying another noun (adjective before noun in compound)
            if tok.pos_ == 'ADJ' and tok.head != tok and tok.head.pos_ == 'NOUN' and tok.i < tok.head.i:
                # Check if this is part of a compound phrase like 'quality assurance' where 'quality' shouldn't highlight
                return False
            # Skip if token is a generic verb (not used as skill)
            if tok.pos_ == 'VERB' and tok.dep_ not in ('ROOT', 'conj', 'xcomp', 'ccomp'):
                return False
            
            # HARD skills: additional validation to prevent false positives
            if label == 'HARD':
                # Skip if it's part of a larger compound phrase (e.g., "python" in "python snake")
                # Check if the token has children that extend beyond the window
                for child in tok.children:
                    if child not in window and child.dep_ in ('compound', 'amod', 'nmod'):
                        # Check if this forms a non-technical phrase
                        if child.i > tok.i:  # child comes after
                            return False
                
                # Skip if the token is being used attributively (modifying a non-technical noun)
                if tok.dep_ in ('amod', 'compound') and tok.head not in window:
                    # It's modifying something outside the window - likely not a skill reference
                    return False
        
        # Multi-token checks: ensure the phrase stands alone (not part of larger phrase)
        first_tok = window[0]
        last_tok = window[-1]
        
        # Skip if first token is modifying something outside the window
        if first_tok.pos_ == 'ADJ' and first_tok.head not in window and first_tok.head.pos_ == 'NOUN':
            return False
        
        # Skip if last token is being modified by something outside the window on the right
        for child in last_tok.children:
            if child.i > last_tok.i and child not in window and child.dep_ in ('amod', 'compound', 'nmod'):
                return False
        
        return True

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

    # Detect skill enumeration patterns: "X and Y skills/abilities" or "X, Y, and Z skills"
    # These should be highlighted as a whole phrase
    def find_skill_enumerations():
        enum_spans = []
        for sent in sents:
            sent_tokens = [t for t in sent if t.i < n]
            for i, tok in enumerate(sent_tokens):
                # Look for "skills", "abilities", "qualities" at end
                if tok.lemma_.lower() in ('skill', 'ability', 'quality', 'strength', 'competency', 'expertise'):
                    # Walk backwards to find start of enumeration
                    start_idx = None
                    for j in range(i-1, max(-1, i-10), -1):  # look back up to 10 tokens
                        t = sent_tokens[j]
                        # Check if this could be start of skill list
                        if t.pos_ in ('NOUN', 'ADJ', 'PROPN') and (j == 0 or sent_tokens[j-1].pos_ not in ('NOUN', 'ADJ', 'PROPN', 'CCONJ', 'PUNCT')):
                            # Verify there's a conjunction or comma in between
                            has_conj = any(sent_tokens[k].pos_ in ('CCONJ',) or sent_tokens[k].text in (',', 'and', 'or') for k in range(j, i))
                            if has_conj:
                                start_idx = sent_tokens[j].i
                                break
                    if start_idx is not None:
                        enum_spans.append((start_idx, tok.i + 1))  # (start_tok_idx, end_tok_idx exclusive)
        return enum_spans
    
    skill_enum_spans = find_skill_enumerations()

    i = 0
    while i < n:
        if i in occupied_tokens:
            i += 1
            continue
        matched = False
        
        # First, check if this position starts a skill enumeration pattern
        for enum_start, enum_end in skill_enum_spans:
            if i == enum_start and not any(k in occupied_tokens for k in range(enum_start, enum_end)):
                # Found skill enumeration starting here, highlight entire phrase
                window = tokens[enum_start:enum_end]
                s_char = window[0].idx
                e_char = window[-1].idx + len(window[-1].text)
                matched_text = text[s_char:e_char].strip()
                # Label as SOFT (skill enumerations are typically soft skills)
                color = '#1976d2'
                html = f"<span style='background-color:{color}; color:white; padding:2px 4px; border-radius:3px;'>{matched_text}</span>"
                replacements.append((s_char, e_char, html))
                for k in range(enum_start, enum_end):
                    occupied_tokens.add(k)
                matched = True
                break
        
        if matched:
            i += 1
            continue
        
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

            # Context validation: ensure the match is used appropriately (prevents false positives)
            # Check if SOFT/RECRUITER keywords are used in proper context (not as incidental modifiers)
            if label in ('SOFT', 'RECRUITER'):
                skip_match = False
                # Single-token checks
                if len(window) == 1:
                    tok = window[0]
                    # Skip if adjective modifying a noun outside the match (e.g., 'quality' in 'quality assurance')
                    if tok.pos_ == 'ADJ' and tok.head != tok and tok.head.pos_ == 'NOUN' and tok.i < tok.head.i:
                        skip_match = True
                    # Skip if verb in dependent clause (not main skill verb)
                    elif tok.pos_ == 'VERB' and tok.dep_ not in ('ROOT', 'conj', 'xcomp', 'ccomp'):
                        skip_match = True
                else:
                    # Multi-token: check if phrase is part of larger compound
                    first_tok = window[0]
                    last_tok = window[-1]
                    # Skip if first token is modifying something outside window
                    if first_tok.pos_ == 'ADJ' and first_tok.head not in window and first_tok.head.pos_ == 'NOUN':
                        skip_match = True
                    # Skip if last token has important children outside window on right
                    if not skip_match:
                        for child in last_tok.children:
                            if child.i > last_tok.i and child not in window and child.dep_ in ('amod', 'compound', 'nmod'):
                                skip_match = True
                                break
                if skip_match:
                    continue

            # compute span chars (original token span)
            s_char = window[0].idx
            e_char = window[-1].idx + len(window[-1].text)

            # extract matched text and trim leading/trailing characters
            matched_text = text[s_char:e_char]

            # find first/last alphanumeric positions and expand to include contiguous allowed symbols
            # By default only allow programming symbols; allow '.' only when not in TOKEN_ALIGNED
            # or when the original keyword explicitly contains a dot (e.g., 'node.js').
            allowed_symbols = set(['+', '#', '-'])
            if (not TOKEN_ALIGNED) or ('.' in (display_kw or '')):
                allowed_symbols.add('.')
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
            # allow matches if at least one token in the window is free
            # (previous behavior blocked the match if any token was occupied).
            # This makes SOFT and RECRUITER keywords more permissive.
            if all(k in occupied_tokens for k in range(i, j+1)):
                continue

            # heuristics per category
            sent_idx = sent_index_by_token.get(i, None)
            # HARD: require noun/proper noun or direct object present in window
            if label == 'HARD':
                if not RELAX_HARD:
                    ok = any(t.pos_ in ('NOUN', 'PROPN') for t in window) or any(t.dep_ == 'dobj' for t in window)
                    if not ok:
                        continue
                    # Additional context validation
                    if not is_valid_context(window, label):
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
            # Escape HTML characters in matched text to prevent injection
            import html as html_module
            matched_text_escaped = html_module.escape(matched_text)
            # build html
            html = f"<span style='background-color:{color}; color:white; padding:2px 4px; border-radius:3px;'>{matched_text_escaped}</span>"
            replacements.append((new_s, new_e, html))
            for k in range(i, j+1):
                occupied_tokens.add(k)
            matched = True
            break
        if not matched:
            i += 1
        else:
            i += 1

    # Escape the entire text first, then apply replacements
    import html as html_module
    out = html_module.escape(text)
    
    # Adjust replacement positions for escaped characters
    # We need to recalculate positions after escaping
    for s, e, html in sorted(replacements, key=lambda x: x[0], reverse=True):
        # Get the original substring and its escaped version
        original_substr = text[:s]
        escaped_substr = html_module.escape(original_substr)
        new_s = len(escaped_substr)
        
        original_substr_end = text[:e]
        escaped_substr_end = html_module.escape(original_substr_end)
        new_e = len(escaped_substr_end)
        
        out = out[:new_s] + html + out[new_e:]
    return out


# Streamlit UI (use this file to validate before replacing main.py)
st.set_page_config(page_title="SkillHighlight Analyzer", layout="wide")

# Sidebar configuration controls
with st.sidebar:
    st.header("⚙️ Configuration")
    
    st.subheader("Matching Settings")
    TOKEN_ALIGNED = st.checkbox("Token-aligned mode", value=TOKEN_ALIGNED, 
                                 help="Highlight whole tokens only (safer for punctuation)")
    
    st.subheader("Heuristic Relaxation")
    st.caption("Enable to make highlighting more permissive")
    RELAX_HARD = st.checkbox("Relax HARD skills", value=RELAX_HARD,
                              help="Allow hard skills without strict noun/object requirements")
    RELAX_ACTION = st.checkbox("Relax ACTION verbs", value=RELAX_ACTION,
                                help="Allow action verbs without direct object requirement")
    RELAX_SOFT = st.checkbox("Relax SOFT skills", value=RELAX_SOFT,
                              help="Skip negative sentiment suppression for soft skills")
    RELAX_RECRUITER = st.checkbox("Relax RECRUITER keywords", value=RELAX_RECRUITER,
                                   help="Allow recruiter keywords in verbless bullets")
    
    st.subheader("Sentiment Threshold")
    SOFT_NEG_THRESHOLD = st.slider("Soft skill sentiment threshold", 
                                     min_value=-1.0, max_value=0.0, value=SOFT_NEG_THRESHOLD, step=0.05,
                                     help="Suppress soft skills below this sentiment score")
    
    st.divider()
    st.caption("💡 Tip: Click the keyword composition bars to toggle highlighting for each category")

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
            if name.endswith('.pdf'):
                # Try PyMuPDF first (better layout preservation), fallback to pdfminer
                try:
                    import fitz  # PyMuPDF
                    pdf_doc = fitz.open(stream=data, filetype="pdf")
                    text_parts = []
                    for page in pdf_doc:
                        # Extract text with layout preservation
                        text_parts.append(page.get_text("text"))
                    pdf_doc.close()
                    text = "\n".join(text_parts)
                    
                    # Clean up excessive whitespace while preserving structure
                    lines = text.split('\n')
                    cleaned_lines = []
                    for line in lines:
                        # Strip trailing/leading whitespace from each line
                        cleaned = line.strip()
                        # Skip completely empty lines, but keep lines with content
                        if cleaned:
                            cleaned_lines.append(cleaned)
                        # Add back single empty line for paragraph breaks (if previous line had content)
                        elif cleaned_lines and cleaned_lines[-1] != '':
                            cleaned_lines.append('')
                    
                    # Join and remove excessive consecutive empty lines
                    text = '\n'.join(cleaned_lines)
                    text = re.sub(r'\n\n+', '\n\n', text)  # Max 1 blank line between paragraphs
                    
                except ImportError:
                    # Fallback to pdfminer if PyMuPDF not available
                    if extract_pdf_text:
                        text = extract_pdf_text(io.BytesIO(data))
                    else:
                        st.warning("Please install PyMuPDF for better PDF extraction: pip install PyMuPDF")
                        text = ""
            elif name.endswith('.docx') and docx2txt:
                with tempfile.NamedTemporaryFile(delete=False, suffix='.docx') as tmp:
                    tmp.write(data)
                    tmp_path = tmp.name
                try:
                    text = docx2txt.process(tmp_path)
                    # Remove any HTML/XML tags and formatting codes that might be embedded
                    text = re.sub(r'<[^>]+>', '', text)  # Remove HTML/XML tags
                    text = re.sub(r'&[a-zA-Z]+;', '', text)  # Remove HTML entities like &nbsp;
                    # Clean up any stray angle brackets
                    text = text.replace('<', '').replace('>', '')
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
    # Initialize disabled classifiers in session state
    if 'disabled_classifiers' not in st.session_state:
        st.session_state.disabled_classifiers = set()
    
    DISABLED_BARS = st.session_state.disabled_classifiers
    
    hard_pct, soft_pct, rec_pct, action_pct = count_keywords(text)
    results, avg_score = analyze_self_promotion(text)

    col1, col2 = st.columns([2, 1])
    with col1:
        st.subheader("Extracted / Input Text")
        # Preserve line breaks and prevent markdown interpretation
        highlighted_text = highlight_keywords(text, disabled_labels=DISABLED_BARS)
        # Use st.html to render only HTML, not markdown (prevents # becoming headers, ** becoming bold)
        st.html(f"<div style='line-height:1.6; font-size:16px; white-space:pre-wrap; font-family: monospace;'>{highlighted_text}</div>")
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
