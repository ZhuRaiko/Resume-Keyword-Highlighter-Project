################################################################################
# Module: scoring.py
#
# What this module does:
#   Implements sentence-level self-promotion scoring for résumé text. It provides
#   utilities to load or train a KNN classifier (using BERT embeddings), split
#   résumé text into sentences robustly across file formats, compute an ML-based
#   base score per sentence, and combine that with rule-based heuristics
#   (metrics, achievement patterns, bullets, sentiment, action verbs) to produce
#   a final per-sentence score and an average score for a document.
#
# Why this module is necessary in the overall system:
#   The larger system needs a deterministic way to quantify "self-promotion"
#   signals in resume text. This module centralizes that logic so the main app
#   can request per-sentence scores (and an aggregate) for ranking, highlighting,
#   and downstream evaluation.
#
# How this module connects to other parts of the NLP/ML pipeline:
#   - It expects a pre-loaded BERT embedder to convert sentences into vectors.
#   - It may load or train a KNN model from labeled examples stored in data/.
#   - It uses spaCy (via a passed `nlp`) for sentence splitting and TextBlob for
#     a lightweight sentiment signal. The main app calls these functions when
#     processing extracted resume text.
################################################################################
"""
Self-promotion scoring using KNN and heuristics.

This module provides functions to:
- Load or train a KNN classifier for self-promotion detection.
- Split résumé text into sentences, handling various file formats and edge cases.
- Score each sentence using a combination of KNN probability and rule-based heuristics.
- Provide fallback mechanisms for poorly formatted input.

Intended for use in the SkillHighlight system.
"""
import streamlit as st
import numpy as np
import pandas as pd
import os
import joblib
import re
from sklearn.neighbors import KNeighborsClassifier
from textblob import TextBlob


# Function: load_knn_model
# What: Load or train a KNN classifier using saved model or CSV dataset.
# Why: Ensure there is a trained KNN model available for scoring sentences.
# Inputs: _bert_model - an object exposing an `encode(list_of_sentences)` method.
# Returns: a fitted KNeighborsClassifier, a Dummy fallback, or None on failure.
# How it contributes: Supplies the ML component used by get_base_score for per-sentence probabilities.
@st.cache_resource
def load_knn_model(_bert_model):
    """
    Load a pre-trained KNN model from disk, or train a new one if not found.

    - If a model file exists, it loads and returns it.
    - If not, it reads the labeled dataset, generates embeddings, trains a new KNN, and saves it.
    - Returns a dummy model if data is missing or loading fails.

    Args:
        _bert_model: Pre-loaded BERT embedding model.

    Returns:
        Trained KNN model or dummy fallback.
    """
    model_path = "models/knn_model.pkl"
    csv_path = "data/self_promotion_dataset.csv"
    
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
        X = _bert_model.encode(df["sentence"].tolist())
        y = df["label"].astype(int).values
        knn = KNeighborsClassifier(n_neighbors=5)
        knn.fit(X, y)
        joblib.dump(knn, model_path)
        return knn
    except Exception:
        return None


# Function: has_metric
# What: Detect numeric metrics, percentages, or monetary figures in a sentence.
# Why: Quantified achievements often indicate stronger self-promotion; this
#      flag is used to add a heuristic bonus to the sentence score.
# Inputs: sentence (str)
# Returns: bool indicating whether numeric metrics are present
# How it contributes: Adds a metric-based boost to final sentence scoring.
def has_metric(sentence: str) -> bool:
    """
    Check if a sentence contains metrics or numbers (e.g., percentages, counts, money).
    Used to boost the score for quantified achievements.
    """
    return bool(re.search(r"(\b\d+%|\b\d{1,3}\b|\$\d+[kKmM]?)", sentence))


# Function: get_base_score
# What: Compute a base ML score (probability) using the KNN model and BERT embeddings.
# Why: Provides an ML-derived probability of a sentence being self-promotional.
# Inputs: knn_model, _bert_model, sentence (str)
# Returns: float probability in [0.0, 1.0]; 0.0 on failure or missing model
# How it contributes: Serves as the core signal which heuristics then augment.
def get_base_score(knn_model, _bert_model, sentence: str) -> float:
    """
    Get the base self-promotion score for a sentence using the KNN model.
    Returns the probability of the sentence being self-promotional.
    Falls back to 0.0 if the model is missing or prediction fails.
    """
    try:
        vec = _bert_model.encode([sentence])
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


# Function: smart_sentence_split
# What: Segment resume text into a list of substantial sentences using spaCy.
# Why: Resumes can contain paragraphs and bullets; this helper preserves
#      paragraph boundaries and filters out fragments to produce scoring candidates.
# Inputs: nlp (spaCy language model), text (str)
# Returns: list of sentence strings suitable for scoring
# How it contributes: Prepares clean sentence inputs for get_base_score and heuristics.
def smart_sentence_split(nlp, text: str) -> list:
    """
    Split resume text into sentences, respecting paragraph boundaries from extraction.
    Uses spaCy for sentence detection within each paragraph.
    Filters out very short or fragmentary sentences.
    """
    if not text:
        return []
    
    sentences = []
    
    # Split by paragraphs (preserved from extraction)
    paragraphs = text.split('\n\n')  # Double newline = paragraph break
    
    for para in paragraphs:
        para = para.strip()
        if not para or len(para) < 10:
            continue
        
        # Use spaCy to split into sentences
        doc = nlp(para)
        for sent in doc.sents:
            s = sent.text.strip()
            # Keep if substantial
            if len(s) >= 10 and len(s.split()) >= 2:
                sentences.append(s)
    
    return sentences


# Function: analyze_sentences
# What: Score all sentences in provided text using ML + heuristic rules.
# Why: Central scoring routine used by the main app to get per-sentence and
#      aggregate self-promotion signals.
# Inputs: nlp (spaCy model), knn_model, _bert_model, text (str), action_verbs (list), file_type (optional)
# Returns: tuple (list_of_(sentence,score), average_score)
# How it contributes: Produces the primary outputs used by highlighting and metrics.
def analyze_sentences(nlp, knn_model, _bert_model, text: str, action_verbs: list, file_type=None):
    """
    Analyze all sentences in the input text and return a list of (sentence, score) pairs.

    - Handles splitting for different file types (PDF, DOCX, TXT).
    - Applies KNN and heuristic scoring to each sentence.
    - Heuristics include bonuses for metrics, achievement patterns, bullets, sentiment, and action verbs.
    - Penalizes very short sentences.

    Args:
        nlp: spaCy language model for sentence splitting.
        knn_model: Trained KNN classifier.
        _bert_model: BERT embedding model.
        text: Input résumé text.
        action_verbs: List of action verbs for scoring.
        file_type: File type hint for splitting logic.

    Returns:
        List of (sentence, score) and average score.
    """
    achievement_words = {'achieved', 'delivered', 'improved', 'increased', 'reduced', 'led', 'managed', 
                        'developed', 'created', 'launched', 'implemented', 'drove', 'exceeded', 'optimized',
                        'streamlined', 'established', 'built', 'designed', 'spearheaded', 'pioneered'}
    impact_words = {'resulting', 'leading to', 'by', 'saved', 'generated', 'boosted', 'enhanced'}
    
    def detect_achievement_pattern(stxt_lower):
        """
        Detect strong achievement patterns like 'Achieved X by doing Y'.
        Returns True if both an achievement verb and an impact word are present.
        """
        has_achievement = any(w in stxt_lower for w in achievement_words)
        has_impact = any(w in stxt_lower for w in impact_words)
        return has_achievement and has_impact
    
    def is_bullet_point(stxt):
        """
        Check if a sentence starts with a bullet or list marker (e.g., '-', '•', '1.').
        Used to give a bonus for bullet-pointed achievements.
        """
        return bool(re.match(r'^\s*[-•●○▪▫◦⦿⦾]\s+|^\s*\d+[\.)]\s+', stxt))
    
    sents = []
    try:
        sentences = []
        
        # Protected terms that should not be split
        protected_terms = {
            '.net', 'node.js', 'next.js', 'express.js', 'socket.io', 'd3.js', 'vue.js',
            'b.s.', 'm.s.', 'ph.d.', 'b.a.', 'm.a.', 'm.b.a.', 
            'inc.', 'corp.', 'ltd.', 'co.', 'dr.', 'mr.', 'mrs.', 'ms.',
            'u.s.', 'u.k.', 'e.g.', 'i.e.', 'etc.', 'vs.', 'c#', 'pl/sql'
        }
        
        # Helper to detect headers (should be filtered out, not scored)
        def is_header(line):
            """
            Detect section headers (e.g., 'EXPERIENCE:', 'EDUCATION') that should not be scored.
            Looks for short, non-sentence lines, colons, or ALL CAPS.
            """
            words = line.split()
            is_short = len(words) <= 4
            no_end_punct = not re.search(r'[.!?]$', line)
            not_bullet = not re.match(r'^[-•●○▪▫◦⦿⦾]', line)
            ends_colon = line.endswith(':')
            is_caps = len(words) >= 2 and line.isupper()
            return (is_short and no_end_punct and not_bullet) or ends_colon or is_caps
        
        # Helper to split multi-sentence content while protecting terms
        def smart_split_sentences(content):
            """
            Split content into sentences, protecting technical terms and removing bullet markers.
            Handles edge cases like abbreviations and technical terms (e.g., '.NET', 'Ph.D.').
            Uses regex to split on sentence boundaries, but avoids splitting protected terms.
            """
            # Remove bullet markers and leading dashes from the beginning for clean splitting
            # This allows us to split the content itself without the bullet interfering
            content_clean = re.sub(r'^\s*[-•●○▪▫◦⦿⦾]\s+', '', content)
            # Also remove standalone dash at the start
            content_clean = re.sub(r'^\s*-\s+', '', content_clean)
            
            # Check if content has clear sentence boundaries (. ! ? followed by space and capital letter)
            # BUT ignore patterns where the capital letter is part of a grade
            
            if not content_clean or len(content_clean) < 10:
                return []
            
            # Count sentence boundaries in the cleaned content
            # Look for ". A" patterns but exclude grade patterns like "A's"
            sentence_boundaries = []
            for match in re.finditer(r'[.!?]\s+[A-Z]', content_clean):
                # Check if it's part of grade notation
                pos = match.start()
                context = content_clean[max(0, pos-5):min(len(content_clean), pos+10)]
                # Skip if it's a grade pattern
                if re.search(r'\d+\s*A[\'s]*', context):
                    continue
                sentence_boundaries.append(match)
            
            has_multiple = len(sentence_boundaries) >= 1
            
            if not has_multiple:
                # Single sentence or fragment, return as-is (without bullet)
                return [content_clean]
            
            # Temporarily replace protected terms in cleaned content
            temp_content = content_clean
            replacements = {}
            for i, term in enumerate(protected_terms):
                if term.lower() in temp_content.lower():
                    placeholder = f"__PROTECTED_{i}__"
                    # Case-insensitive replacement
                    pattern = re.compile(re.escape(term), re.IGNORECASE)
                    temp_content = pattern.sub(placeholder, temp_content)
                    replacements[placeholder] = term
            
            # Use regex-based splitting for more reliable sentence boundaries
            # Split on . ! ? followed by whitespace and capital letter
            # BUT not if followed by bullet marker
            parts = re.split(r'(?<=[.!?])\s+(?=[A-Z](?![•●○▪▫◦⦿⦾]))', temp_content)
            
            results = []
            for part in parts:
                s = part.strip()
                # Restore protected terms
                for placeholder, original in replacements.items():
                    s = s.replace(placeholder, original)
                # Only keep if it's substantial enough
                if len(s) >= 10 and len(s.split()) >= 2:
                    results.append(s)
            
            return results
        
        if file_type in ['pdf', 'docx']:
            # For PDF and DOCX: Split by single newlines (each line is a candidate sentence)
            # Bullets are already normalized to • so treat both the same way
            lines = text.split('\n')
            for line in lines:
                line = line.strip()
                if not line or len(line) < 10:
                    continue
                # Skip headers entirely - don't score them
                if is_header(line):
                    continue
                # Only split if there are genuinely multiple complete sentences
                sentence_count = len(re.findall(r'[.!?]\s+[A-Z]', line))
                if sentence_count >= 1:
                    # Has multiple sentences - split them (bullets will be removed in splitting)
                    split_sents = smart_split_sentences(line)
                    if split_sents:
                        sentences.extend(split_sents)
                    else:
                        # Fallback: keep original with bullet and dash removed
                        clean_line = re.sub(r'^\s*[-•●○▪▫◦⦿⦾]\s+', '', line)
                        clean_line = re.sub(r'^\s*-\s+', '', clean_line)
                        if clean_line:
                            sentences.append(clean_line)
                else:
                    # Single sentence - remove bullet, dash and keep content
                    clean_line = re.sub(r'^\s*[-•●○▪▫◦⦿⦾]\s+', '', line)
                    clean_line = re.sub(r'^\s*-\s+', '', clean_line)
                    if clean_line:
                        sentences.append(clean_line)
        elif file_type == 'txt' or file_type == 'text_input':
            # For TXT/text input: Use spaCy sentence detection with protection
            split_sents = smart_split_sentences(text)
            sentences.extend(split_sents)
        else:
            # Unknown type: fallback to spaCy with protection
            split_sents = smart_split_sentences(text)
            sentences.extend(split_sents)
        
        # Score each sentence using KNN and heuristics
        for stxt in sentences:
            if not stxt:
                continue
            # Normalize whitespace and punctuation for consistent scoring
            stxt = re.sub(r'\s+', ' ', stxt).strip()
            stxt = re.sub(r'\s+([.,;:!?])', r'\1', stxt)
            stxt_lower = stxt.lower()
            base = get_base_score(knn_model, _bert_model, stxt)
            # Heuristic bonuses/penalties:
            metric_bonus = 0.15 if has_metric(stxt) else 0.0  # Bonus for metrics/numbers
            achievement_bonus = 0.12 if detect_achievement_pattern(stxt_lower) else 0.0  # Bonus for achievement pattern
            bullet_bonus = 0.08 if is_bullet_point(stxt) else 0.0  # Bonus for bullet points
            try:
                pol = TextBlob(stxt).sentiment.polarity
            except Exception:
                pol = 0.0
            pol_boost = 0.06 if pol > 0.15 else 0.0  # Bonus for positive sentiment
            action_verb_bonus = 0.0
            if action_verbs:
                first_word = stxt_lower.split()[0] if stxt_lower else ''
                if any(first_word.startswith(v.lower()) for v in action_verbs[:50]):
                    action_verb_bonus = 0.08  # Bonus for starting with action verb
            length_penalty = -0.1 if len(stxt.split()) < 5 else 0.0  # Penalty for very short sentences
            # Combine all scores and clamp between 0 and 1
            score = base + metric_bonus + achievement_bonus + bullet_bonus + pol_boost + action_verb_bonus + length_penalty
            score = min(1.0, max(0.0, score))
            sents.append((stxt, score))
    except Exception:
        # fallback: split on periods and bullets
        parts = [p.strip() for p in re.split(r"(?<=[.!?])\s+|(?<=^)\s*[-•●]\s+", text, flags=re.MULTILINE) if p.strip()]
        for p in parts:
            base = get_base_score(knn_model, _bert_model, p)
            metric_bonus = 0.15 if has_metric(p) else 0.0
            sents.append((p, min(1.0, base + metric_bonus)))
    
    avg = float(np.mean([s for _, s in sents])) if sents else 0.0
    return sents, avg


# Function: fallback_segment
# What: A gentle, robust sentence splitter for very messy or unstructured input.
# Why: When spaCy splitting and other heuristics fail, this provides a simpler
#      segmentation that protects common technical abbreviations.
# Inputs: text (str)
# Returns: list of cleaned sentence-like strings
# How it contributes: Supplies fallback sentences for analyze_with_fallback.
def fallback_segment(text: str) -> list:
    """
    Gentle sentence splitter for poorly formatted resumes.
    - Protects technical terms and common abbreviations from being split.
    - Removes unwanted characters and splits on sentence-ending punctuation.
    - Used when standard splitting fails or input is messy.
    """
    if not text:
        return []
    
    # Protect technical terms and common abbreviations
    protected_terms = {
        ".net", "node.js", "next.js", "express.js", "socket.io", "d3.js", "vue.js",
        "b.s.", "m.s.", "ph.d.", "b.a.", "m.a.", "m.b.a.", 
        "inc.", "corp.", "ltd.", "co.", "dr.", "mr.", "mrs.", "ms.",
        "u.s.", "u.k.", "e.g.", "i.e.", "etc.", "vs."
    }
    
    tmp = text
    for term in protected_terms:
        tmp = re.sub(re.escape(term), term.replace('.', '•DOT•'), tmp, flags=re.IGNORECASE)
    
    # Remove bullet points and special characters (except sentence punctuation, @, commas, hyphens in words)
    # Keep: letters, numbers, spaces, . ! ? , @ - / ( ) and newlines
    tmp = re.sub(r'[^\w\s.,!?@\-/()\n]', ' ', tmp)
    
    # Replace multiple newlines with single space
    tmp = re.sub(r'\n+', ' ', tmp)
    
    # Replace multiple spaces with single space
    tmp = re.sub(r'\s+', ' ', tmp)
    
    # Split on sentence-ending punctuation followed by space
    chunks = []
    for s in re.split(r"(?<=[.!?])\s+", tmp):
        s = s.strip()
        if not s:
            continue
        
        # Restore protected dots
        s = s.replace('•DOT•', '.')
        
        # Filter out very short fragments (likely not real sentences)
        if len(s) >= 10:
            chunks.append(s)
    
    # De-duplicate consecutive identical lines
    out = []
    prev = None
    for c in chunks:
        if c != prev:
            out.append(c)
        prev = c
    
    return out


# Function: analyze_with_fallback
# What: Use fallback_segment and apply the same scoring heuristics to produce scores.
# Why: Provide robustness for very messy input where the primary splitter fails.
# Inputs: knn_model, _bert_model, text (str), action_verbs (list), max_sentences (optional int)
# Returns: tuple (list_of_(sentence,score), average_score)
# How it contributes: Ensures scoring is available even for poorly formatted resumes.
def analyze_with_fallback(knn_model, _bert_model, text: str, action_verbs: list, max_sentences: int = None):
    """
    Analyze using fallback segmentation with full scoring.
    - Uses fallback_segment to split text into sentences.
    - Applies KNN and all heuristic bonuses/penalties to each sentence.
    - Returns a list of (sentence, score) and the average score.
    """
    fb_sents = fallback_segment(text)
    if not fb_sents:
        return [], 0.0
    
    # Apply sentence limit if specified
    if max_sentences and max_sentences > 0:
        fb_sents = fb_sents[:max_sentences]
    
    achievement_words = {'achieved', 'delivered', 'improved', 'increased', 'reduced', 'led', 'managed', 
                        'developed', 'created', 'launched', 'implemented', 'drove', 'exceeded', 'optimized',
                        'streamlined', 'established', 'built', 'designed', 'spearheaded', 'pioneered'}
    impact_words = {'resulting', 'leading to', 'by', 'saved', 'generated', 'boosted', 'enhanced'}
    
    scored = []
    for s in fb_sents:
        s_lower = s.lower()
        base = get_base_score(knn_model, _bert_model, s)
        metric_bonus = 0.15 if has_metric(s) else 0.0
        has_achievement = any(w in s_lower for w in achievement_words)
        has_impact = any(w in s_lower for w in impact_words)
        achievement_bonus = 0.12 if (has_achievement and has_impact) else 0.0
        bullet_bonus = 0.08 if re.match(r'^\s*[-•●○▪▫◦⦿⦾]\s+|^\s*\d+[\.)]\s+', s) else 0.0
        try:
            pol = TextBlob(s).sentiment.polarity
        except Exception:
            pol = 0.0
        pol_boost = 0.06 if pol > 0.15 else 0.0
        action_verb_bonus = 0.0
        if action_verbs:
            first_word = s_lower.split()[0] if s_lower else ''
            if any(first_word.startswith(v.lower()) for v in action_verbs[:50]):
                action_verb_bonus = 0.08
        length_penalty = -0.1 if len(s.split()) < 5 else 0.0
        score = base + metric_bonus + achievement_bonus + bullet_bonus + pol_boost + action_verb_bonus + length_penalty
        score = min(1.0, max(0.0, score))
        scored.append((s, score))
    
    avg = np.mean([s for _, s in scored]) if scored else 0.0
    return scored, avg
