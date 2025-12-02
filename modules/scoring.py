"""Self-promotion scoring using KNN and heuristics"""
import streamlit as st
import numpy as np
import pandas as pd
import os
import joblib
import re
from sklearn.neighbors import KNeighborsClassifier
from textblob import TextBlob


@st.cache_resource
def load_knn_model(_bert_model):
    """Load or train KNN model for self-promotion scoring"""
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


def has_metric(sentence: str) -> bool:
    """Check if sentence contains metrics/numbers"""
    return bool(re.search(r"(\b\d+%|\b\d{1,3}\b|\$\d+[kKmM]?)", sentence))


def get_base_score(knn_model, _bert_model, sentence: str) -> float:
    """Get base KNN self-promotion score"""
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


def smart_sentence_split(nlp, text: str) -> list:
    """Split resume text into sentences, respecting paragraph boundaries from extraction"""
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


def analyze_sentences(nlp, knn_model, _bert_model, text: str, action_verbs: list, file_type=None):
    """Analyze all sentences in text and return scored results"""
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
            words = line.split()
            # Must be short and not have sentence ending
            is_short = len(words) <= 4
            no_end_punct = not re.search(r'[.!?]$', line)
            not_bullet = not re.match(r'^[-•●○▪▫◦⦿⦾]', line)
            # Headers often end with colon or are ALL CAPS
            ends_colon = line.endswith(':')
            is_caps = len(words) >= 2 and line.isupper()
            
            return (is_short and no_end_punct and not_bullet) or ends_colon or is_caps
        
        # Helper to split multi-sentence content while protecting terms
        def smart_split_sentences(content):
            """Split content into sentences, protecting technical terms"""
            # Don't split if it's a bullet point with only one sentence
            # Check if content has clear sentence boundaries (. ! ? followed by space and capital letter)
            has_multiple = len(re.findall(r'[.!?]\s+[A-Z]', content)) >= 1
            
            if not has_multiple:
                # Single sentence or fragment, return as-is
                if len(content) >= 10:
                    return [content]
                return []
            
            # Temporarily replace protected terms
            temp_content = content
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
            # This is more reliable than spaCy for resume text
            parts = re.split(r'(?<=[.!?])\s+(?=[A-Z])', temp_content)
            
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
            # PDF and DOCX: Both split by single newlines after normalization
            # Bullets are already normalized to • so treat both the same way
            lines = text.split('\n')
            
            for line in lines:
                line = line.strip()
                if not line or len(line) < 10:
                    continue
                
                # Skip headers entirely - don't score them
                if is_header(line):
                    continue
                
                # Keep each line as-is (already separated by extraction/paragraphs)
                # Only split if there are genuinely multiple complete sentences
                sentence_count = len(re.findall(r'[.!?]\s+[A-Z]', line))
                
                if sentence_count >= 1:
                    # Has multiple sentences - split them
                    split_sents = smart_split_sentences(line)
                    if split_sents:
                        sentences.extend(split_sents)
                    else:
                        # Fallback: keep original if split failed
                        sentences.append(line)
                else:
                    # Single sentence or bullet - keep as-is
                    sentences.append(line)
        
        elif file_type == 'txt' or file_type == 'text_input':
            # TXT/text input: Use spaCy sentence detection with protection
            split_sents = smart_split_sentences(text)
            sentences.extend(split_sents)
        
        else:
            # Unknown type: fallback to spaCy with protection
            split_sents = smart_split_sentences(text)
            sentences.extend(split_sents)
        
        for stxt in sentences:
            if not stxt:
                continue
            stxt_lower = stxt.lower()
            base = get_base_score(knn_model, _bert_model, stxt)
            metric_bonus = 0.15 if has_metric(stxt) else 0.0
            achievement_bonus = 0.12 if detect_achievement_pattern(stxt_lower) else 0.0
            bullet_bonus = 0.08 if is_bullet_point(stxt) else 0.0
            try:
                pol = TextBlob(stxt).sentiment.polarity
            except Exception:
                pol = 0.0
            pol_boost = 0.06 if pol > 0.15 else 0.0
            action_verb_bonus = 0.0
            if action_verbs:
                first_word = stxt_lower.split()[0] if stxt_lower else ''
                if any(first_word.startswith(v.lower()) for v in action_verbs[:50]):
                    action_verb_bonus = 0.08
            length_penalty = -0.1 if len(stxt.split()) < 5 else 0.0
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


def fallback_segment(text: str) -> list:
    """Gentle sentence splitter for poorly formatted resumes"""
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


def analyze_with_fallback(knn_model, _bert_model, text: str, action_verbs: list, max_sentences: int = None):
    """Analyze using fallback segmentation with full scoring"""
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
