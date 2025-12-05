"""Keyword highlighting with context validation"""
import re
from textblob import TextBlob


def highlight_keywords(nlp, text: str, hard_skills: list, soft_skills: list,
                      recruiter_keywords: list, action_verbs: list,
                      disabled_labels=None, token_aligned=True, relax_hard=False,
                      relax_action=False, relax_soft=True, relax_recruiter=True,
                      soft_neg_threshold=-0.1, render_legacy=False) -> str:
    """Highlight keywords in text with context validation"""
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

    # Context validation function
    def is_valid_context(window, label):
        """Validate that the matched window is used in appropriate context"""
        # HARD skills: Check for date/time context patterns
        if label == 'HARD':
            for tok in window:
                nearby_tokens = []
                for offset in range(-3, 4):
                    idx = tok.i + offset
                    if 0 <= idx < len(tok.doc) and idx != tok.i:
                        nearby_tokens.append(tok.doc[idx])
                
                has_year_nearby = any(t.text.isdigit() and len(t.text) == 4 and 1900 <= int(t.text) <= 2100 
                                     for t in nearby_tokens if t.text.isdigit())
                if has_year_nearby:
                    non_tech_in_dates = {'spring', 'summer', 'fall', 'winter', 'may', 'march', 'august'}
                    if tok.text.lower() in non_tech_in_dates:
                        return False
        
        # Single-token checks
        if len(window) == 1:
            tok = window[0]
            if tok.pos_ == 'ADJ' and tok.head != tok and tok.head.pos_ == 'NOUN' and tok.i < tok.head.i:
                return False
            if tok.pos_ == 'VERB' and tok.dep_ not in ('ROOT', 'conj', 'xcomp', 'ccomp'):
                return False
            
            if label == 'HARD':
                for child in tok.children:
                    if child not in window and child.dep_ in ('compound', 'amod', 'nmod'):
                        if child.i > tok.i:
                            return False
                if tok.dep_ in ('amod', 'compound') and tok.head not in window:
                    return False
        
        # Multi-token checks
        first_tok = window[0]
        last_tok = window[-1]
        
        if first_tok.pos_ == 'ADJ' and first_tok.head not in window and first_tok.head.pos_ == 'NOUN':
            return False
        
        for child in last_tok.children:
            if child.i > last_tok.i and child not in window and child.dep_ in ('amod', 'compound', 'nmod'):
                return False
        
        return True

    # Build keyword maps
    all_keywords = []
    for k in recruiter_keywords:
        if k:
            all_keywords.append(("RECRUITER", k))
    for k in soft_skills:
        if k:
            all_keywords.append(("SOFT", k))
    for k in hard_skills:
        if k:
            all_keywords.append(("HARD", k))
    for k in action_verbs:
        if k:
            all_keywords.append(("ACTION", k))

    seen = set()
    uniq_keywords = []
    for label, kw in all_keywords:
        kn = kw.strip()
        key = (label, kn.lower())
        if key in seen:
            continue
        seen.add(key)
        uniq_keywords.append((label, kn))

    # Process keywords via nlp.pipe
    map_by_tuple = {}
    map_by_joined = {}
    max_kw_len = 1
    if uniq_keywords:
        docs = list(nlp.pipe([kw for _, kw in uniq_keywords]))
        for (label, kw), kdoc in zip(uniq_keywords, docs):
            toks = [t for t in kdoc if not t.is_space]
            lem_tup = tuple(t.lemma_.lower() for t in toks) if toks else (kw.lower(),)
            joined = ''.join(re.sub(r"\W+", "", t.text.lower()) for t in toks)
            if not joined and not any(re.search(r"\w", lt) for lt in lem_tup):
                continue
            joined_key = joined if joined else ''.join(re.sub(r"\W+", "", kw.lower()))
            map_by_tuple[lem_tup] = (label, kw)
            if joined_key:
                map_by_joined[joined_key] = (label, kw)
            if len(lem_tup) > max_kw_len:
                max_kw_len = len(lem_tup)

    # Parse document
    doc = nlp(text)
    tokens = [tok for tok in doc]
    n = len(tokens)

    # Prepare sentence-level caches
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
        for si, s in enumerate(sents):
            if tok.idx >= s.start_char and tok.idx < s.end_char:
                sent_index_by_token[idx] = si
                break

    occupied_tokens = set()
    replacements = []

    # Detect skill enumerations
    def find_skill_enumerations():
        enum_spans = []
        for sent in sents:
            sent_tokens = [t for t in sent if t.i < n]
            for i, tok in enumerate(sent_tokens):
                if tok.lemma_.lower() in ('skill', 'ability', 'quality', 'strength', 'competency', 'expertise'):
                    start_idx = None
                    for j in range(i-1, max(-1, i-10), -1):
                        t = sent_tokens[j]
                        if t.pos_ in ('NOUN', 'ADJ', 'PROPN') and (j == 0 or sent_tokens[j-1].pos_ not in ('NOUN', 'ADJ', 'PROPN', 'CCONJ', 'PUNCT')):
                            has_conj = any(sent_tokens[k].pos_ in ('CCONJ',) or sent_tokens[k].text in (',', 'and', 'or') for k in range(j, i))
                            if has_conj:
                                start_idx = sent_tokens[j].i
                                break
                    if start_idx is not None:
                        enum_spans.append((start_idx, tok.i + 1))
        return enum_spans
    
    skill_enum_spans = find_skill_enumerations()

    # Main matching loop
    i = 0
    while i < n:
        if i in occupied_tokens:
            i += 1
            continue
        matched = False
        
        # Check skill enumerations first
        for enum_start, enum_end in skill_enum_spans:
            if i == enum_start and not any(k in occupied_tokens for k in range(enum_start, enum_end)):
                window = tokens[enum_start:enum_end]
                s_char = window[0].idx
                e_char = window[-1].idx + len(window[-1].text)
                matched_text = text[s_char:e_char].strip()
                color = '#7e57c2'
                html = f"<span style='background-color:{color}; color:white; padding:2px 4px; border-radius:3px;'>{matched_text}</span>"
                replacements.append((s_char, e_char, html))
                for k in range(enum_start, enum_end):
                    occupied_tokens.add(k)
                matched = True
                break
        
        if matched:
            i += 1
            continue
        
        # Try longest-first windows
        for L in range(min(max_kw_len, n - i), 0, -1):
            j = i + L - 1
            window = tokens[i:j+1]
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

            # Skip if disabled
            if label in disabled_labels:
                continue

            # Context validation for SOFT/RECRUITER
            if label in ('SOFT', 'RECRUITER'):
                skip_match = False
                if len(window) == 1:
                    tok = window[0]
                    if tok.pos_ == 'ADJ' and tok.head != tok and tok.head.pos_ == 'NOUN' and tok.i < tok.head.i:
                        if not (label == 'SOFT' and relax_soft):
                            skip_match = True
                    elif tok.pos_ == 'VERB' and tok.dep_ not in ('ROOT', 'conj', 'xcomp', 'ccomp'):
                        skip_match = True
                else:
                    first_tok = window[0]
                    last_tok = window[-1]
                    if first_tok.pos_ == 'ADJ' and first_tok.head not in window and first_tok.head.pos_ == 'NOUN':
                        if not (label == 'SOFT' and relax_soft):
                            skip_match = True
                    if not skip_match:
                        for child in last_tok.children:
                            if child.i > last_tok.i and child not in window and child.dep_ in ('amod', 'compound', 'nmod'):
                                if not (label == 'SOFT' and relax_soft):
                                    skip_match = True
                                break
                if skip_match:
                    continue

            # Compute span
            s_char = window[0].idx
            e_char = window[-1].idx + len(window[-1].text)
            matched_text = text[s_char:e_char]

            # Trim to alphanumeric boundaries
            allowed_symbols = set(['+', '#', '-'])
            if (not token_aligned) or ('.' in (display_kw or '')):
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
            
            first_i = first_alnum
            while first_i > 0 and matched_text[first_i-1] in allowed_symbols:
                first_i -= 1
            last_i = last_alnum
            while last_i+1 < len(matched_text) and matched_text[last_i+1] in allowed_symbols:
                last_i += 1

            # Clamp symbol runs
            allowed_counts = {'+': 2, '#': 1, '.': 1, '-': 1}
            if last_i > last_alnum:
                run_char = matched_text[last_alnum+1]
                if run_char in allowed_counts:
                    run_len = 0
                    p = last_alnum + 1
                    while p < len(matched_text) and matched_text[p] == run_char:
                        run_len += 1
                        p += 1
                    max_allowed = allowed_counts.get(run_char, 1)
                    if run_len > max_allowed:
                        last_i = last_alnum + max_allowed
            
            if first_i < first_alnum:
                run_char = matched_text[first_i]
                if run_char in allowed_counts:
                    run_len = 0
                    p = first_i
                    while p < first_alnum and matched_text[p] == run_char:
                        run_len += 1
                        p += 1
                    max_allowed = allowed_counts.get(run_char, 1)
                    if run_len > max_allowed:
                        first_i = first_alnum - max_allowed

            new_s = s_char + first_i
            new_e = s_char + last_i + 1
            
            if new_s >= new_e:
                continue
            if overlaps((new_s, new_e)):
                continue
            if all(k in occupied_tokens for k in range(i, j+1)):
                continue

            # Apply heuristics per category
            sent_idx = sent_index_by_token.get(i, None)
            
            if label == 'HARD':
                if not relax_hard:
                    # Check if it's an acronym (all caps, 2-5 chars) - always accept
                    is_acronym = any(t.text.isupper() and 2 <= len(t.text) <= 5 for t in window)
                    # Check if it contains special chars like C++, C#, .NET
                    has_special = any(c in display_kw for c in ['+', '#', '.'])
                    # Check for standard technical term patterns
                    ok = (is_acronym or has_special or 
                          any(t.pos_ in ('NOUN', 'PROPN', 'X') for t in window) or 
                          any(t.dep_ in ('dobj', 'pobj', 'attr', 'appos', 'compound') for t in window))
                    if not ok:
                        continue
                    if not is_acronym and not has_special and not is_valid_context(window, label):
                        continue
                color = '#26a69a'
            
            elif label == 'SOFT':
                if relax_soft:
                    color = '#7e57c2'
                else:
                    neg = False
                    boost = False
                    if sent_idx is not None:
                        pol = sent_polarity.get(sent_idx, 0.0)
                        if pol < soft_neg_threshold:
                            neg = True
                        if sent_idx < 3:
                            boost = True
                    if neg:
                        continue
                    color = '#7e57c2' if not boost else '#5e35b1'
            
            elif label == 'ACTION':
                if not relax_action:
                    is_valid_action = False
                    for t in window:
                        if t.pos_ == 'VERB':
                            # Accept verbs with direct objects (dobj)
                            if any(c.dep_ == 'dobj' for c in t.children):
                                is_valid_action = True
                                break
                            # Accept verbs with prepositional objects (pobj via prep)
                            if any(c.dep_ in ('prep', 'agent') for c in t.children):
                                is_valid_action = True
                                break
                            # Accept ROOT verbs (main verb of sentence)
                            if t.dep_ == 'ROOT':
                                is_valid_action = True
                                break
                            # Accept sentence-initial verbs (common in resume bullets)
                            if t.i == 0 or (t.i > 0 and t.doc[t.i-1].text in ('•', '-', '*', '·', '►', '●')):
                                is_valid_action = True
                                break
                            # Accept conjoined verbs (e.g., "developed and deployed")
                            if t.dep_ in ('conj', 'xcomp', 'ccomp', 'advcl'):
                                is_valid_action = True
                                break
                    if not is_valid_action:
                        continue
                color = '#ef5350'
            
            else:  # RECRUITER
                if not relax_recruiter and not sent_has_verb.get(sent_idx, True):
                    downgraded = True
                else:
                    downgraded = False
                color = '#ff9f43' if not downgraded else '#ffd79d'

            # Build HTML
            matched_text = text[new_s:new_e]
            if render_legacy:
                html = f"<span style='background-color:{color}; color:white;'>{matched_text}</span>"
            else:
                html = f"<span style='background-color:{color}; color:white; padding:2px 4px; border-radius:3px;'>{matched_text}</span>"
            
            replacements.append((new_s, new_e, html))
            for k in range(i, j+1):
                occupied_tokens.add(k)
            matched = True
            break
        
        if not matched:
            i += 1
        else:
            i += 1

    # Apply replacements
    out = text
    for s, e, html in sorted(replacements, key=lambda x: x[0], reverse=True):
        out = out[:s] + html + out[e:]
    return out
