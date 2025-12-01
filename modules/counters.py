"""Keyword counting and percentage calculation"""
import re


def escape_regex(s: str) -> str:
    """Escape special regex characters"""
    return re.escape(s)


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
