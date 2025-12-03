"""
Quick System Evaluation for SkillHighlight
Tests: Keyword Highlighting + End-to-End Scoring
"""
import sys
import re
sys.path.insert(0, '.')
import spacy
import json
from modules.highlight import highlight_keywords
from modules.scoring import analyze_sentences
from models.embedder import load_bert_model
from models.knn_classifier import load_or_train_knn

print('Loading models...')
nlp = spacy.load('en_core_web_sm')
bert_model = load_bert_model()
knn_model = load_or_train_knn(model_path="models/knn_model.pkl", csv_path="data/self_promotion_dataset.csv")

with open('data/keywords.json', 'r') as f:
    kw = json.load(f)
# Note: keys are uppercase in the JSON
HARD = set(kw.get('HARD_SKILLS', []))
SOFT = set(kw.get('SOFT_SKILLS', []))
RECRUITER = set(kw.get('RECRUITER_KEYWORDS', []))
ACTION = set(kw.get('ACTION_VERBS', []))

print(f"Loaded {len(HARD)} hard skills, {len(SOFT)} soft skills, {len(RECRUITER)} recruiter kw, {len(ACTION)} action verbs")

def extract_highlighted(html):
    """Extract highlighted words from HTML output"""
    # Pattern to find text inside spans
    pattern = r"<span[^>]*>([^<]+)</span>"
    matches = re.findall(pattern, html)
    return set(w.lower().strip() for w in matches)

print()
print('='*60)
print('TEST 1: KEYWORD HIGHLIGHTING ACCURACY')
print('='*60)

# (text, should_highlight, should_NOT_highlight)
tests = [
    ('Proficient in Python and JavaScript programming', ['python', 'javascript'], []),
    ('Graduated in Spring 2022 from university', [], ['spring']),
    ('Led a team of 5 developers to success', ['led'], []),
    ('Strong communication and leadership skills', ['communication', 'leadership'], []),
    ('Responsible for data entry tasks', [], []),
    ('Developed new features using React', ['developed', 'react'], []),
    ('Experience with Machine Learning and SQL', ['machine learning', 'sql'], []),
    ('Increased sales by 30% annually', ['increased'], []),
]

tp, fp, fn = 0, 0, 0
passed = 0

for text, should_hl, should_not in tests:
    html = highlight_keywords(nlp, text, HARD, SOFT, RECRUITER, ACTION)
    found = extract_highlighted(html)
    
    expected = set(w.lower() for w in should_hl)
    not_expected = set(w.lower() for w in should_not)
    
    correct_found = expected & found
    missed = expected - found
    false_pos = found & not_expected
    
    tp += len(correct_found)
    fn += len(missed)
    fp += len(false_pos)
    
    ok = len(missed) == 0 and len(false_pos) == 0
    if ok:
        passed += 1
    
    status = "PASS" if ok else "FAIL"
    print(f"\n[{status}] \"{text[:50]}\"")
    print(f"       Expected: {should_hl}")
    print(f"       Found:    {list(found)}")
    if missed:
        print(f"       Missed:   {list(missed)}")
    if false_pos:
        print(f"       False+:   {list(false_pos)}")

prec = tp/(tp+fp) if (tp+fp) else 1.0
rec = tp/(tp+fn) if (tp+fn) else 1.0
f1 = 2*(prec*rec)/(prec+rec) if (prec+rec) else 0

print(f"\n{'='*60}")
print(f"HIGHLIGHTING RESULTS:")
print(f"  Tests Passed: {passed}/{len(tests)} ({passed/len(tests)*100:.1f}%)")
print(f"  Precision:    {prec:.1%}")
print(f"  Recall:       {rec:.1%}")
print(f"  F1-Score:     {f1:.1%}")

print()
print('='*60)
print('TEST 2: END-TO-END SELF-PROMOTION SCORING')
print('='*60)

# HIGH = should score > 0.6
# LOW = should score < 0.5
high_sentences = [
    'Increased revenue by 45% through strategic sales initiatives',
    'Led cross-functional team of 12 engineers to deliver ahead of schedule',
    'Spearheaded development of award-winning mobile application',
    'Achieved 98% customer satisfaction through proactive problem resolution',
    'Pioneered machine learning model that reduced costs by $500K annually',
]
low_sentences = [
    'Responsible for data entry and filing documents',
    'Worked on various projects as assigned by manager',
    'Attended weekly team meetings and took notes',
    'Assisted with general office administrative duties',
    'Helped the team complete tasks on time',
]

# Test HIGH sentences
print("\nHIGH self-promotion sentences (expected > 0.55):")
high_scores = []
high_correct = 0
for sent in high_sentences:
    results, _ = analyze_sentences(nlp, knn_model, bert_model, sent, ACTION, 'txt')
    score = results[0][1] if results else 0.0
    high_scores.append(score)
    status = "PASS" if score > 0.55 else "FAIL"
    if score > 0.55:
        high_correct += 1
    print(f"  [{status}] {score:.3f} - \"{sent[:50]}\"")

# Test LOW sentences
print("\nLOW self-promotion sentences (expected < 0.50):")
low_scores = []
low_correct = 0
for sent in low_sentences:
    results, _ = analyze_sentences(nlp, knn_model, bert_model, sent, ACTION, 'txt')
    score = results[0][1] if results else 0.0
    low_scores.append(score)
    status = "PASS" if score < 0.50 else "FAIL"
    if score < 0.50:
        low_correct += 1
    print(f"  [{status}] {score:.3f} - \"{sent[:50]}\"")

avg_high = sum(high_scores)/len(high_scores) if high_scores else 0
avg_low = sum(low_scores)/len(low_scores) if low_scores else 0
separation = avg_high - avg_low
total_correct = high_correct + low_correct
total_tests = len(high_sentences) + len(low_sentences)
accuracy = total_correct / total_tests * 100

print(f"\n{'='*60}")
print(f"END-TO-END RESULTS:")
print(f"  Classification: {total_correct}/{total_tests} ({accuracy:.1f}%)")
print(f"  Avg HIGH score: {avg_high:.3f}")
print(f"  Avg LOW score:  {avg_low:.3f}")
print(f"  Separation:     {separation:.3f}")
if separation >= 0.25:
    print(f"  Status: GOOD - Model distinguishes quality well")
elif separation >= 0.15:
    print(f"  Status: ACCEPTABLE - Model shows reasonable separation")
else:
    print(f"  Status: POOR - Model struggles to distinguish")

print()
print('='*60)
print('SUMMARY')
print('='*60)
print(f"  Keyword Highlighting: {passed}/{len(tests)} tests ({passed/len(tests)*100:.1f}%)")
print(f"  End-to-End Scoring:   {total_correct}/{total_tests} tests ({accuracy:.1f}%)")
print(f"  Score Separation:     {separation:.3f}")
print('='*60)
