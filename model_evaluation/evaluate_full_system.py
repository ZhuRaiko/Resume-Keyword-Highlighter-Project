"""
Full System Evaluation Script for SkillHighlight
Evaluates: Text Extraction, Keyword Highlighting, End-to-End Pipeline

This script provides comprehensive evaluation beyond just the KNN model.
"""

import os
import sys
import json
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import spacy
from collections import defaultdict

# Import project modules
from modules.extractor import extract_from_file
from modules.highlight import highlight_keywords
from modules.scoring import analyze_sentences
from models.embedder import load_bert_model
from models.knn_classifier import load_or_train_knn

# Load models
print("Loading models...")
nlp = spacy.load("en_core_web_sm")
bert_model = load_bert_model()
knn_model = load_or_train_knn()

# Load keywords
with open(Path(__file__).parent.parent / "data" / "keywords.json", "r") as f:
    keywords_data = json.load(f)

HARD_SKILLS = set(keywords_data.get("hard_skills", []))
SOFT_SKILLS = set(keywords_data.get("soft_skills", []))
RECRUITER_KEYWORDS = set(keywords_data.get("recruiter_keywords", []))
ACTION_VERBS = set(keywords_data.get("action_verbs", []))


# ============================================================================
# TEST 1: TEXT EXTRACTION EVALUATION
# ============================================================================

def evaluate_text_extraction():
    """
    Evaluates text extraction quality by comparing extracted text
    against expected content.
    
    Metrics:
    - Extraction Success Rate: Did it extract without errors?
    - Content Completeness: Are key sections present?
    - Text Quality: Is the text clean and readable?
    """
    print("\n" + "="*60)
    print("TEST 1: TEXT EXTRACTION EVALUATION")
    print("="*60)
    
    # Define test cases with expected content markers
    # These are phrases that SHOULD appear in the extracted text
    test_cases = [
        {
            "description": "Sample text content for extraction testing",
            "expected_phrases": [
                # Add phrases you expect to find in your test files
            ]
        }
    ]
    
    # Look for test files in a test_resumes folder
    test_folder = Path(__file__).parent / "test_resumes"
    
    if not test_folder.exists():
        print(f"\n⚠️  No test_resumes folder found at: {test_folder}")
        print("   Create this folder and add sample PDF/DOCX files to test extraction.")
        print("\n   Manual Extraction Test:")
        print("   ─────────────────────────")
        
        # Interactive test
        test_text = """
        SAMPLE RESUME TEXT FOR TESTING
        
        John Doe
        Software Developer
        
        EXPERIENCE
        • Led development of Python web applications using Django framework
        • Increased system performance by 40% through code optimization
        • Collaborated with cross-functional teams to deliver projects on time
        
        SKILLS
        Python, JavaScript, SQL, Machine Learning, Communication, Leadership
        """
        
        print("   Using embedded sample text for demonstration...")
        
        # Count expected elements
        sentences = [s.strip() for s in test_text.split('\n') if s.strip()]
        print(f"   ✅ Lines extracted: {len(sentences)}")
        print(f"   ✅ Characters: {len(test_text)}")
        
        return {
            "status": "MANUAL_TEST_NEEDED",
            "message": "Add PDF/DOCX files to test_resumes/ folder for full evaluation"
        }
    
    # Test actual files
    results = []
    test_files = list(test_folder.glob("*.pdf")) + list(test_folder.glob("*.docx")) + list(test_folder.glob("*.txt"))
    
    if not test_files:
        print(f"\n⚠️  No test files found in {test_folder}")
        return {"status": "NO_FILES", "message": "Add test files to evaluate"}
    
    for file_path in test_files:
        print(f"\n   Testing: {file_path.name}")
        try:
            extracted_text = extract_from_file(str(file_path))
            
            if extracted_text and len(extracted_text.strip()) > 0:
                word_count = len(extracted_text.split())
                char_count = len(extracted_text)
                line_count = len(extracted_text.split('\n'))
                
                print(f"   ✅ Success - Words: {word_count}, Chars: {char_count}, Lines: {line_count}")
                
                results.append({
                    "file": file_path.name,
                    "success": True,
                    "word_count": word_count,
                    "char_count": char_count
                })
            else:
                print(f"   ❌ Failed - Empty extraction")
                results.append({"file": file_path.name, "success": False, "error": "Empty"})
                
        except Exception as e:
            print(f"   ❌ Error: {str(e)[:50]}")
            results.append({"file": file_path.name, "success": False, "error": str(e)})
    
    # Calculate success rate
    success_count = sum(1 for r in results if r.get("success"))
    success_rate = (success_count / len(results) * 100) if results else 0
    
    print(f"\n   EXTRACTION SUCCESS RATE: {success_rate:.1f}% ({success_count}/{len(results)})")
    
    return {
        "status": "COMPLETED",
        "success_rate": success_rate,
        "files_tested": len(results),
        "results": results
    }


# ============================================================================
# TEST 2: KEYWORD HIGHLIGHTING EVALUATION
# ============================================================================

def evaluate_keyword_highlighting():
    """
    Evaluates keyword highlighting accuracy using test sentences
    with known expected outcomes.
    
    Metrics:
    - True Positives: Correctly highlighted keywords
    - False Positives: Incorrectly highlighted words
    - False Negatives: Missed keywords
    - Precision & Recall for highlighting
    """
    print("\n" + "="*60)
    print("TEST 2: KEYWORD HIGHLIGHTING EVALUATION")
    print("="*60)
    
    # Test cases: (text, expected_highlights, should_not_highlight)
    test_cases = [
        # Hard Skills - Should highlight
        {
            "text": "Proficient in Python and JavaScript programming",
            "should_highlight": ["Python", "JavaScript"],
            "should_not_highlight": []
        },
        {
            "text": "Experience with Machine Learning and SQL databases",
            "should_highlight": ["Machine Learning", "SQL"],
            "should_not_highlight": []
        },
        
        # Context tests - Should NOT highlight
        {
            "text": "Graduated in Spring 2022 from MIT",
            "should_highlight": [],
            "should_not_highlight": ["Spring"]  # Season, not framework
        },
        {
            "text": "The project took three months to complete",
            "should_highlight": [],
            "should_not_highlight": ["project"]  # Generic noun
        },
        
        # Action Verbs - Should highlight
        {
            "text": "Led a team of 5 developers to deliver the product",
            "should_highlight": ["Led"],
            "should_not_highlight": []
        },
        {
            "text": "Developed and implemented new features",
            "should_highlight": ["Developed", "implemented"],
            "should_not_highlight": []
        },
        {
            "text": "Increased sales by 30% through strategic marketing",
            "should_highlight": ["Increased"],
            "should_not_highlight": []
        },
        
        # Soft Skills - Should highlight
        {
            "text": "Strong communication and leadership skills",
            "should_highlight": ["communication", "leadership"],
            "should_not_highlight": []
        },
        {
            "text": "Excellent teamwork and problem-solving abilities",
            "should_highlight": ["teamwork", "problem-solving"],
            "should_not_highlight": []
        },
        
        # Mixed/Edge cases
        {
            "text": "Responsible for data entry tasks",
            "should_highlight": [],  # "Responsible for" is weak language
            "should_not_highlight": ["Responsible"]
        },
        {
            "text": "Worked on various projects",
            "should_highlight": [],  # Vague
            "should_not_highlight": ["Worked", "projects"]
        },
    ]
    
    true_positives = 0
    false_positives = 0
    false_negatives = 0
    total_expected = 0
    
    detailed_results = []
    
    for i, test in enumerate(test_cases, 1):
        text = test["text"]
        expected = set(kw.lower() for kw in test["should_highlight"])
        should_not = set(kw.lower() for kw in test["should_not_highlight"])
        
        # Get highlighting result
        highlighted_html, stats = highlight_keywords(
            text, nlp,
            hard_skills=HARD_SKILLS,
            soft_skills=SOFT_SKILLS,
            recruiter_kw=RECRUITER_KEYWORDS,
            action_verbs=ACTION_VERBS,
            show_hard=True,
            show_soft=True,
            show_recruiter=True,
            show_action=True
        )
        
        # Extract highlighted words from stats
        highlighted_words = set()
        for category in ['HARD', 'SOFT', 'RECRUITER', 'ACTION']:
            if category in stats:
                for word in stats[category]:
                    highlighted_words.add(word.lower())
        
        # Calculate metrics for this test
        test_tp = len(expected & highlighted_words)
        test_fn = len(expected - highlighted_words)
        test_fp = len(highlighted_words & should_not)
        
        true_positives += test_tp
        false_negatives += test_fn
        false_positives += test_fp
        total_expected += len(expected)
        
        # Determine pass/fail
        passed = test_tp == len(expected) and test_fp == 0
        status = "✅" if passed else "❌"
        
        print(f"\n   Test {i}: {status}")
        print(f"   Text: \"{text[:50]}...\"" if len(text) > 50 else f"   Text: \"{text}\"")
        print(f"   Expected: {test['should_highlight']}")
        print(f"   Found: {list(highlighted_words)}")
        if test_fn > 0:
            print(f"   Missed: {list(expected - highlighted_words)}")
        if test_fp > 0:
            print(f"   False Positives: {list(highlighted_words & should_not)}")
        
        detailed_results.append({
            "text": text,
            "passed": passed,
            "expected": list(expected),
            "found": list(highlighted_words)
        })
    
    # Calculate overall metrics
    precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
    recall = true_positives / total_expected if total_expected > 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    
    passed_tests = sum(1 for r in detailed_results if r["passed"])
    accuracy = passed_tests / len(detailed_results) * 100
    
    print(f"\n   ─────────────────────────────────")
    print(f"   HIGHLIGHTING METRICS:")
    print(f"   Tests Passed: {passed_tests}/{len(detailed_results)} ({accuracy:.1f}%)")
    print(f"   True Positives: {true_positives}")
    print(f"   False Positives: {false_positives}")
    print(f"   False Negatives: {false_negatives}")
    print(f"   Precision: {precision:.1%}")
    print(f"   Recall: {recall:.1%}")
    print(f"   F1-Score: {f1:.1%}")
    
    return {
        "status": "COMPLETED",
        "tests_passed": passed_tests,
        "total_tests": len(detailed_results),
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1_score": f1,
        "true_positives": true_positives,
        "false_positives": false_positives,
        "false_negatives": false_negatives
    }


# ============================================================================
# TEST 3: END-TO-END PIPELINE EVALUATION
# ============================================================================

def evaluate_end_to_end():
    """
    Evaluates the complete pipeline from text input to final scores.
    
    Tests:
    - Self-promotion scoring on known sentences
    - Score distribution (high-quality vs low-quality sentences)
    - Consistency between similar sentences
    """
    print("\n" + "="*60)
    print("TEST 3: END-TO-END PIPELINE EVALUATION")
    print("="*60)
    
    # Test sentences with expected score ranges
    # HIGH = strong self-promotion (>0.6)
    # LOW = weak/neutral (<0.5)
    test_sentences = [
        # HIGH self-promotion sentences
        {"text": "Increased revenue by 45% through strategic sales initiatives", "expected": "HIGH"},
        {"text": "Led cross-functional team of 12 engineers to deliver product ahead of schedule", "expected": "HIGH"},
        {"text": "Spearheaded development of award-winning mobile application with 1M+ downloads", "expected": "HIGH"},
        {"text": "Achieved 98% customer satisfaction rating through proactive problem resolution", "expected": "HIGH"},
        {"text": "Pioneered implementation of machine learning model that reduced costs by $500K annually", "expected": "HIGH"},
        
        # LOW self-promotion sentences
        {"text": "Responsible for data entry and filing", "expected": "LOW"},
        {"text": "Worked on various projects as assigned", "expected": "LOW"},
        {"text": "Attended weekly team meetings", "expected": "LOW"},
        {"text": "Assisted with general office duties", "expected": "LOW"},
        {"text": "Helped the team with tasks", "expected": "LOW"},
        
        # MEDIUM - borderline cases (0.4-0.6)
        {"text": "Managed customer accounts and resolved inquiries", "expected": "MEDIUM"},
        {"text": "Developed Python scripts for data analysis", "expected": "MEDIUM"},
        {"text": "Collaborated with team members on project deliverables", "expected": "MEDIUM"},
    ]
    
    results = []
    correct_predictions = 0
    
    # Analyze each sentence
    full_text = "\n".join([t["text"] for t in test_sentences])
    sentence_results, avg_score = analyze_sentences(
        nlp, knn_model, bert_model, full_text, ACTION_VERBS, "txt"
    )
    
    print(f"\n   Analyzing {len(test_sentences)} test sentences...")
    print(f"   ─────────────────────────────────")
    
    for i, (test, result) in enumerate(zip(test_sentences, sentence_results)):
        score = result[1]  # (sentence_text, score)
        expected = test["expected"]
        
        # Determine predicted category
        if score >= 0.6:
            predicted = "HIGH"
        elif score <= 0.45:
            predicted = "LOW"
        else:
            predicted = "MEDIUM"
        
        # Check if prediction matches (MEDIUM matches anything in 0.4-0.6)
        if expected == "MEDIUM":
            correct = 0.4 <= score <= 0.65
        elif expected == "HIGH":
            correct = score >= 0.55  # Allow some tolerance
        else:  # LOW
            correct = score <= 0.5   # Allow some tolerance
        
        if correct:
            correct_predictions += 1
            status = "✅"
        else:
            status = "❌"
        
        print(f"\n   {status} \"{test['text'][:45]}...\"" if len(test['text']) > 45 else f"\n   {status} \"{test['text']}\"")
        print(f"      Score: {score:.3f} | Expected: {expected} | Got: {predicted}")
        
        results.append({
            "text": test["text"],
            "score": score,
            "expected": expected,
            "predicted": predicted,
            "correct": correct
        })
    
    accuracy = correct_predictions / len(test_sentences) * 100
    
    # Calculate average scores by category
    high_scores = [r["score"] for r in results if r["expected"] == "HIGH"]
    low_scores = [r["score"] for r in results if r["expected"] == "LOW"]
    
    avg_high = sum(high_scores) / len(high_scores) if high_scores else 0
    avg_low = sum(low_scores) / len(low_scores) if low_scores else 0
    separation = avg_high - avg_low
    
    print(f"\n   ─────────────────────────────────")
    print(f"   END-TO-END METRICS:")
    print(f"   Classification Accuracy: {accuracy:.1f}% ({correct_predictions}/{len(test_sentences)})")
    print(f"   Avg Score (HIGH sentences): {avg_high:.3f}")
    print(f"   Avg Score (LOW sentences): {avg_low:.3f}")
    print(f"   Score Separation: {separation:.3f}")
    
    if separation >= 0.3:
        print(f"   ✅ Good separation between HIGH and LOW categories")
    else:
        print(f"   ⚠️  Low separation - model may struggle to distinguish quality")
    
    return {
        "status": "COMPLETED",
        "accuracy": accuracy,
        "correct": correct_predictions,
        "total": len(test_sentences),
        "avg_high_score": avg_high,
        "avg_low_score": avg_low,
        "score_separation": separation,
        "results": results
    }


# ============================================================================
# MAIN EVALUATION RUNNER
# ============================================================================

def main():
    """Run all evaluations and generate summary report."""
    print("\n" + "="*60)
    print("   SKILLHIGHLIGHT FULL SYSTEM EVALUATION")
    print("="*60)
    
    all_results = {}
    
    # Run all tests
    all_results["extraction"] = evaluate_text_extraction()
    all_results["highlighting"] = evaluate_keyword_highlighting()
    all_results["end_to_end"] = evaluate_end_to_end()
    
    # Summary
    print("\n" + "="*60)
    print("   EVALUATION SUMMARY")
    print("="*60)
    
    print("\n   ┌─────────────────────────────┬───────────┬────────────┐")
    print("   │ Component                   │ Status    │ Score      │")
    print("   ├─────────────────────────────┼───────────┼────────────┤")
    
    # Extraction
    ext = all_results["extraction"]
    if ext["status"] == "COMPLETED":
        print(f"   │ Text Extraction             │ ✅ Tested │ {ext['success_rate']:.1f}%      │")
    else:
        print(f"   │ Text Extraction             │ ⚠️  Manual │ N/A        │")
    
    # Highlighting
    hl = all_results["highlighting"]
    print(f"   │ Keyword Highlighting        │ ✅ Tested │ {hl['accuracy']:.1f}%      │")
    print(f"   │   └─ Precision              │           │ {hl['precision']:.1%}      │")
    print(f"   │   └─ Recall                 │           │ {hl['recall']:.1%}      │")
    
    # End-to-end
    e2e = all_results["end_to_end"]
    print(f"   │ End-to-End Pipeline         │ ✅ Tested │ {e2e['accuracy']:.1f}%      │")
    print(f"   │   └─ Score Separation       │           │ {e2e['score_separation']:.3f}      │")
    
    print("   └─────────────────────────────┴───────────┴────────────┘")
    
    # Save results
    output_path = Path(__file__).parent / "full_system_metrics.json"
    with open(output_path, "w") as f:
        # Convert for JSON serialization
        json_results = {
            "extraction": {
                "status": ext["status"],
                "success_rate": ext.get("success_rate", "N/A")
            },
            "highlighting": {
                "accuracy": hl["accuracy"],
                "precision": hl["precision"],
                "recall": hl["recall"],
                "f1_score": hl["f1_score"]
            },
            "end_to_end": {
                "accuracy": e2e["accuracy"],
                "avg_high_score": e2e["avg_high_score"],
                "avg_low_score": e2e["avg_low_score"],
                "score_separation": e2e["score_separation"]
            }
        }
        json.dump(json_results, f, indent=2)
    
    print(f"\n   Results saved to: {output_path}")
    print("\n" + "="*60)
    
    return all_results


if __name__ == "__main__":
    main()
