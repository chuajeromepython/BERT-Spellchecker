"""
run_accuracy_tests.py

Runs every case in accuracy_tests/test_cases.json through the actual
BERT-spellchecker.py pipeline (both stages) and scores the output
against the known-correct "expected" answer for each case. This turns
"does it seem better?" into an actual number you can compare across
model versions, code changes, and tuning attempts.

--- Why this exists ---
Eyeballing one test sentence at a time can't tell you whether a change
(more training data, a different SymSpell parameter, a new custom
vocab entry) actually helped, hurt, or did nothing -- you need the
same fixed set of test cases scored the same way, every time, so the
only thing that changes between runs is whatever you just tuned.

--- Usage ---
    python run_accuracy_tests.py

Run this after any change (new fine-tuned model, edited custom vocab,
adjusted SymSpell settings, etc.) and compare the printed score, and
the saved result file, against the previous run.

--- Scoring ---
Each case is scored two ways:
  1. EXACT match -- final output equals the expected answer exactly
     (whitespace-normalized only). Strict.
  2. WORD match % -- normalized (lowercased, punctuation-stripped)
     word-by-word comparison. More forgiving of casing quirks (e.g.
     acronyms coming out uppercase) that aren't real correction
     failures -- see the script's own casing behavior.

Both are reported. Exact match is the headline "did this fully work"
number; word match % is useful for partial-credit / trend-tracking
and for telling "close" failures apart from "way off" ones.

Results are also saved to accuracy_tests/results/<timestamp>.json so
you can track scores over time, and the script prints a comparison
against the most recent previous run if one exists.
"""

import difflib
import importlib.util
import json
import os
import re
import sys
from datetime import datetime, timezone

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(SCRIPT_DIR)  # this file lives in accuracy_tests/
TEST_CASES_PATH = os.path.join(SCRIPT_DIR, "test_cases.json")
RESULTS_DIR = os.path.join(SCRIPT_DIR, "results")
PIPELINE_FILE = os.path.join(REPO_ROOT, "BERT-spellchecker.py")


def load_pipeline_module():
    """
    BERT-spellchecker.py can't be imported with a normal `import`
    statement (the hyphen in the filename isn't valid in a Python
    identifier), so it's loaded directly from its file path instead.
    """
    if not os.path.isfile(PIPELINE_FILE):
        raise SystemExit(
            f"Could not find {PIPELINE_FILE}. Run this script from inside "
            f"accuracy_tests/, with BERT-spellchecker.py one level up."
        )
    spec = importlib.util.spec_from_file_location("bert_spellchecker_pipeline", PIPELINE_FILE)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def normalize_words(text):
    """Lowercase, strip punctuation, collapse whitespace -- for the
    forgiving word-match comparison."""
    text = text.lower()
    words = re.findall(r"[a-z0-9']+", text)
    return words


def exact_match(a, b):
    return " ".join(a.split()) == " ".join(b.split())


def word_match_score(expected, got):
    """Returns (matching_word_count, total_expected_words, aligned_diff)."""
    exp_words = normalize_words(expected)
    got_words = normalize_words(got)
    sm = difflib.SequenceMatcher(None, exp_words, got_words, autojunk=False)
    matching = sum(block.size for block in sm.get_matching_blocks())
    diff_ops = []
    for tag, i1, i2, j1, j2 in sm.get_opcodes():
        if tag == "equal":
            continue
        diff_ops.append({
            "type": tag,
            "expected": exp_words[i1:i2],
            "got": got_words[j1:j2],
        })
    total = max(len(exp_words), 1)
    return matching, total, diff_ops


def run_tests():
    with open(TEST_CASES_PATH, "r", encoding="utf-8") as f:
        test_cases = json.load(f)

    print(f"Loaded {len(test_cases)} test cases.\n")

    module = load_pipeline_module()
    sym_spell = module.build_symspell()
    tokenizer, model, torch_mod = module.build_bert_scorer()
    print()

    results = []
    for case in test_cases:
        stage1_out, final_out = module.correct_text(
            sym_spell, tokenizer, model, torch_mod, case["input"]
        )
        is_exact = exact_match(case["expected"], final_out)
        matching, total, diff_ops = word_match_score(case["expected"], final_out)
        word_pct = round(100 * matching / total, 1)

        results.append({
            "id": case["id"],
            "category": case.get("category", ""),
            "notes": case.get("notes", ""),
            "input": case["input"],
            "expected": case["expected"],
            "stage1_output": stage1_out,
            "final_output": final_out,
            "exact_match": is_exact,
            "word_match_pct": word_pct,
            "diff": diff_ops,
        })

    return results


def classify_status(exact, word_pct):
    if exact:
        return "PASS"
    if word_pct >= 99.9:
        return "PASS*"  # every word correct; only punctuation/whitespace differs
    if word_pct >= 90:
        return "CLOSE"
    return "FAIL"


def print_report(results):
    print("=" * 78)
    print("RESULTS BY CASE")
    print("=" * 78)
    print("(PASS* = every word correct, only punctuation/whitespace differs --")
    print(" see the punctuation note at the end of this report)\n")
    for r in results:
        status = classify_status(r["exact_match"], r["word_match_pct"])
        print(f"\n[{status}] {r['id']}  ({r['category']})  word-match: {r['word_match_pct']}%")
        print(f"  Input:    {r['input']}")
        print(f"  Expected: {r['expected']}")
        print(f"  Got:      {r['final_output']}")
        if r["notes"]:
            print(f"  Note:     {r['notes']}")
        if status in ("CLOSE", "FAIL") and r["diff"]:
            for d in r["diff"]:
                exp_str = " ".join(d["expected"]) or "(nothing)"
                got_str = " ".join(d["got"]) or "(nothing)"
                print(f"    -> expected [{exp_str}]  but got  [{got_str}]")

    total = len(results)
    exact_passes = sum(1 for r in results if r["exact_match"])
    content_passes = sum(1 for r in results if r["word_match_pct"] >= 99.9)
    avg_word_pct = round(sum(r["word_match_pct"] for r in results) / total, 1) if total else 0

    print("\n" + "=" * 78)
    print("SUMMARY")
    print("=" * 78)
    print(f"Exact match (incl. punctuation): {exact_passes}/{total}  ({round(100*exact_passes/total,1)}%)")
    print(f"Content match (words only):      {content_passes}/{total}  ({round(100*content_passes/total,1)}%)")
    print(f"Avg word-match:                  {avg_word_pct}%")

    if exact_passes == 0 and content_passes > 0:
        print(
            "\nNote: exact match is 0 even though word content is often fully correct.\n"
            "This pipeline's Stage 1 (SymSpell lookup_compound) systematically drops\n"
            "most punctuation -- periods, commas, question marks, apostrophes. That's\n"
            "a real, separate behavior worth knowing about, but it isn't a spelling-\n"
            "correction failure. 'Content match' above ignores punctuation and is the\n"
            "more meaningful number for tracking correction accuracy specifically."
        )

    print("\nBy category:")
    categories = sorted(set(r["category"] for r in results))
    for cat in categories:
        cat_results = [r for r in results if r["category"] == cat]
        cat_content = sum(1 for r in cat_results if r["word_match_pct"] >= 99.9)
        cat_avg = round(sum(r["word_match_pct"] for r in cat_results) / len(cat_results), 1)
        print(f"  {cat:<25} content {cat_content}/{len(cat_results)}   avg word-match {cat_avg}%")

    return exact_passes, total, avg_word_pct


def save_and_compare(results, exact_passes, total, avg_word_pct):
    os.makedirs(RESULTS_DIR, exist_ok=True)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out_path = os.path.join(RESULTS_DIR, f"{timestamp}.json")

    existing = sorted(
        [f for f in os.listdir(RESULTS_DIR) if f.endswith(".json")]
    )
    previous_summary = None
    if existing:
        with open(os.path.join(RESULTS_DIR, existing[-1]), "r", encoding="utf-8") as f:
            prev = json.load(f)
            previous_summary = prev.get("summary")

    summary = {
        "timestamp": timestamp,
        "exact_matches": exact_passes,
        "total_cases": total,
        "exact_match_pct": round(100 * exact_passes / total, 1) if total else 0,
        "avg_word_match_pct": avg_word_pct,
    }
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump({"summary": summary, "results": results}, f, indent=2)

    print(f"\nSaved results to: {out_path}")

    if previous_summary:
        print("\nCompared to previous run:")
        d_exact = summary["exact_match_pct"] - previous_summary["exact_match_pct"]
        d_word = summary["avg_word_match_pct"] - previous_summary["avg_word_match_pct"]
        sign = lambda v: f"+{v}" if v >= 0 else str(v)
        print(f"  Exact match:    {previous_summary['exact_match_pct']}% -> {summary['exact_match_pct']}%  ({sign(round(d_exact,1))} pts)")
        print(f"  Avg word-match: {previous_summary['avg_word_match_pct']}% -> {summary['avg_word_match_pct']}%  ({sign(round(d_word,1))} pts)")
    else:
        print("\n(No previous run found -- this is the new baseline.)")


def main():
    results = run_tests()
    exact_passes, total, avg_word_pct = print_report(results)
    save_and_compare(results, exact_passes, total, avg_word_pct)


if __name__ == "__main__":
    main()