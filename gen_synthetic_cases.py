"""
gen_synthetic_cases.py

Generates synthetic OCR-style test cases in the same format as
accuracy_tests/test_cases.json (id, category, input, expected, notes),
plus a "split" field ("dev" / "holdout") so tuning and final scoring
can use different cases.

Clean sentences go in; corrupted copies come out. Corruptions imitate
handwriting-OCR failure modes: look-alike character swaps (rn <-> m,
u <-> n, cl <-> d ...), dropped letters, extra letters, merged words,
and split words. About 15% of cases are untouched controls, to catch
over-correction.

--- Usage ---
Edit the SETTINGS block near the top (SEED, COUNT, ...), save, then:
    python gen_synthetic_cases.py

Each run writes synthetic_cases_seed<SEED>.json. Change SEED for a new batch.
(Command-line options like --seed 8 --count 50 still work as overrides.)

--- Important ---
Synthetic errors are a stand-in. They follow the confusion tables below,
not your real OCR model's error distribution, so a score on this file
tells you whether the pipeline handles these error TYPES, not what it
will score on real output. Use real OCR output paired with ground truth
(e.g. IAM transcriptions) for the headline number.
"""

import argparse
import json
import random
import re

# =====================================================================
# SETTINGS -- edit these, save, then just run:  python gen_synthetic_cases.py
# =====================================================================

# SEED picks WHICH random cases get made. Same seed + same COUNT = the
# exact same cases every time. Change this number (7, 8, 9, 123 ...)
# each time you want a fresh batch, otherwise you'll just regenerate
# the batch you already have.
SEED = 7

# How many test cases to generate in one run.
COUNT = 100

# Optional: path to a text file of CLEAN sentences, one per line (e.g.
# IAM transcriptions). Leave as None to use the built-in sentences.
# Example:  INPUT_FILE = "clean.txt"
INPUT_FILE = None

# Output filename. Leave as None and the file is named after the seed
# (synthetic_cases_seed7.json), so changing SEED never overwrites an
# older batch. Set a name here only if you want to pick it yourself.
OUTPUT_FILE = None

# Every Nth case is labeled "holdout" (the rest are "dev"). 4 = 25%
# holdout. Tune your code against "dev" cases; check "holdout" cases
# only occasionally so you don't overfit.
HOLDOUT_EVERY = 4

# =====================================================================

# (what the letters really are, what OCR misreads them as)
CONFUSIONS = [
    ("m", "rn"), ("rn", "m"), ("u", "n"), ("n", "u"),
    ("e", "c"), ("c", "e"), ("a", "o"), ("o", "a"),
    ("d", "cl"), ("cl", "d"), ("h", "b"), ("b", "h"),
    ("l", "i"), ("i", "l"), ("w", "vv"), ("vv", "w"),
]

DEFAULT_SENTENCES = [
    "The committee reviewed the proposal before the end of the week.",
    "Several students asked questions about the final project requirements.",
    "We tested the application on three different devices yesterday.",
    "The report explains how the system handles missing information.",
    "She walked to the library to return the books she borrowed.",
    "Please send the updated schedule to everyone on the team.",
    "The results were better than we expected after the second trial.",
    "He explained the problem clearly and answered every question.",
    "The new policy takes effect at the beginning of next month.",
    "Our group decided to split the work into smaller tasks.",
    "The teacher handed back the graded papers during the lesson.",
    "A reliable connection is necessary for the data to sync properly.",
    "They compared both methods and chose the faster approach.",
    "The meeting was moved because several people were unavailable.",
    "Each answer sheet is scanned and stored in the database.",
    "The weather was cold, but the event went ahead as planned.",
    "I believe the second draft is much stronger than the first.",
    "The manager approved the budget after reading the summary.",
    "Students must submit their forms before the deadline passes.",
    "The old computer was too slow to run the program.",
    "We noticed a pattern in the errors that appeared last week.",
    "The author describes a quiet village near the mountains.",
    "Their research focuses on improving the accuracy of scanning tools.",
    "Nothing in the document suggested that the rules had changed.",
    "The instructor gave detailed feedback on every submitted essay.",
    "Many people prefer working from home on rainy days.",
    "The package arrived late, so the repair was delayed again.",
    "It is important to back up your files before any update.",
    "The survey showed that most users liked the simpler layout.",
    "After lunch, the group returned to the unfinished diagram.",
    "The library closes early on weekends during the summer months.",
    "He forgot his password, so the account had to be reset.",
    "We measured the temperature every hour throughout the experiment.",
    "The company plans to open a second office in the city.",
    "Reading the instructions carefully would have saved us time.",
    "The children watched the rain from the window all afternoon.",
    "An error message appeared when the file was too large.",
    "She practiced the presentation until she felt confident.",
    "The council announced a new program to support local farmers.",
    "Good notes make it easier to review the material later.",
]


def split_punct(token):
    m = re.match(r"^(\W*)(.*?)(\W*)$", token)
    return m.group(1), m.group(2), m.group(3)


def eligible(core):
    return len(core) >= 4 and core.isalpha()


def confuse(word, rng):
    spots = []
    low = word.lower()
    for src, dst in CONFUSIONS:
        start = 0
        while True:
            i = low.find(src, start)
            if i == -1:
                break
            spots.append((i, src, dst))
            start = i + 1
    if not spots:
        return None
    i, src, dst = rng.choice(spots)
    return word[:i] + dst + word[i + len(src):]


def drop_letter(word, rng):
    if len(word) < 5:
        return None
    i = rng.randrange(1, len(word) - 1)
    return word[:i] + word[i + 1:]


def extra_letter(word, rng):
    i = rng.randrange(1, len(word))
    return word[:i] + word[i - 1] + word[i:]


WORD_LEVEL = {
    "ocr_char_confusion": confuse,
    "ocr_dropped_letter": drop_letter,
    "ocr_extra_letter": extra_letter,
}


def corrupt_words(sentence, fn, rng):
    tokens = sentence.split(" ")
    idxs = [i for i, t in enumerate(tokens) if eligible(split_punct(t)[1])]
    rng.shuffle(idxs)
    target = rng.randint(1, 3)
    changed = []
    for i in idxs:
        if len(changed) >= target:
            break
        pre, core, post = split_punct(tokens[i])
        new = fn(core, rng)
        if new and new.lower() != core.lower():
            tokens[i] = pre + new + post
            changed.append(core)
    if not changed:
        return None
    return " ".join(tokens), changed


def merge_words(sentence, rng):
    tokens = sentence.split(" ")
    idxs = [i for i in range(len(tokens) - 1)
            if tokens[i][-1:].isalpha() and tokens[i + 1][:1].isalpha()]
    if not idxs:
        return None
    i = rng.choice(idxs)
    joined = tokens[i] + tokens[i + 1]
    tokens[i:i + 2] = [joined]
    return " ".join(tokens), [joined]


def split_word(sentence, rng):
    tokens = sentence.split(" ")
    idxs = [i for i, t in enumerate(tokens) if len(split_punct(t)[1]) >= 7
            and split_punct(t)[1].isalpha()]
    if not idxs:
        return None
    i = rng.choice(idxs)
    pre, core, post = split_punct(tokens[i])
    cut = rng.randint(3, len(core) - 3)
    tokens[i] = pre + core[:cut] + " " + core[cut:] + post
    return " ".join(tokens), [core]


CATEGORIES = [
    "ocr_char_confusion", "ocr_char_confusion",
    "ocr_dropped_letter", "ocr_extra_letter",
    "ocr_merge", "ocr_split", "over_correction_check",
]


def make_case(category, sentence, rng):
    if category == "over_correction_check":
        return sentence, "Clean control; output should be unchanged."
    if category == "ocr_merge":
        res = merge_words(sentence, rng)
    elif category == "ocr_split":
        res = split_word(sentence, rng)
    else:
        res = corrupt_words(sentence, WORD_LEVEL[category], rng)
    if res is None:
        return None
    corrupted, words = res
    return corrupted, f"Synthetic {category}; affected: {', '.join(words)}"


def main():
    # Command-line options are OPTIONAL overrides. With no options, the
    # values from the SETTINGS block at the top of the file are used.
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default=INPUT_FILE,
                    help="text file, one clean sentence per line")
    ap.add_argument("--output", default=OUTPUT_FILE)
    ap.add_argument("--count", type=int, default=COUNT)
    ap.add_argument("--seed", type=int, default=SEED)
    ap.add_argument("--holdout_every", type=int, default=HOLDOUT_EVERY,
                    help="every Nth case is marked holdout (4 = 25%%)")
    args = ap.parse_args()

    output_path = args.output or f"synthetic_cases_seed{args.seed}.json"
    rng = random.Random(args.seed)

    if args.input:
        with open(args.input, "r", encoding="utf-8") as f:
            sentences = [l.strip() for l in f if l.strip()]
    else:
        sentences = list(DEFAULT_SENTENCES)
    if not sentences:
        raise SystemExit("No sentences to work from.")

    cases, counters = [], {}
    attempts = 0
    while len(cases) < args.count and attempts < args.count * 20:
        attempts += 1
        category = CATEGORIES[len(cases) % len(CATEGORIES)]
        sentence = rng.choice(sentences)
        made = make_case(category, sentence, rng)
        if made is None:
            continue
        corrupted, notes = made
        counters[category] = counters.get(category, 0) + 1
        cases.append({
            "id": f"syn_{category}_{counters[category]}",
            "category": category,
            "input": corrupted,
            "expected": sentence,
            "notes": notes,
            "split": "holdout" if (len(cases) + 1) % args.holdout_every == 0 else "dev",
        })

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(cases, f, indent=2, ensure_ascii=False)

    n_hold = sum(1 for c in cases if c["split"] == "holdout")
    print(f"Wrote {len(cases)} cases to {output_path} "
          f"({len(cases) - n_hold} dev, {n_hold} holdout).")
    for cat, n in sorted(counters.items()):
        print(f"  {cat}: {n}")


if __name__ == "__main__":
    main()