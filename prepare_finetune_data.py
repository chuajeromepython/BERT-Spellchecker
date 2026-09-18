"""
prepare_finetune_data.py

Turns a folder of clean, correctly-spelled .txt files into training
data for fine-tuning BERT (masked language modeling).

--- Why you don't need "error -> correct" pairs ---
Stage 2 of the spellchecker only ever uses BERT to answer one
question: "given this sentence with one word masked out, how likely
is candidate word X at that position?" That's exactly BERT's original
training objective (masked language modeling, MLM). Fine-tuning it
further on MLM, using clean text from YOUR domain, makes it better at
that same question for your vocabulary and writing style -- e.g. it
will learn that "STARS" or "iStar" or "Xampp" fit naturally in your
sentences, so it scores real-word-error candidates more accurately
in your kind of text. You do NOT need pairs of broken/fixed text for
this -- just a pile of good, correct writing in your domain.

--- What to put in the input folder ---
Any .txt files containing clean, correctly-spelled text similar in
topic/vocabulary to what your OCR pipeline will produce. Good
sources: your thesis chapters/drafts, related project documentation,
reports, or any other correctly-typed text in the same domain. More
text is better, but even a few thousand sentences helps.

--- What this script does ---
1. Reads every .txt file in the input folder.
2. Splits into sentences, drops junk (too short/too long, duplicates).
3. Groups sentences into ~128-token chunks (rough word-count proxy)
   so training examples look like realistic paragraphs of context,
   not isolated one-liners.
4. Shuffles and splits into train / validation (90 / 10).
5. Writes train.txt and val.txt, one training example per line, to
   the output folder.

--- Usage ---
    python prepare_finetune_data.py --input_dir finetune_data/raw --output_dir finetune_data
"""

import argparse
import glob
import os
import random
import re

MIN_SENTENCE_WORDS = 4
CHUNK_TARGET_WORDS = 128
VAL_FRACTION = 0.1


def read_all_text(input_dir):
    paths = sorted(glob.glob(os.path.join(input_dir, "**", "*.txt"), recursive=True))
    if not paths:
        raise SystemExit(
            f"No .txt files found in {input_dir!r}. Put your clean domain text "
            f"there first (one or more .txt files, any filenames)."
        )
    texts = []
    for p in paths:
        with open(p, "r", encoding="utf-8", errors="ignore") as f:
            texts.append(f.read())
    print(f"Read {len(paths)} file(s) from {input_dir}")
    return "\n".join(texts)


def split_sentences(text):
    # Normalize whitespace first so multi-line paragraphs don't break
    # the sentence-boundary regex.
    text = re.sub(r"\s+", " ", text).strip()
    sentences = re.split(r"(?<=[.!?])\s+", text)
    cleaned = []
    for s in sentences:
        s = s.strip()
        if len(s.split()) >= MIN_SENTENCE_WORDS:
            cleaned.append(s)
    return cleaned


def chunk_sentences(sentences, target_words=CHUNK_TARGET_WORDS):
    chunks = []
    current, current_words = [], 0
    for s in sentences:
        n = len(s.split())
        if current and current_words + n > target_words:
            chunks.append(" ".join(current))
            current, current_words = [], 0
        current.append(s)
        current_words += n
    if current:
        chunks.append(" ".join(current))
    return chunks


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input_dir", default="finetune_data/raw",
                     help="Folder of clean .txt files (searched recursively).")
    ap.add_argument("--output_dir", default="finetune_data",
                     help="Where to write train.txt / val.txt.")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    random.seed(args.seed)

    raw_text = read_all_text(args.input_dir)
    sentences = split_sentences(raw_text)
    print(f"Extracted {len(sentences)} usable sentences.")

    # De-duplicate identical sentences (common with headers/boilerplate).
    seen = set()
    deduped = []
    for s in sentences:
        key = s.lower()
        if key not in seen:
            seen.add(key)
            deduped.append(s)
    print(f"{len(deduped)} sentences after de-duplication.")

    chunks = chunk_sentences(deduped)
    random.shuffle(chunks)
    print(f"Built {len(chunks)} training chunks (~{CHUNK_TARGET_WORDS} words each).")

    if len(chunks) < 20:
        print(
            "\nWarning: that's very little data. Fine-tuning will still run, but "
            "gains will be small/noisy. More clean domain text = a better model."
        )

    if len(chunks) < 5:
        # Too few chunks for a meaningful held-out split -- reuse all
        # of them for both, just to keep the training script runnable.
        # (You really want more data than this -- see warning above.)
        print("Too few chunks for a train/val split -- using all chunks for both.")
        train_chunks = chunks
        val_chunks = chunks
    else:
        n_val = max(1, int(len(chunks) * VAL_FRACTION))
        n_val = min(n_val, len(chunks) - 1)  # always leave >=1 chunk for training
        val_chunks = chunks[:n_val]
        train_chunks = chunks[n_val:]

    os.makedirs(args.output_dir, exist_ok=True)
    train_path = os.path.join(args.output_dir, "train.txt")
    val_path = os.path.join(args.output_dir, "val.txt")

    with open(train_path, "w", encoding="utf-8") as f:
        f.write("\n".join(train_chunks))
    with open(val_path, "w", encoding="utf-8") as f:
        f.write("\n".join(val_chunks))

    print(f"\nWrote {len(train_chunks)} train examples -> {train_path}")
    print(f"Wrote {len(val_chunks)} val examples   -> {val_path}")
    print("\nNext: python finetune_bert.py")


if __name__ == "__main__":
    main()
