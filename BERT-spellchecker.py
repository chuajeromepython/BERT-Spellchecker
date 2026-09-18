"""
spellcheck_standalone.py

Standalone text corrector for OCR output -- completely decoupled from
the OCR pipeline. Paste or type text in, get corrected text out.
Useful for testing correction quality on its own, independent of
whatever PaddleOCR produced.

--- Why two stages ---
Two different failure modes need two different fixes:

  STAGE 1 -- SymSpell `lookup_compound` (non-neural, noisy-channel /
  frequency-based). This does spelling correction *and* word
  segmentation (splitting merged words, joining wrongly-split words)
  in a single pass using edit distance + word-frequency statistics --
  what fixes boundary errors like "Pcontinuid oudying" -> "continued
  studying" style corruption. It has no concept of sentence context,
  though, so among several similarly-scoring candidates it can pick
  the wrong one (e.g. picking "buying" over "studying").

  STAGE 2 -- BERT masked-word reranking (encoder, not a generative/
  decoder LLM -- it scores how well a candidate fits, it doesn't write
  new text). An earlier version of this script used the
  `contextualSpellCheck` library here, but that library can only fix
  NON-WORD errors (tokens it doesn't recognize as any real word at
  all) -- it has no way to tell that a real, correctly-spelled word
  is simply wrong for the sentence (a REAL-WORD error), which is
  exactly what Stage 1 sometimes produces. So this stage is a custom
  noisy-channel reranker instead: for each word, it finds the original
  raw OCR token it came from, asks SymSpell for several spelling-
  neighbor candidates from that raw token, masks the word's position
  in the Stage-1 sentence, and asks BERT which candidate actually fits
  the context -- then keeps the best-scoring one.

Both stages are protected by the same custom vocabulary list so your
domain terms/acronyms/proper nouns (STARS, iStar, GROQ, Xampp, etc.)
survive both passes untouched.

--- Known limitation ---
Stage 2's word-level alignment between the raw and Stage-1 text is
best-effort. When Stage 1 splits one garbled word into two real words
("continerd" -> "cont nerd"), both halves are already valid dictionary
words individually, so nothing downstream flags them as wrong -- this
is a structural gap (fixing it requires a merge-aware reranker, not
just a per-word one), not a bug, and worth naming as a limitation if
you're writing this up.

--- Install ---
    python -m pip install symspellpy transformers torch

--- Usage ---
    python spellcheck_standalone.py

Then paste/type your text. On Windows, finish input with:
    Ctrl+Z, then Enter
(On Mac/Linux it's Ctrl+D instead.)
"""

import os
import re
import difflib
import importlib.resources
from symspellpy import SymSpell, Verbosity


# Add any domain-specific terms/acronyms/proper nouns your essays use
# so neither stage tries to "correct" them -- extend this as you run
# into false flags. Case doesn't matter for the SymSpell dictionary
# lookup, but keep the casing here how you want it to appear if a
# stage inserts it fresh.
CUSTOM_VOCAB = [
    "readme", "groq", "xampp", "opencv", "php", "laravel", "api",
    "github", "python", "javascript", "html", "css", "sql", "json",
    "vscode", "istars", "istar", "stars", "paddleocr", "trocr",
]

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
CUSTOM_VOCAB_PATH = os.path.join(SCRIPT_DIR, "custom_vocab.txt")

# SymSpell tuning. Edit distance 3 (vs. the old 2) gives room for the
# genuinely mangled words without going so loose it starts corrupting
# already-correct text -- tune this per your OCR's typical error rate.
SYMSPELL_MAX_EDIT_DIST = 3
SYMSPELL_PREFIX_LEN = 7

# High frequency count for custom vocab entries -- makes SymSpell
# strongly prefer keeping/recognizing them over "correcting" them
# into a similar-looking dictionary word.
CUSTOM_VOCAB_FREQUENCY = 10_000_000


def build_symspell():
    print("Loading SymSpell dictionary (unigram + bigram)...")
    sym_spell = SymSpell(
        max_dictionary_edit_distance=SYMSPELL_MAX_EDIT_DIST,
        prefix_length=SYMSPELL_PREFIX_LEN,
    )
    dict_path = str(
        importlib.resources.files("symspellpy").joinpath(
            "frequency_dictionary_en_82_765.txt"
        )
    )
    bigram_path = str(
        importlib.resources.files("symspellpy").joinpath(
            "frequency_bigramdictionary_en_243_342.txt"
        )
    )
    sym_spell.load_dictionary(dict_path, term_index=0, count_index=1)
    sym_spell.load_bigram_dictionary(bigram_path, term_index=0, count_index=2)

    # Inject custom vocab with a high frequency so lookup_compound and
    # single-word lookup both treat these as well-known words instead
    # of "correcting" them, AND so they become candidates other garbled
    # tokens can be corrected *into* (e.g. "Ptp" -> "php").
    for term in CUSTOM_VOCAB:
        sym_spell.create_dictionary_entry(term.lower(), CUSTOM_VOCAB_FREQUENCY)

    return sym_spell


FINETUNED_MODEL_DIR = os.path.join(SCRIPT_DIR, "finetuned-bert-spellchecker")


def build_bert_scorer():
    # Plain transformers, not contextualSpellCheck -- we need direct
    # access to masked-position logits to SCORE specific candidate
    # words, not just contextualSpellCheck's built-in yes/no "is this
    # token even a known word" gate (see module docstring). BERT is an
    # encoder used here purely as a probability scorer, not a
    # generative/decoder LLM.
    import torch
    from transformers import AutoTokenizer, AutoModelForMaskedLM

    if os.path.isdir(FINETUNED_MODEL_DIR):
        model_source = FINETUNED_MODEL_DIR
        print(f"Loading fine-tuned BERT from {FINETUNED_MODEL_DIR} for contextual scoring...")
    else:
        model_source = "bert-base-uncased"
        print("Loading BERT (bert-base-uncased) for contextual scoring...")
        print("(No fine-tuned model found -- run finetune_bert.py to train one on your data.)")

    tokenizer = AutoTokenizer.from_pretrained(model_source)
    model = AutoModelForMaskedLM.from_pretrained(model_source)
    model.eval()
    return tokenizer, model, torch


def read_multiline_input():
    print("\nPaste or type your text below.")
    print("When finished: Windows = Ctrl+Z then Enter | Mac/Linux = Ctrl+D\n")
    lines = []
    try:
        while True:
            lines.append(input())
    except EOFError:
        pass
    return "\n".join(lines)


def stage1_symspell_compound(sym_spell, text):
    """
    Segmentation + spelling correction, line by line so line breaks
    in the original OCR output are preserved (lookup_compound works
    on a single string without embedded newlines).
    """
    corrected_lines = []
    for line in text.split("\n"):
        if not line.strip():
            corrected_lines.append(line)
            continue
        result = sym_spell.lookup_compound(
            line,
            max_edit_distance=SYMSPELL_MAX_EDIT_DIST,
            ignore_non_words=True,  # leave numbers/punctuation-only tokens alone
        )
        corrected_lines.append(result[0].term if result else line)
    return "\n".join(corrected_lines)


def restore_casing(original_text, symspell_output):
    """
    lookup_compound lowercases everything. Re-title-case the first
    letter of each sentence and re-uppercase any token that matches a
    custom-vocab acronym, so output isn't all-lowercase.
    """
    vocab_upper = {w.lower(): w.upper() for w in CUSTOM_VOCAB if len(w) <= 6}

    def fix_token(tok):
        low = tok.lower()
        if low in vocab_upper:
            return vocab_upper[low]
        return tok

    sentences = re.split(r"(?<=[.!?])\s+", symspell_output)
    fixed_sentences = []
    for sent in sentences:
        words = sent.split(" ")
        words = [fix_token(w) for w in words]
        if words and words[0]:
            words[0] = words[0][0].upper() + words[0][1:]
        fixed_sentences.append(" ".join(words))
    return " ".join(fixed_sentences)


# ---------------------------------------------------------------------
# STAGE 2 -- real-word error correction via noisy-channel reranking.
#
# contextualSpellCheck (a prior version of this script) can only flag
# a token as wrong if it's outright unrecognized. It has no way to ask
# "is this a real word that's simply wrong here?" -- so it can't touch
# cases like Stage 1 producing "buying" when the OCR/context clearly
# wanted "studying". That's a distinct, well-known problem in spell-
# checking called a REAL-WORD ERROR (as opposed to a non-word error),
# and it needs a different technique: generate several spelling-
# neighbor candidates for the word, then use a language model to see
# which candidate actually fits the sentence, and pick that one.
#
# Concretely, per word:
#   1. Find which ORIGINAL raw token this Stage-1 word came from
#      (word-level alignment via difflib), since candidates must be
#      generated from the raw corruption, not from Stage 1's already-
#      committed (and possibly wrong) guess.
#   2. Ask SymSpell for several dictionary-word candidates within edit
#      distance of that raw token (the "channel model": what could a
#      real word have looked like once OCR-mangled into this?).
#   3. Mask that word's position in the Stage-1 sentence and ask BERT
#      for the probability of each candidate at that position (the
#      "language model": which candidate actually fits the context?).
#   4. Keep whichever candidate BERT scores highest.
# ---------------------------------------------------------------------

def strip_punct(token):
    return re.sub(r"^\W+|\W+$", "", token)


def align_raw_to_stage1(raw_tokens, stage1_tokens):
    """
    Best-effort word-level alignment: for each Stage-1 token, find the
    original raw token it most likely descended from, using
    difflib's sequence matching over normalized (lowercased,
    punctuation-stripped) tokens.

    Only clean 1:1 blocks (one raw token became one Stage-1 token,
    whether changed or not) get a mapping -- positions inside a block
    where the raw/Stage-1 token counts differ (Stage 1 merged or split
    words, e.g. "continerd" -> "cont nerd", or "codebase" -> "code
    base") are marked as None. Reranking a single word against a
    multi-word split doesn't make sense with this candidate-generation
    approach, so those stay a known limitation instead of risking
    nonsense output (see module docstring).
    """
    raw_norm = [strip_punct(t).lower() for t in raw_tokens]
    stage1_norm = [strip_punct(t).lower() for t in stage1_tokens]
    sm = difflib.SequenceMatcher(None, raw_norm, stage1_norm, autojunk=False)

    mapping = {}
    for tag, i1, i2, j1, j2 in sm.get_opcodes():
        raw_slice = raw_tokens[i1:i2]
        stage1_len = j2 - j1
        if len(raw_slice) == stage1_len:
            for k, j in enumerate(range(j1, j2)):
                mapping[j] = raw_slice[k]
        else:
            for j in range(j1, j2):
                mapping[j] = None  # merge/split block -- don't rerank individually
    return mapping


def generate_candidates(sym_spell, raw_token, current_token, max_k=8):
    core = strip_punct(raw_token).lower()
    if not core:
        return []
    suggestions = sym_spell.lookup(
        core, Verbosity.ALL, max_edit_distance=SYMSPELL_MAX_EDIT_DIST
    )
    candidates = [s.term for s in suggestions[:max_k]]
    current_core = strip_punct(current_token).lower()
    if current_core and current_core not in candidates:
        candidates.append(current_core)
    return list(dict.fromkeys(candidates))


def rerank_line_with_bert(sym_spell, tokenizer, model, torch_mod, raw_line, stage1_line):
    raw_tokens = raw_line.split()
    stage1_tokens = stage1_line.split()
    if not stage1_tokens:
        return stage1_line

    mapping = align_raw_to_stage1(raw_tokens, stage1_tokens)
    new_tokens = list(stage1_tokens)
    custom_vocab_lower = {w.lower() for w in CUSTOM_VOCAB}

    for idx, tok in enumerate(stage1_tokens):
        core = strip_punct(tok).lower()
        if len(core) < 3 or not core.isalpha():
            continue  # skip short tokens/dates/acronym-fragments, not worth scoring

        if core in custom_vocab_lower:
            continue  # trust Stage 1's high-confidence domain-vocab match as-is

        raw_source = mapping.get(idx)
        if raw_source is None:
            continue  # inside a merge/split block, or unmapped -- don't touch

        raw_core = strip_punct(raw_source).lower()
        if raw_core == core:
            continue  # Stage 1 didn't change this word -- nothing to correct,
            # and reranking an already-fine word only risks swapping it for
            # whatever BERT happens to rate slightly higher in an odd sentence

        candidates = generate_candidates(sym_spell, raw_source, tok)

        # Only candidates that map to a SINGLE whole BERT token can be
        # scored directly by reading one logit off the mask position.
        # Multi-subword candidates are skipped rather than approximated.
        scoreable = [c for c in candidates if len(tokenizer.tokenize(c)) == 1]
        if len(scoreable) < 2:
            continue  # nothing to choose between

        masked_tokens = list(stage1_tokens)
        masked_tokens[idx] = tokenizer.mask_token
        encoding = tokenizer(" ".join(masked_tokens), return_tensors="pt")
        mask_positions = (
            encoding["input_ids"][0] == tokenizer.mask_token_id
        ).nonzero(as_tuple=True)[0]
        if len(mask_positions) == 0:
            continue
        mask_pos = mask_positions[0].item()

        with torch_mod.no_grad():
            logits = model(**encoding).logits[0, mask_pos]
        log_probs = torch_mod.log_softmax(logits, dim=-1)

        best_candidate, best_score = None, float("-inf")
        for cand in scoreable:
            cand_id = tokenizer.convert_tokens_to_ids(cand)
            score = log_probs[cand_id].item()
            if score > best_score:
                best_candidate, best_score = cand, score

        if best_candidate and best_candidate != core:
            if tok[:1].isupper():
                best_candidate = best_candidate[:1].upper() + best_candidate[1:]
            trailing_punct = re.sub(r"^\w+", "", tok)
            new_tokens[idx] = best_candidate + trailing_punct

    return " ".join(new_tokens)


def stage2_bert_rerank(sym_spell, tokenizer, model, torch_mod, raw_text, stage1_text):
    raw_lines = raw_text.split("\n")
    stage1_lines = stage1_text.split("\n")
    # Lines should correspond 1:1 (Stage 1 preserves line breaks).
    out_lines = []
    for i, s1_line in enumerate(stage1_lines):
        raw_line = raw_lines[i] if i < len(raw_lines) else s1_line
        if not s1_line.strip():
            out_lines.append(s1_line)
            continue
        out_lines.append(
            rerank_line_with_bert(sym_spell, tokenizer, model, torch_mod, raw_line, s1_line)
        )
    return "\n".join(out_lines)


def correct_text(sym_spell, tokenizer, model, torch_mod, text):
    stage1_out = stage1_symspell_compound(sym_spell, text)
    stage1_out = restore_casing(text, stage1_out)
    stage2_out = stage2_bert_rerank(sym_spell, tokenizer, model, torch_mod, text, stage1_out)
    return stage1_out, stage2_out


def main():
    sym_spell = build_symspell()
    tokenizer, model, torch_mod = build_bert_scorer()

    raw_text = read_multiline_input()
    if not raw_text.strip():
        print("No text entered -- exiting.")
        return

    print("\nRunning Stage 1 (SymSpell compound: segmentation + frequency correction)...")
    print("Running Stage 2 (BERT reranking of real-word errors)...")
    stage1_text, final_text = correct_text(sym_spell, tokenizer, model, torch_mod, raw_text)

    print("\n--- Original ---")
    print(raw_text)
    print("\n--- After Stage 1 (SymSpell) ---")
    print(stage1_text)
    print("\n--- After Stage 2 (BERT rerank, final) ---")
    print(final_text)

    save = input("\nSave corrected text to a file? (y/n): ").strip().lower()
    if save == "y":
        out_path = input("Filename (e.g. corrected.txt): ").strip() or "corrected.txt"
        with open(out_path, "w", encoding="utf-8") as f:
            f.write(final_text)
        print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()