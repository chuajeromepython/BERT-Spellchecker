<p align="center">
  <img src="https://karmajack.com/wp-content/uploads/2019/12/Google-BERT-What-you-probably-didnt-know-about-the-AI.png" alt="BERT-Spellchecker" height="250" width="400">
</p>

<p align="center">OCR text correction: SymSpell segmentation + BERT contextual reranking</p>

<p align="center">
  <img src="https://img.shields.io/badge/python-3.9%2B-blue?style=flat-square" alt="python version">
  <img src="https://img.shields.io/badge/model-bert--base--uncased-yellow?style=flat-square" alt="model">
  <img src="https://img.shields.io/badge/license-MIT-lightgrey?style=flat-square" alt="license">
</p>

---

A standalone text corrector for OCR output. It's a two-stage pipeline:

1. **SymSpell** — fixes spelling and word-boundary errors (like merged or split words).
2. **BERT** — re-checks the result and fixes "real-word" errors (a correctly spelled word that's just wrong for the sentence).

It runs entirely on your own computer, in a terminal — no web app, no upload. You paste text in, it prints the corrected text back out.

---

## 1. Install Python

You need Python **3.9–3.12** (3.10 or 3.11 is safest if you're installing fresh).

1. Go to [python.org/downloads](https://www.python.org/downloads/)
2. Download and run the installer for your OS.
3. **On Windows**, on the first install screen, check the box that says **"Add Python to PATH"** before clicking Install. This step is easy to miss and causes the most common setup problems.

To check it worked, open a terminal (Command Prompt / PowerShell / Terminal) and run:

```
python --version
```

You should see something like `Python 3.11.x`. If that command isn't recognized, try `python3 --version` instead.

## 2. Install Git and Git LFS

Check Git is installed:

```
git --version
```

If it's not found, download it from [git-scm.com](https://git-scm.com/downloads) and install with default options.

This project also ships a fine-tuned BERT model (a few hundred MB) stored using **Git LFS** (Large File Storage), since GitHub blocks regular files over 100MB. You need Git LFS installed once per machine, or cloning will fail or only download placeholder files instead of the real model:

1. Download the installer from [git-lfs.com](https://git-lfs.com/), or on Windows with winget: `winget install GitHub.GitLFS`
2. Then run, once:
```
git lfs install
```

## 3. Download the project

Open a terminal, navigate to wherever you want the folder to live (e.g. your Desktop), and run:

```
git clone https://github.com/chuajeromepython/BERT-Spellchecker.git
cd BERT-Spellchecker
```

Because of the Git LFS setup above, this clone will also download the fine-tuned model — expect it to take longer than a typical small-repo clone, since it's pulling several hundred MB.

## 4. Install the required packages

Still in that folder, run:

```
pip install -r requirements.txt
pip install accelerate
```

This installs `symspellpy`, `transformers`, and `torch` from requirements.txt, plus `accelerate` (needed only if you plan to fine-tune the model yourself — see Section 9 — but harmless to install either way). It may take a few minutes — `torch` in particular is a large download.

If `pip` gives an error saying it's not recognized, try `pip3` or `python -m pip install -r requirements.txt` instead. If packages install successfully but the script still can't find them, your `python` and `pip` commands may point to two different Python installations on your machine — check with `python -c "import sys; print(sys.executable)"` and `pip -V`, and if the paths differ, install using `python -m pip install ...` instead of plain `pip install ...`.

## 5. Run the script

```
python BERT-spellchecker.py
```

**No internet connection is needed to run this script**, since the fine-tuned model (`finetuned-bert-spellchecker/`) ships with the repo via Git LFS — the script loads it entirely from local files. (The only time this script needs internet is if that folder is ever missing and it has to fall back to downloading the stock `bert-base-uncased` model from Hugging Face — see below.)

You'll see some loading messages first. Normally, since the fine-tuned model comes with the repo, you'll see:

```
Loading SymSpell dictionary (unigram + bigram)...
Loading fine-tuned BERT from .../finetuned-bert-spellchecker for contextual scoring...
```

If that folder is ever missing (e.g. you're working from a copy that intentionally excluded it), the script falls back automatically to the stock model instead — and in that case only, it needs internet on first run to download it, then caches it locally for offline use after that:

```
Loading SymSpell dictionary (unigram + bigram)...
Loading BERT (bert-base-uncased) for contextual scoring...
(No fine-tuned model found -- run finetune_bert.py to train one on your data.)
```

## 6. Enter your text

Once you see this prompt:

```
Paste or type your text below.
When finished: Windows = Ctrl+Z then Enter | Mac/Linux = Ctrl+D
```

Paste or type in the OCR text you want corrected. It can be multiple lines.

**To finish and submit the text:**
- **Windows:** press `Ctrl+Z`, then press `Enter`
- **Mac/Linux:** press `Ctrl+D`

## 7. Reading the output

The script prints three sections:

```
--- Original ---
(exactly what you typed in)

--- After Stage 1 (SymSpell) ---
(spelling + word-boundary corrections applied)

--- After Stage 2 (BERT rerank, final) ---
(the final, best-corrected version — this is the one you want)
```

**The "After Stage 2" section is the final answer.** The other two are shown so you can see what each stage changed, but you don't need to do anything with them.

## 8. Saving the result

After the output prints, you'll be asked:

```
Save corrected text to a file? (y/n):
```

- Type `y` and press Enter to save it, then type a filename when asked (e.g. `corrected.txt`). It will save in the same folder you're running the script from.
- Type `n` to skip saving — the text will still be visible in the terminal, but won't be saved anywhere.

---

## 9. Fine-tuning on your own data (optional)

The model that ships with this repo was fine-tuned on the [TU Darmstadt Argument Annotated Essays corpus](https://tudatalib.ulb.tu-darmstadt.de/handle/tudatalib/2422), a collection of college-level argumentative essays. If you want to improve it further with your own domain-specific text, you don't need error/correct pairs — just clean, correctly-spelled text similar to what you'll actually be correcting. BERT is used here purely as a masked-word scorer, so continuing its training on domain text is enough to teach it that domain's vocabulary and phrasing.

**Steps:**

1. Put `.txt` files of clean, correct text into `finetune_data/raw/` (any filenames).
2. Prepare the training data:
   ```
   python prepare_finetune_data.py
   ```
   This reads everything in `finetune_data/raw/`, cleans and chunks it, and writes `finetune_data/train.txt` and `finetune_data/val.txt`.
3. Run the actual fine-tuning:
   ```
   python finetune_bert.py
   ```
   This is the slow step — expect roughly 1–2 hours on a CPU-only machine, depending on how much data you have. It prints progress per step, plus a loss number (should trend downward) and an eval loss at the end of each epoch.
4. Once done, it saves the result to `finetuned-bert-spellchecker/`, which `BERT-spellchecker.py` will automatically use on its next run — no code changes needed.

**Note on training checkpoints:** while training, the script also writes intermediate checkpoints to `finetuned-bert-spellchecker/checkpoints/`. These are only needed if training crashes and you want to resume — once training finishes successfully, that subfolder is safe to delete (it can be several GB, and `.gitignore` already excludes it from being committed).

**If you retrain and want to share the new model:** commit and push as usual — `.gitattributes` already routes `finetuned-bert-spellchecker/` through Git LFS, so no extra setup is needed on your end. Anyone pulling the repo afterward just needs Git LFS installed (Section 2) to get the new weights automatically.

---

## 10. Working on accuracy

If you're improving how well the corrector performs (rather than setting up the project for the first time), there's a dedicated test harness in `accuracy_tests/` — a fixed set of sentences with known-correct answers, plus a script that scores the pipeline against them. This turns "does it seem better?" into an actual number you can compare before and after a change.

**Steps:**

1. Make sure you've completed Sections 1–5 above first (Python, Git LFS, the project cloned, packages installed) — the test harness runs the real pipeline, so everything it depends on needs to already be working.

2. Run the baseline test:
   ```
   cd accuracy_tests
   python run_accuracy_tests.py
   ```
   This loads the model (same startup messages as running `BERT-spellchecker.py` directly), runs all 20 test sentences through both stages, and prints a result for each one.

3. **Read the report.** Each case is marked:
   - `PASS` — output matches exactly, including punctuation.
   - `PASS*` — every word is correct; only punctuation/whitespace differs. (Punctuation loss is a known, separate behavior of Stage 1 — see the note the script prints — not a spelling-correction failure.)
   - `CLOSE` — mostly right, a small number of words differ.
   - `FAIL` — meaningfully wrong. These are the ones worth digging into.

   For each `FAIL` or `CLOSE` case, the script prints exactly which words differ (`-> expected [...] but got [...]`), so you don't have to eyeball the full sentence to spot the problem.

4. **Pick one `FAIL` case and investigate why.** Common causes and where to look:
   - Wrong word chosen for a typo → `SYMSPELL_MAX_EDIT_DIST` or the candidate-ranking logic in `BERT-spellchecker.py`.
   - A domain term getting "corrected" into something else → add it to `CUSTOM_VOCAB` in `BERT-spellchecker.py`.
   - An already-correct word getting changed anyway (over-correction) → look at `rerank_line_with_bert()` in `BERT-spellchecker.py`.
   - The model seems to consistently misunderstand domain-specific phrasing (not a single parameter issue) → may need more/better fine-tuning data instead — see Section 9.

5. **Make one change at a time**, then re-run:
   ```
   python run_accuracy_tests.py
   ```
   The script automatically compares this run against your previous one and prints the score difference, so you'll immediately see whether that specific change actually helped, hurt, or did nothing.

6. **Repeat** — pick the next `FAIL`/`CLOSE` case, change one thing, re-run, compare. Avoid changing multiple things between runs; if the score moves, you want to know which change caused it.

7. If you want to add a new test case (e.g. a real failure you found while using the tool normally), add it to `accuracy_tests/test_cases.json` following the same format as the existing entries (`id`, `category`, `input`, `expected`, `notes`), then re-run to include it going forward.

8. When you're done for the session, push your changes:
   ```
   git add BERT-spellchecker.py accuracy_tests/test_cases.json
   git commit -m "Describe what you changed and why"
   git push origin main
   ```
   Note: `accuracy_tests/results/` (the saved score history) is intentionally excluded from Git via `.gitignore` — it's personal run history, not something that needs to be shared.

---

## Running it again later

You don't need to reinstall anything. Just open a terminal, go back into the folder, and run:

```
cd BERT-Spellchecker
python BERT-spellchecker.py
```

## Troubleshooting

| Problem | Fix |
|---|---|
| `python` / `pip` not recognized | Try `python3` / `pip3`, or reinstall Python and check "Add to PATH" |
| Packages installed but script still can't find them (`ModuleNotFoundError`) | `python` and `pip` may point to different Python installs. Check with `python -c "import sys; print(sys.executable)"` and `pip -V` — if the paths differ, use `python -m pip install ...` instead of plain `pip install ...` |
| Install takes a long time | Normal — `torch` is a large package |
| First run needs internet (only if the fine-tuned model folder is missing) | Expected in that case only — it falls back to downloading the stock BERT model once and caches it |
| Script seems to hang after loading | It's likely just processing; larger text takes longer |
| Clone is very slow, or only downloads small placeholder files | Make sure Git LFS is installed (`git lfs install`) *before* cloning — see Section 2. Without it, you'll get tiny LFS pointer files instead of the real model |
| Clone/checkout errors mentioning "smudge filter lfs failed" or "LFS: Authorization error" | Usually means Git LFS isn't installed/initialized on your machine, or a network/firewall is blocking GitHub's LFS storage host. Confirm `git lfs install` has been run, then retry the clone |
| `finetune_bert.py` fails with an argument error on `TrainingArguments` | Some `transformers` versions have removed older arguments (e.g. `overwrite_output_dir`). If you hit this, it usually means the script needs a small update for your installed version — check what argument is unrecognized and remove/rename it |
| `git push` fails or hangs on a large commit | The fine-tuned model is large — make sure it's being tracked via Git LFS (`git lfs track "finetuned-bert-spellchecker/**"`) rather than committed as a normal file, or GitHub will reject anything over 100MB |
