<p align="center">
  <img src="https://img.shields.io/badge/🤖-BERT--Spellchecker-2b2b2b?style=for-the-badge&labelColor=1a1a1a" alt="BERT-Spellchecker" height="60">
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

## 2. Install Git (if not already installed)

Check with:

```
git --version
```

If it's not found, download it from [git-scm.com](https://git-scm.com/downloads) and install with default options.

## 3. Download the project

Open a terminal, navigate to wherever you want the folder to live (e.g. your Desktop), and run:

```
git clone https://github.com/chuajeromepython/BERT-Spellchecker.git
cd BERT-Spellchecker
```

## 4. Install the required packages

Still in that folder, run:

```
pip install -r requirements.txt
```

This installs three packages: `symspellpy`, `transformers`, and `torch`. It may take a few minutes — `torch` in particular is a large download.

If `pip` gives an error saying it's not recognized, try `pip3` or `python -m pip install -r requirements.txt` instead.

## 5. Run the script

```
python BERT-spellchecker.py
```

**The first time you run it**, you need an internet connection — it will automatically download the BERT language model (a few hundred MB). This only happens once; after that it's cached and works offline.

You'll see some loading messages first:

```
Loading SymSpell dictionary (unigram + bigram)...
Loading BERT (bert-base-uncased) for contextual scoring...
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
| Install takes a long time | Normal — `torch` is a large package |
| First run needs internet | Expected — it downloads the BERT model once |
| Script seems to hang after loading | It's likely just processing; larger text takes longer |
