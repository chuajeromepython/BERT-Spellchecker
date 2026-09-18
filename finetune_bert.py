"""
finetune_bert.py

Fine-tunes bert-base-uncased on your domain text using the same
masked-language-modeling (MLM) objective BERT was originally trained
with -- this is what Stage 2 of BERT-spellchecker.py relies on to
score candidate words. Continuing this training on your own clean
text teaches the model your domain's vocabulary and typical phrasing,
so its scoring is sharper for your use case.

--- Before running this ---
    python prepare_finetune_data.py --input_dir finetune_data/raw --output_dir finetune_data
This needs finetune_data/train.txt and finetune_data/val.txt to exist.

--- Install (one extra package beyond the base spellchecker) ---
    python -m pip install accelerate

--- Usage ---
    python finetune_bert.py
    python finetune_bert.py --epochs 5 --batch_size 8

--- Output ---
Saves the fine-tuned model + tokenizer to ./finetuned-bert-spellchecker/
BERT-spellchecker.py automatically uses this folder instead of the
stock bert-base-uncased if it exists (see build_bert_scorer()).
"""

import argparse
import os

import torch
from torch.utils.data import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForMaskedLM,
    DataCollatorForLanguageModeling,
    Trainer,
    TrainingArguments,
)

BASE_MODEL = "bert-base-uncased"
DEFAULT_OUTPUT_DIR = "finetuned-bert-spellchecker"
MAX_SEQ_LEN = 256


class LineDataset(Dataset):
    """Tokenizes each line (chunk) up front; masking is applied later,
    dynamically, per-batch, by DataCollatorForLanguageModeling -- so
    each epoch sees different random masks rather than one fixed set,
    which is standard practice and makes better use of small datasets."""

    def __init__(self, path, tokenizer):
        with open(path, "r", encoding="utf-8") as f:
            lines = [l.strip() for l in f if l.strip()]
        self.encodings = tokenizer(
            lines,
            truncation=True,
            max_length=MAX_SEQ_LEN,
            padding=False,
        )

    def __len__(self):
        return len(self.encodings["input_ids"])

    def __getitem__(self, idx):
        return {k: v[idx] for k, v in self.encodings.items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", default="finetune_data")
    ap.add_argument("--output_dir", default=DEFAULT_OUTPUT_DIR)
    ap.add_argument("--epochs", type=float, default=3.0)
    ap.add_argument("--batch_size", type=int, default=8)
    ap.add_argument("--learning_rate", type=float, default=5e-5)
    ap.add_argument("--mlm_probability", type=float, default=0.15,
                     help="Fraction of tokens masked per example (0.15 is BERT's original default).")
    args = ap.parse_args()

    train_path = os.path.join(args.data_dir, "train.txt")
    val_path = os.path.join(args.data_dir, "val.txt")
    if not (os.path.exists(train_path) and os.path.exists(val_path)):
        raise SystemExit(
            f"Missing {train_path} / {val_path}. Run prepare_finetune_data.py first."
        )

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    print(f"Loading base model: {BASE_MODEL}")
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
    model = AutoModelForMaskedLM.from_pretrained(BASE_MODEL)

    print("Tokenizing data...")
    train_dataset = LineDataset(train_path, tokenizer)
    val_dataset = LineDataset(val_path, tokenizer)
    print(f"Train examples: {len(train_dataset)} | Val examples: {len(val_dataset)}")

    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=True,
        mlm_probability=args.mlm_probability,
    )

    training_args = TrainingArguments(
        output_dir=os.path.join(args.output_dir, "checkpoints"),
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        eval_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=2,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        logging_steps=25,
        report_to=[],  # no wandb/tensorboard by default
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=data_collator,
    )

    print("\nStarting fine-tuning...")
    trainer.train()

    eval_metrics = trainer.evaluate()
    print(f"\nFinal eval loss: {eval_metrics['eval_loss']:.4f}")

    print(f"\nSaving fine-tuned model to {args.output_dir}")
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)

    print(
        f"\nDone. BERT-spellchecker.py will automatically pick up "
        f"'{args.output_dir}/' the next time you run it."
    )


if __name__ == "__main__":
    main()
