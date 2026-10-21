"""
Fine-tune a multilingual transformer for language identification.

Default backbone: xlm-roberta-base (278M params, 100 pretraining languages).
On a free Colab T4 with ~120 languages x ~1,000 sentences, one epoch at
max_length=64 takes roughly 15-30 minutes.

Usage:
  python scripts/train.py --data data --out model --epochs 1
"""
import argparse
import inspect
import json
import math
import os

import numpy as np
import pandas as pd
import torch
from datasets import Dataset
from sklearn.metrics import f1_score
from transformers import (AutoModelForSequenceClassification, AutoTokenizer,
                          DataCollatorWithPadding, Trainer, TrainingArguments)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data")
    ap.add_argument("--out", default="model")
    ap.add_argument("--model_name", default="xlm-roberta-base")
    ap.add_argument("--epochs", type=float, default=1.0)
    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--lr", type=float, default=3e-5)
    ap.add_argument("--max_length", type=int, default=64)
    ap.add_argument("--val_frac", type=float, default=0.02)
    args = ap.parse_args()

    labels = json.load(open(os.path.join(args.data, "labels.json")))
    label2id = {l: i for i, l in enumerate(labels)}
    id2label = {i: l for l, i in label2id.items()}

    df = pd.read_csv(os.path.join(args.data, "train.csv"), dtype=str, keep_default_na=False, na_filter=False)
    df["text"] = df["text"].astype(str).str.strip()
    df = df[(df["text"] != "") & df["label"].isin(label2id)].copy()
    df["labels"] = df["label"].map(label2id)
    df = df.sample(frac=1.0, random_state=13).reset_index(drop=True)
    n_val = int(len(df) * args.val_frac)
    val_df, train_df = df.iloc[:n_val], df.iloc[n_val:]
    print(f"train={len(train_df)} val={len(val_df)} labels={len(labels)}")

    tok = AutoTokenizer.from_pretrained(args.model_name)

    def encode(batch):
        return tok(batch["text"], truncation=True, max_length=args.max_length)

    train_ds = Dataset.from_pandas(train_df[["text", "labels"]], preserve_index=False).map(
        encode, batched=True, remove_columns=["text"])
    val_ds = Dataset.from_pandas(val_df[["text", "labels"]], preserve_index=False).map(
        encode, batched=True, remove_columns=["text"])

    model = AutoModelForSequenceClassification.from_pretrained(
        args.model_name, num_labels=len(labels), id2label=id2label, label2id=label2id
    )

    def compute_metrics(p):
        preds = np.argmax(p.predictions, axis=-1)
        return {"macro_f1": f1_score(p.label_ids, preds, average="macro"),
                "accuracy": float((preds == p.label_ids).mean())}

    use_cuda = torch.cuda.is_available()

    # Linear warmup over the first 6% of optimizer steps. Expressed as an
    # explicit step count because `warmup_ratio` was removed in transformers 5.x
    # while `warmup_steps` exists in every version.
    steps_per_epoch = math.ceil(len(train_ds) / args.batch_size)
    total_steps = max(1, int(round(steps_per_epoch * args.epochs)))
    warmup_steps = int(0.06 * total_steps)

    targs_kwargs = dict(
        output_dir=os.path.join(args.out, "checkpoints"),
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size * 2,
        learning_rate=args.lr,
        num_train_epochs=args.epochs,
        warmup_steps=warmup_steps,
        weight_decay=0.01,
        save_strategy="no",
        logging_steps=50,
        fp16=use_cuda,
        report_to="none",
        dataloader_num_workers=2,
    )
    # TrainingArguments fields get renamed between transformers versions
    # (evaluation_strategy -> eval_strategy, warmup_ratio removed, ...).
    # Only pass what this installed version actually accepts.
    ta_params = set(inspect.signature(TrainingArguments.__init__).parameters)
    if "eval_strategy" in ta_params:
        targs_kwargs["eval_strategy"] = "epoch"
    elif "evaluation_strategy" in ta_params:
        targs_kwargs["evaluation_strategy"] = "epoch"
    for k in [k for k in targs_kwargs if k not in ta_params]:
        print(f"note: this transformers version has no TrainingArguments.{k}; skipping it")
        targs_kwargs.pop(k)
    print(f"steps/epoch={steps_per_epoch} total_steps={total_steps} warmup_steps={warmup_steps} fp16={use_cuda}")
    targs = TrainingArguments(**targs_kwargs)

    # tokenizer kwarg was renamed processing_class in newer versions
    tr_params = set(inspect.signature(Trainer.__init__).parameters)
    tok_kwarg = {"processing_class": tok} if "processing_class" in tr_params else {"tokenizer": tok}
    trainer = Trainer(
        model=model, args=targs, train_dataset=train_ds, eval_dataset=val_ds,
        data_collator=DataCollatorWithPadding(tok), compute_metrics=compute_metrics, **tok_kwarg,
    )
    trainer.train()
    print("validation:", trainer.evaluate())

    os.makedirs(args.out, exist_ok=True)
    trainer.save_model(args.out)
    tok.save_pretrained(args.out)
    json.dump(labels, open(os.path.join(args.out, "labels.json"), "w"), indent=2)
    assert os.path.exists(os.path.join(args.out, "config.json")), "model was not saved"
    print(f"saved to {args.out}")


if __name__ == "__main__":
    main()
