"""
Evaluate the fine-tuned LangID model on the clean test set and every
robustness slice. Produces per-language precision / recall / F1, macro-F1
per slice, the top confusion pairs, and a confidence-vs-accuracy table you
can use to pick the fallback-routing threshold.

Outputs:
  reports/metrics.json
  reports/REPORT.md          (paste the tables into your README)

Usage:
  python scripts/evaluate.py --data data --model model --out reports
"""
import argparse
import json
import os
from collections import Counter

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import precision_recall_fscore_support, f1_score
from transformers import AutoModelForSequenceClassification, AutoTokenizer

SLICES = ["clean", "short", "noisy", "code_mixed", "mixed_script"]


@torch.no_grad()
def predict(model, tok, texts, device, bs=128, max_length=64):
    probs_all = []
    for i in range(0, len(texts), bs):
        enc = tok(texts[i:i + bs], truncation=True, max_length=max_length,
                  padding=True, return_tensors="pt").to(device)
        logits = model(**enc).logits
        probs_all.append(torch.softmax(logits, dim=-1).cpu().numpy())
    return np.concatenate(probs_all, axis=0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data")
    ap.add_argument("--model", default="model")
    ap.add_argument("--out", default="reports")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    tok = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForSequenceClassification.from_pretrained(args.model).to(device).eval()
    # label order: prefer labels.json written by train.py, else the config
    # (config keys may be str or int depending on the transformers version)
    labels_path = os.path.join(args.model, "labels.json")
    if os.path.exists(labels_path):
        labels = json.load(open(labels_path))
    else:
        id2label = {int(k): v for k, v in model.config.id2label.items()}
        labels = [id2label[i] for i in range(len(id2label))]
    label2id = {l: i for i, l in enumerate(labels)}
    assert len(labels) == model.config.num_labels, "labels.json does not match the model head"
    print(f"device={device} labels={len(labels)}")

    results = {"slices": {}, "per_language_clean": {}, "top_confusions_clean": [],
               "confidence_routing_clean": []}
    md = ["# Polyglot-120 evaluation report\n"]
    md.append("| Slice | N | Macro-F1 | Accuracy |\n|---|---:|---:|---:|")

    for s in SLICES:
        path = os.path.join(args.data, f"test_{s}.csv")
        if not os.path.exists(path):
            continue
        df = pd.read_csv(path, dtype=str, keep_default_na=False, na_filter=False)
        df["text"] = df["text"].astype(str).str.strip()
        df = df[(df["text"] != "") & df["label"].isin(label2id)]
        y = df["label"].map(label2id).to_numpy()
        probs = predict(model, tok, df["text"].astype(str).tolist(), device)
        pred = probs.argmax(axis=1)
        conf = probs.max(axis=1)
        macro = f1_score(y, pred, average="macro")
        acc = float((pred == y).mean())
        results["slices"][s] = {"n": int(len(y)), "macro_f1": float(macro), "accuracy": acc}
        md.append(f"| {s} | {len(y)} | {macro:.4f} | {acc:.4f} |")

        if s == "clean":
            p, r, f, sup = precision_recall_fscore_support(y, pred, labels=list(range(len(labels))), zero_division=0)
            for i, l in enumerate(labels):
                results["per_language_clean"][l] = {"precision": float(p[i]), "recall": float(r[i]),
                                                    "f1": float(f[i]), "support": int(sup[i])}
            conf_pairs = Counter((labels[a], labels[b]) for a, b in zip(y, pred) if a != b)
            results["top_confusions_clean"] = [{"true": a, "pred": b, "count": c}
                                               for (a, b), c in conf_pairs.most_common(20)]
            for thr in [0.5, 0.6, 0.7, 0.8, 0.9, 0.95]:
                keep = conf >= thr
                results["confidence_routing_clean"].append({
                    "threshold": thr, "coverage": float(keep.mean()),
                    "accuracy_on_kept": float((pred[keep] == y[keep]).mean()) if keep.any() else None,
                    "accuracy_on_routed": float((pred[~keep] == y[~keep]).mean()) if (~keep).any() else None,
                })

    # worst languages on clean
    per = results["per_language_clean"]
    worst = sorted(per.items(), key=lambda kv: kv[1]["f1"])[:15]
    md.append("\n## 15 lowest-F1 languages (clean)\n")
    md.append("| Language | P | R | F1 | n |\n|---|---:|---:|---:|---:|")
    for l, m in worst:
        md.append(f"| {l} | {m['precision']:.3f} | {m['recall']:.3f} | {m['f1']:.3f} | {m['support']} |")

    md.append("\n## Top confusion pairs (clean)\n")
    md.append("| True | Predicted | Count |\n|---|---|---:|")
    for c in results["top_confusions_clean"]:
        md.append(f"| {c['true']} | {c['pred']} | {c['count']} |")

    md.append("\n## Confidence routing (clean): pick your fallback threshold\n")
    md.append("| Threshold | Coverage | Acc on kept | Acc on routed |\n|---:|---:|---:|---:|")
    for row in results["confidence_routing_clean"]:
        ak = f"{row['accuracy_on_kept']:.4f}" if row["accuracy_on_kept"] is not None else "-"
        ar = f"{row['accuracy_on_routed']:.4f}" if row["accuracy_on_routed"] is not None else "-"
        md.append(f"| {row['threshold']} | {row['coverage']:.3f} | {ak} | {ar} |")

    json.dump(results, open(os.path.join(args.out, "metrics.json"), "w"), indent=2)
    open(os.path.join(args.out, "REPORT.md"), "w", encoding="utf-8").write("\n".join(md) + "\n")
    print("\n".join(md))


if __name__ == "__main__":
    main()
