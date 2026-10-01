"""Fine-tune three-class DistilBERT and enforce the requested quality gates."""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from datasets import Dataset
from sklearn.metrics import accuracy_score, roc_auc_score
from transformers import AutoModelForSequenceClassification, AutoTokenizer, DataCollatorWithPadding, Trainer, TrainingArguments

LABELS = ["negative", "neutral", "positive"]
LABEL_TO_ID = {label: i for i, label in enumerate(LABELS)}


def read_csv(path: str) -> Dataset:
    frame = pd.read_csv(path).dropna(subset=["text", "label"])
    if not set(frame.label.unique()).issubset(LABEL_TO_ID):
        raise ValueError(f"labels must be one of {LABELS}")
    frame["labels"] = frame.label.map(LABEL_TO_ID)
    return Dataset.from_pandas(frame[["text", "labels"]], preserve_index=False)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", required=True)
    parser.add_argument("--test", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--epochs", type=float, default=3)
    args = parser.parse_args()
    output = Path(args.output)
    tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
    tokenise = lambda batch: tokenizer(batch["text"], truncation=True, max_length=256)
    train, test = read_csv(args.train).map(tokenise, batched=True), read_csv(args.test).map(tokenise, batched=True)
    model = AutoModelForSequenceClassification.from_pretrained("distilbert-base-uncased", num_labels=3, id2label=dict(enumerate(LABELS)), label2id=LABEL_TO_ID)

    def metrics(prediction):
        logits, labels = prediction
        probability = np.exp(logits - logits.max(axis=1, keepdims=True)); probability /= probability.sum(axis=1, keepdims=True)
        return {"accuracy": accuracy_score(labels, probability.argmax(axis=1)), "roc_auc": roc_auc_score(labels, probability, multi_class="ovr", average="weighted")}

    trainer = Trainer(model=model, args=TrainingArguments(output_dir=str(output / "checkpoints"), learning_rate=2e-5, per_device_train_batch_size=16, per_device_eval_batch_size=32, num_train_epochs=args.epochs, eval_strategy="epoch", save_strategy="no", report_to="none"), train_dataset=train, eval_dataset=test, processing_class=tokenizer, data_collator=DataCollatorWithPadding(tokenizer), compute_metrics=metrics)
    trainer.train()
    results = trainer.evaluate()
    output.mkdir(parents=True, exist_ok=True)
    tokenizer.save_pretrained(output); trainer.save_model(output)
    summary = {"accuracy": results["eval_accuracy"], "roc_auc": results["eval_roc_auc"]}
    (output / "metrics.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    if summary["accuracy"] < 0.894 or summary["roc_auc"] < 0.91:
        raise SystemExit("Quality gate failed: improve data/model before deployment.")


if __name__ == "__main__":
    main()
