"""
train_intent_classifier.py

INCHANGÉ par rapport à la version précédente — ce script est déjà générique
(il lit les labels dynamiquement depuis le CSV), donc il fonctionne tel quel
avec le nouveau dataset_intentions.csv généré par generate_intent_dataset.py.

Fine-tune un DistilBERT multilingue léger pour la classification d'intention
NEXOR (FR/AR/EN). Conçu pour tourner sur Kaggle (GPU T4) mais fonctionne aussi
en local (CPU/GPU) avec un dataset de cette taille.

Prérequis:
    pip install transformers datasets scikit-learn torch accelerate evaluate

Usage (Kaggle ou local):
    python train_intent_classifier.py \
        --dataset dataset_intentions.csv \
        --output_dir ./intent_model \
        --epochs 8 \
        --batch_size 16

Note VRAM: distilbert-base-multilingual-cased fait ~135M paramètres.
Avec batch_size=16 et max_length=64, ça tient largement sur un RTX 2050 4GB
en local, mais Kaggle (T4 16GB) permet des batchs plus gros et un entraînement
plus rapide.
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from datasets import Dataset
from sklearn.metrics import accuracy_score, f1_score, precision_recall_fscore_support
from sklearn.model_selection import train_test_split
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    Trainer,
    TrainingArguments,
)

MODEL_CHECKPOINT = "distilbert-base-multilingual-cased"
MAX_LENGTH = 64


def load_and_split(csv_path: str, test_size: float = 0.15):
    df = pd.read_csv(csv_path)
    df = df.dropna(subset=["text", "intent"])
    df["text"] = df["text"].astype(str).str.strip()
    df = df[df["text"].str.len() > 0]

    labels = sorted(df["intent"].unique())
    label2id = {label: i for i, label in enumerate(labels)}
    id2label = {i: label for label, i in label2id.items()}
    df["label"] = df["intent"].map(label2id)

    train_df, eval_df = train_test_split(
        df, test_size=test_size, stratify=df["label"], random_state=42
    )
    return train_df, eval_df, label2id, id2label


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    acc = accuracy_score(labels, preds)
    precision, recall, f1, _ = precision_recall_fscore_support(
        labels, preds, average="weighted", zero_division=0
    )
    return {"accuracy": acc, "f1": f1, "precision": precision, "recall": recall}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="dataset_intentions.csv")
    parser.add_argument("--output_dir", default="./intent_model")
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=2e-5)
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device utilisé: {device}")

    train_df, eval_df, label2id, id2label = load_and_split(args.dataset)
    print(f"Train: {len(train_df)} exemples | Eval: {len(eval_df)} exemples")
    print(f"Classes ({len(label2id)}): {list(label2id.keys())}")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_CHECKPOINT)

    def tokenize_fn(batch):
        return tokenizer(
            batch["text"], truncation=True, max_length=MAX_LENGTH, padding=False
        )

    train_ds = Dataset.from_pandas(train_df[["text", "label"]].reset_index(drop=True))
    eval_ds = Dataset.from_pandas(eval_df[["text", "label"]].reset_index(drop=True))

    train_ds = train_ds.map(tokenize_fn, batched=True)
    eval_ds = eval_ds.map(tokenize_fn, batched=True)

    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_CHECKPOINT,
        num_labels=len(label2id),
        id2label=id2label,
        label2id=label2id,
    )

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        eval_strategy="epoch",
        save_strategy="epoch",
        learning_rate=args.lr,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        num_train_epochs=args.epochs,
        weight_decay=0.01,
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        logging_steps=10,
        save_total_limit=2,
        fp16=torch.cuda.is_available(),
        report_to="none",
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )

    trainer.train()

    print("\n=== Évaluation finale ===")
    metrics = trainer.evaluate()
    for k, v in metrics.items():
        print(f"  {k}: {v:.4f}" if isinstance(v, float) else f"  {k}: {v}")

    final_dir = Path(args.output_dir) / "final"
    final_dir.mkdir(parents=True, exist_ok=True)
    trainer.save_model(str(final_dir))
    tokenizer.save_pretrained(str(final_dir))

    with (final_dir / "label_mapping.json").open("w", encoding="utf-8") as f:
        json.dump({"label2id": label2id, "id2label": id2label}, f, ensure_ascii=False, indent=2)

    print(f"\n✅ Modèle sauvegardé dans: {final_dir}")
    print("Place ce dossier 'final/' dans <ton_projet>/intent_model/final/ (chemin attendu par intent_classifier.py)")


if __name__ == "__main__":
    main()
