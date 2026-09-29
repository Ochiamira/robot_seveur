"""
intent_classifier_distilbert.py

Wrapper minimal autour du modèle DistilBERT fine-tuné pour la classification
d'intention. Ne contient AUCUNE logique métier (pas de fallback LLM, pas de
regex) — c'est intent_classifier.py qui orchestre tout, ce fichier ne fait
que charger le modèle et prédire.

Import tardif volontaire (torch/transformers sont lourds à charger) :
intent_classifier.py ne l'importe que si le regex échoue et que le dossier
du modèle existe.
"""

import json
from pathlib import Path

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer


class IntentClassifier:
    """Chargé une seule fois (singleton géré par intent_classifier.py)."""

    def __init__(self, model_dir: str, device: str | None = None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        model_dir_path = Path(model_dir)

        self.tokenizer = AutoTokenizer.from_pretrained(model_dir_path)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_dir_path)
        self.model.to(self.device)
        self.model.eval()

        with (model_dir_path / "label_mapping.json").open("r", encoding="utf-8") as f:
            mapping = json.load(f)
        self.id2label = {int(k): v for k, v in mapping["id2label"].items()}

        print(f"[intent_classifier_distilbert] modèle chargé sur {self.device}")

    @torch.no_grad()
    def predict(self, text: str) -> tuple[str, float]:
        """Retourne (intent, confiance) — confiance = score softmax de la classe prédite."""
        inputs = self.tokenizer(
            text, truncation=True, max_length=64, padding=True, return_tensors="pt"
        ).to(self.device)

        logits = self.model(**inputs).logits
        probs = torch.softmax(logits, dim=-1)[0]
        pred_id = int(torch.argmax(probs).item())
        confidence = float(probs[pred_id].item())

        return self.id2label[pred_id], confidence


if __name__ == "__main__":
    import sys

    model_dir = sys.argv[1] if len(sys.argv) > 1 else "./intent_model/final"
    classifier = IntentClassifier(model_dir)

    tests = [
        "finalement mets-en moi un de plus",
        "je changerais bien d'avis en fait",
        "c parfait comme ça niquel",
    ]
    for t in tests:
        intent, conf = classifier.predict(t)
        print(f"'{t}' -> {intent} (confiance={conf:.3f})")
