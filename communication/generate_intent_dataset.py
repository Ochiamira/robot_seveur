"""
generate_intent_dataset.py (v2 — aligné sur config.INTENTS)

Génère un dataset synthétique via Ollama/Mistral, avec les intentions
EXACTES de ton config.py (lowercase_avec_underscores), pour que le modèle
DistilBERT retourne des labels compatibles avec intent_classifier.py.
"""

import argparse
import csv
import json
import random
import re
import time
from pathlib import Path

import requests
from tqdm import tqdm

OLLAMA_URL = "http://localhost:11434/api/generate"
MODEL_NAME = "mistral"

# Doit matcher EXACTEMENT config.INTENTS
INTENTS = [
    "commander",
    "ajouter",
    "supprimer",
    "modifier",
    "confirmer_commande",
    "annuler_commande",
    "demander_total",
    "demander_menu",
    "salutation",
    "au_revoir",
    "autre",
]

LANGUES = ["fr", "ar", "en"]

INTENT_DESCRIPTIONS = {
    "commander": {
        "fr": "Le client passe une première commande (ex: 'je voudrais un couscous')",
        "ar": "الزبون يطلب طلبية جديدة (مثال: نحب كسكسي)",
        "en": "The customer places a first order (e.g. 'I'd like a couscous')",
    },
    "ajouter": {
        "fr": "Le client ajoute un item à une commande déjà en cours (ex: 'ajoute aussi un café')",
        "ar": "الزبون يزيد صنف على طلبية موجودة (مثال: زيد قهوة)",
        "en": "The customer adds an item to an existing order (e.g. 'also add a coffee')",
    },
    "supprimer": {
        "fr": "Le client retire un item de sa commande (ex: 'enlève le café')",
        "ar": "الزبون يشيل صنف من طلبيته (مثال: شيل القهوة)",
        "en": "The customer removes an item from their order (e.g. 'remove the coffee')",
    },
    "modifier": {
        "fr": "Le client remplace un item par un autre (ex: 'change le jus en grande taille')",
        "ar": "الزبون يبدل شيء (مثال: غير العصير لحجم كبير)",
        "en": "The customer swaps or changes an item (e.g. 'change the juice to large')",
    },
    "confirmer_commande": {
        "fr": "Le client confirme que sa commande est terminée (ex: 'c'est tout merci', 'voilà')",
        "ar": "الزبون يأكد نهاية الطلبية (مثال: هذا كل شيء)",
        "en": "The customer confirms the order is complete (e.g. 'that's all thanks')",
    },
    "annuler_commande": {
        "fr": "Le client annule toute la commande (ex: 'annule tout', 'recommençons')",
        "ar": "الزبون يلغي الطلبية كاملة (مثال: ألغِ كل شيء)",
        "en": "The customer cancels the whole order (e.g. 'cancel everything')",
    },
    "demander_total": {
        "fr": "Le client demande le prix total (ex: 'combien ça fait ?', 'l'addition')",
        "ar": "الزبون يسأل على المجموع (مثال: قداش الحساب)",
        "en": "The customer asks for the total price (e.g. 'how much is that')",
    },
    "demander_menu": {
        "fr": "Le client demande ce qui est disponible (ex: 'qu'est-ce que vous avez ?')",
        "ar": "الزبون يسأل شنوة عندكم (مثال: شنوة فما في المنيو)",
        "en": "The customer asks what's available (e.g. 'what do you have')",
    },
    "salutation": {
        "fr": "Le client salue en arrivant (ex: 'bonjour')",
        "ar": "الزبون يسلم عند وصوله (مثال: السلام عليكم)",
        "en": "The customer greets on arrival (e.g. 'hello')",
    },
    "au_revoir": {
        "fr": "Le client prend congé (ex: 'merci au revoir', 'bonne journée')",
        "ar": "الزبون يودع (مثال: مع السلامة)",
        "en": "The customer says goodbye (e.g. 'thanks, bye')",
    },
    "autre": {
        "fr": "Message hors contexte restaurant ou incompréhensible",
        "ar": "رسالة مالهاش علاقة بالمطعم",
        "en": "Off-topic or unclear message",
    },
}

GENERATION_PROMPT = """Tu génères des données d'entraînement pour un classifieur NLP.

Contexte: {description}
Langue de sortie: {langue_nom}

Génère exactement {n} phrases DIFFÉRENTES et NATURELLES qu'un client pourrait dire
dans un restaurant tunisien, correspondant à ce contexte. Varie le style
(formel/familier), la longueur, les tournures. Inclus des variations réalistes
(hésitations, fautes légères de transcription vocale possibles).

Retourne UNIQUEMENT un tableau JSON de strings, sans aucun texte autour, sans markdown.
Exemple de format: ["phrase 1", "phrase 2", "phrase 3"]
"""

LANGUE_NOMS = {"fr": "français", "ar": "arabe tunisien (darja)", "en": "anglais"}


def call_ollama(prompt: str, retries: int = 3) -> str:
    for attempt in range(retries):
        try:
            resp = requests.post(
                OLLAMA_URL,
                json={
                    "model": MODEL_NAME,
                    "prompt": prompt,
                    "stream": False,
                    "options": {"temperature": 0.9},
                },
                timeout=120,
            )
            resp.raise_for_status()
            return resp.json()["response"]
        except Exception as e:
            print(f"  [retry {attempt + 1}/{retries}] erreur Ollama: {e}")
            time.sleep(2)
    raise RuntimeError("Échec appel Ollama après plusieurs tentatives")


def extract_json_array(raw_text: str) -> list:
    raw_text = raw_text.strip()
    raw_text = re.sub(r"^```json\s*|\s*```$", "", raw_text, flags=re.MULTILINE)
    match = re.search(r"\[.*\]", raw_text, flags=re.DOTALL)
    if not match:
        raise ValueError(f"Pas de tableau JSON trouvé dans: {raw_text[:200]}")
    return json.loads(match.group(0))


def generate_for_combo(intent: str, langue: str, n: int) -> list:
    description = INTENT_DESCRIPTIONS[intent][langue]
    prompt = GENERATION_PROMPT.format(
        description=description, langue_nom=LANGUE_NOMS[langue], n=n
    )
    raw = call_ollama(prompt)
    try:
        phrases = extract_json_array(raw)
    except Exception as e:
        print(f"  [warn] parsing échoué pour {intent}/{langue}: {e}")
        return []
    return [p for p in phrases if isinstance(p, str) and len(p.strip()) > 0]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="dataset_intentions.csv")
    parser.add_argument("--n_per_combo", type=int, default=40)
    parser.add_argument("--batch_size", type=int, default=15)
    args = parser.parse_args()

    rows = []
    combos = [(i, l) for i in INTENTS for l in LANGUES]

    for intent, langue in tqdm(combos, desc="Génération"):
        collected = []
        while len(collected) < args.n_per_combo:
            remaining = args.n_per_combo - len(collected)
            batch_n = min(args.batch_size, remaining)
            phrases = generate_for_combo(intent, langue, batch_n)
            collected.extend(phrases)
            if not phrases:
                break

        for phrase in collected[: args.n_per_combo]:
            rows.append({"text": phrase.strip(), "intent": intent, "langue": langue})

    seen = set()
    unique_rows = []
    for r in rows:
        key = (r["text"].lower(), r["intent"])
        if key not in seen:
            seen.add(key)
            unique_rows.append(r)

    random.shuffle(unique_rows)

    output_path = Path(args.output)
    with output_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["text", "intent", "langue"])
        writer.writeheader()
        writer.writerows(unique_rows)

    print(f"\n✅ Dataset généré: {output_path} ({len(unique_rows)} exemples uniques)")
    print("\nRépartition par intention:")
    for intent in INTENTS:
        count = sum(1 for r in unique_rows if r["intent"] == intent)
        print(f"  {intent}: {count}")


if __name__ == "__main__":
    main()
