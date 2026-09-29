"""
preprocessing.py
================
Normalisation du texte brut issu du STT (Whisper).
Nettoie les artefacts de transcription avant traitement NLP.
"""

import re
import unicodedata
from config import SUPPORTED_LANGS

# ── Corrections courantes STT par langue ──────────────────────────────────────
_STT_CORRECTIONS = {
    "fr": {
        "je voudrai ": "je voudrais ",
        "sil vous plait": "s'il vous plaît",
        "sil vous plat":  "s'il vous plaît",
        "cest":           "c'est",
        "jai":            "j'ai",
    },
    "ar": {},
    "en": {
        "i'd like": "i would like",
        "i wanna":  "i want",
        "gimme":    "give me",
        "gonna":    "going to",
    },
}

# ── Mots de remplissage (hésitations STT) ─────────────────────────────────────
_FILLERS = {
    "fr": ["euh", "heu", "bah", "ben", "voilà voilà", "donc", "en fait"],
    "ar": ["يعني", "اممم", "اه", "والله"],
    "en": ["uh", "um", "like", "you know", "well", "so"],
}


def normalize(text: str, lang: str = "fr") -> str:
    """
    Pipeline complet de normalisation :
    1. Minuscules
    2. Normalisation unicode
    3. Suppression caractères parasites
    4. Corrections STT courantes
    5. Suppression mots de remplissage
    6. Nettoyage espaces multiples
    """
    if not text or not text.strip():
        return ""

    text = text.lower().strip()
    text = unicodedata.normalize("NFC", text)
    text = re.sub(r"[^\w\s\'\-\,\.\!\?\؟،]", " ", text, flags=re.UNICODE)

    corrections = _STT_CORRECTIONS.get(lang, {})
    for wrong, right in corrections.items():
        text = text.replace(wrong, right)

    fillers = _FILLERS.get(lang, [])
    for filler in fillers:
        # "like" est ambigu : mot de remplissage ("it's like, whatever") MAIS
        # aussi partie intégrante de "would like" / "'d like" (after STT
        # correction ci-dessus), où le supprimer casse la formulation de
        # commande la plus courante en anglais ("i'd like a coffee" devenait
        # "i would a coffee" -> intent "autre" au lieu de "commander").
        # On ne le traite comme filler que s'il n'est PAS précédé de
        # "would "/"d " (issu de "i'd").
        if filler == "like" and lang == "en":
            text = re.sub(
                r"(?<!would )(?<!'d )(?<!d )\blike\b",
                " ", text, flags=re.IGNORECASE | re.UNICODE
            )
            continue
        text = re.sub(
            rf"(?<!\w){re.escape(filler)}(?!\w)",
            " ", text, flags=re.IGNORECASE | re.UNICODE
        )

    text = re.sub(r"[\.]{2,}", ".", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def remove_punctuation(text: str) -> str:
    return re.sub(r"[^\w\s]", " ", text, flags=re.UNICODE).strip()


if __name__ == "__main__":
    tests = [
        ("euh je voudrai commander un couscous sil vous plait", "fr"),
        ("I wanna get uh like a coffee", "en"),
        ("يعني انا نحب نطلب كسكسي", "ar"),
    ]
    print("=== Test preprocessing ===")
    for text, lang in tests:
        result = normalize(text, lang)
        print(f"  [{lang}] IN  : {text}")
        print(f"         OUT : {result}\n")
