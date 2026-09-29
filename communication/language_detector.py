"""
language_detector.py
====================
Détecte la langue du texte transcrit par Whisper.
Whisper fournit déjà la langue — ce module sert de fallback
et de validation quand la confiance Whisper est faible.
"""

import re
from config import SUPPORTED_LANGS, DEFAULT_LANG

# Plage Unicode de l'écriture arabe (couvre aussi la darja tunisienne écrite en
# caractères arabes, même quand le vocabulaire spécifique n'est pas dans la
# liste de mots-clés ci-dessous).
_ARABIC_SCRIPT_RE = re.compile(r"[\u0600-\u06FF\u0750-\u077F]")

# Mots-clés caractéristiques par langue (rapide, sans dépendance externe)
_KEYWORDS = {
    "ar": [
        "أنا", "نحب", "نبغي", "أريد", "من فضلك", "شكرا", "واحد", "اثنين",
        "كسكسي", "قهوة", "عصير", "ماء", "مرحبا", "السلام", "بغيت", "نطلب"
    ],
    "fr": [
        "je", "veux", "voudrais", "bonjour", "merci", "s'il", "plaît",
        "commande", "café", "couscous", "une", "un", "le", "la", "les",
        "avec", "sans", "pour", "bien", "aussi"
    ],
    "en": [
        "i", "want", "would", "like", "please", "hello", "thank", "coffee",
        "order", "menu", "the", "a", "an", "with", "without", "and", "can"
    ],
}

# Un seul mot très caractéristique suffit pour corriger une langue Whisper
# erronée sur les phrases courtes (cas fréquent avec « bonjour »/« hello »).
_DISTINCTIVE = {
    "fr": {"bonjour", "voudrais", "merci"},
    "en": {"hello", "please", "without", "thanks", "goodbye"},
}
_DISTINCTIVE_PHRASES = {
    "fr": (r"\bc['’]?est tout\b", r"\bau revoir\b", r"\bcombien (?:ça|ca)\b"),
    "en": (r"\bthat['’]?s all\b", r"\bhow much\b", r"\bthank you\b"),
}


def detect(text: str, whisper_lang: str = None) -> str:
    """
    Détecte la langue du texte.

    Args:
        text         : texte transcrit (déjà normalisé)
        whisper_lang : langue détectée par Whisper (ar/en/fr) — prioritaire

    Returns:
        code langue : "ar", "fr" ou "en"
    """
    if not text or not text.strip():
        return whisper_lang if whisper_lang in SUPPORTED_LANGS else DEFAULT_LANG

    # 2. Signal fort et bon marché : présence de caractères arabes.
    # Couvre la darja tunisienne même quand le mot précis n'est pas dans
    # _KEYWORDS["ar"] — sans ça, un texte arabe non reconnu par mots-clés
    # retombait silencieusement sur DEFAULT_LANG ("fr"), ce qui envoyait
    # ensuite tout le texte arabe dans les regex FRANÇAISES de l'intent
    # classifier (échec total de classification pour ces cas).
    if _ARABIC_SCRIPT_RE.search(text):
        return "ar"

    # 3. Validation lexicale : un modèle finement entraîné peut perdre une
    # partie de sa précision de détection de langue. Deux indices textuels
    # cohérents priment donc sur un token Whisper contradictoire.
    text_lower = text.lower()
    for lang, patterns in _DISTINCTIVE_PHRASES.items():
        if any(re.search(pattern, text_lower, flags=re.UNICODE) for pattern in patterns):
            return lang
    scores = {lang: 0 for lang in SUPPORTED_LANGS}

    for lang, keywords in _KEYWORDS.items():
        for kw in keywords:
            if re.search(rf"\b{re.escape(kw)}\b", text_lower, flags=re.UNICODE):
                scores[lang] += 1

    best_lang = max(scores, key=scores.get)

    if scores[best_lang] >= 2:
        return best_lang

    words = set(re.findall(r"\b[^\W\d_]+\b", text_lower, flags=re.UNICODE))
    for lang, distinctive_words in _DISTINCTIVE.items():
        if words & distinctive_words:
            return lang

    if whisper_lang and whisper_lang in SUPPORTED_LANGS:
        return whisper_lang

    if scores[best_lang] == 0:
        return DEFAULT_LANG

    return best_lang


if __name__ == "__main__":
    tests = [
        ("je voudrais un café s'il vous plaît", None),
        ("i want a coffee please", None),
        ("نحب نطلب كسكسي من فضلك", None),
        ("bonjour", "fr"),
        ("hello", "en"),
    ]
    print("=== Test language_detector ===")
    for text, whisper_lang in tests:
        lang = detect(text, whisper_lang)
        print(f"  '{text[:40]}' → {lang}")
