"""
confidence_checker.py
=====================
Vérifie la confiance de la classification d'intent.
Si le score est trop bas → demande une clarification au lieu de passer au LLM.
Évite les réponses absurdes quand le client dit quelque chose d'ambigu.
"""

from config import DEFAULT_LANG

# Seuil minimum pour considérer un intent comme fiable
CONFIDENCE_THRESHOLD = 0.25

# Intents qui n'ont jamais besoin de clarification (toujours clairs)
_ALWAYS_CLEAR = {
    "salutation", "au_revoir", "confirmer_commande", "annuler_commande",
    "paiement",  # toujours détecté par mot-clé, jamais ambigu
}


def is_confident(intent: str, score: float) -> bool:
    """Retourne True si le score est suffisant pour agir."""
    if intent in _ALWAYS_CLEAR:
        return True
    return score >= CONFIDENCE_THRESHOLD


def clarification_message(lang: str = DEFAULT_LANG) -> str:
    """Message de clarification multilingue."""
    msgs = {
        "fr": "Je n'ai pas bien compris votre demande. Pouvez-vous reformuler ? Par exemple : 'je voudrais un café' ou 'c'est tout'.",
        "ar": "لم أفهم طلبك جيداً. هل يمكنك إعادة الصياغة؟ مثلاً: 'أريد قهوة' أو 'هذا كل شيء'.",
        "en": "I didn't quite understand. Could you rephrase? For example: 'I'd like a coffee' or 'that's all'.",
    }
    return msgs.get(lang, msgs["fr"])


def check(intent: str, score: float, lang: str = DEFAULT_LANG) -> tuple[bool, str]:
    """
    Vérifie la confiance et retourne (ok, message_clarification).

    Returns:
        (True, "")           → intent fiable, on continue
        (False, "message")  → intent ambigu, demander clarification
    """
    if is_confident(intent, score):
        return True, ""
    return False, clarification_message(lang)


if __name__ == "__main__":
    tests = [
        ("commander",  0.67, "fr"),
        ("autre",      0.0,  "fr"),
        ("commander",  0.10, "fr"),  # score trop bas
        ("salutation", 0.0,  "ar"),  # toujours clair
    ]
    print("=== Test confidence_checker ===")
    for intent, score, lang in tests:
        ok, msg = check(intent, score, lang)
        print(f"  intent={intent:<20} score={score:.2f}  → {'✅ OK' if ok else f'❌ {msg[:50]}'}")