"""
response_formatter.py
=====================
Nettoie et formate la réponse du LLM pour la synthèse vocale (TTS).
Supprime tout ce qui n'est pas naturel à l'oral.
"""

import re


def format_response(text: str, lang: str = "fr") -> str:
    """
    Pipeline de nettoyage pour TTS :
    1. Supprime markdown (**, *, #, `, ---)
    2. Supprime listes à puces et numérotées
    3. Supprime URLs et balises HTML
    4. Normalise la ponctuation pour le TTS
    5. Tronque si trop long (TTS doit rester < 30 mots)
    6. Supprime espaces multiples
    """
    if not text or not text.strip():
        return _fallback(lang)

    # 1. Supprime markdown
    text = re.sub(r"\*{1,3}(.*?)\*{1,3}", r"\1", text)   # bold/italic
    text = re.sub(r"#{1,6}\s*",           "",    text)     # headers
    text = re.sub(r"`{1,3}.*?`{1,3}",     "",    text, flags=re.DOTALL)  # code
    text = re.sub(r"---+",                "",    text)     # hr

    # 2. Supprime listes (- item, * item, 1. item)
    text = re.sub(r"^\s*[-\*•]\s+", "", text, flags=re.MULTILINE)
    text = re.sub(r"^\s*\d+\.\s+",  "", text, flags=re.MULTILINE)

    # 3. Supprime HTML et URLs
    text = re.sub(r"<[^>]+>",           "", text)
    text = re.sub(r"https?://\S+",      "", text)

    # 4. Remplace sauts de ligne par des espaces
    text = re.sub(r"\n+", " ", text)

    # 5. Normalise ponctuation pour TTS
    text = re.sub(r"\s*:\s*",  " : ", text)
    text = re.sub(r"\s*,\s*",  ", ",  text)
    text = re.sub(r"\s*\.\s*(?=[^\s\d]|\s)", ". ", text)    
    text = re.sub(r"!+",       "!",   text)
    text = re.sub(r"\?+",      "?",   text)

    # 6. Supprime emojis
    text = re.sub(
        r"[\U0001F600-\U0001F64F\U0001F300-\U0001F5FF"
        r"\U0001F680-\U0001F6FF\U0001F1E0-\U0001F1FF]+",
        "", text, flags=re.UNICODE
    )

    # 7. Nettoyage espaces
    text = re.sub(r"\s+", " ", text).strip()

    # 8. Tronque si > 50 mots (trop long pour TTS)
    words = text.split()
    if len(words) > 50:
        text = " ".join(words[:50])
        # Coupe à la dernière phrase complète
        for punct in [".", "!", "?"]:
            idx = text.rfind(punct)
            if idx > 0:
                text = text[:idx + 1]
                break

    if not text.strip():
        return _fallback(lang)

    return text.strip()


def _fallback(lang: str) -> str:
    msgs = {
        "fr": "Je n'ai pas compris. Pouvez-vous répéter ?",
        "ar": "لم أفهم. هل يمكنك التكرار؟",
        "en": "I didn't understand. Could you repeat please?",
    }
    return msgs.get(lang, msgs["fr"])


if __name__ == "__main__":
    tests = [
        "**Bien sûr !** Voici votre commande :\n- 1× Couscous (18.000 TND)\n- 1× Café (1.500 TND)\n\nTotal : **19.500 TND** 😊",
        "# Confirmation\nVotre commande est confirmée. Merci !",
        "J'ai ajouté 1× Fondant au chocolat (7.000 TND) à votre commande. Autre chose ?",
    ]
    print("=== Test response_formatter ===")
    for t in tests:
        print(f"\n  IN  : {t[:80]}")
        print(f"  OUT : {format_response(t, 'fr')}")
