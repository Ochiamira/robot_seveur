"""
fallback_handler.py
===================
Gère tous les cas d'échec du pipeline :
- LLM timeout ou indisponible
- Intent non reconnu après plusieurs tours
- Texte vide ou inaudible
- Erreur inattendue
Retourne toujours une réponse propre pour le TTS.
"""

from config import DEFAULT_LANG

# Nombre de tours consécutifs "autre" avant d'activer le fallback fort
MAX_UNCLEAR_TURNS = 2


class FallbackHandler:
    def __init__(self):
        self._unclear_count = 0

    def reset(self):
        self._unclear_count = 0

    def handle(
        self,
        reason:  str,
        lang:    str = DEFAULT_LANG,
        context: dict = None,
    ) -> str:
        """
        Retourne une réponse de secours selon la raison.

        Raisons :
          "empty"       : texte STT vide / silence
          "llm_timeout" : Ollama ne répond pas
          "llm_error"   : erreur LLM générique
          "low_score"   : intent non reconnu (géré par confidence_checker)
          "unclear"     : plusieurs tours incompréhensibles
          "generic"     : erreur inattendue
        """
        _messages = {
            "empty": {
                "fr": "Je n'ai rien entendu. Pouvez-vous répéter ?",
                "ar": "لم أسمع شيئاً. هل يمكنك التكرار؟",
                "en": "I didn't hear anything. Could you repeat?",
            },
            "llm_timeout": {
                "fr": "Je réfléchis encore un instant, veuillez patienter.",
                "ar": "أحتاج لحظة للتفكير، يرجى الانتظار.",
                "en": "I need a moment to think, please wait.",
            },
            "llm_error": {
                "fr": "Je rencontre une difficulté technique. Réessayons. Que souhaitez-vous commander ?",
                "ar": "أواجه صعوبة تقنية. لنحاول مجدداً. ماذا تريد أن تطلب؟",
                "en": "I'm experiencing a technical issue. Let's try again. What would you like to order?",
            },
            "unclear": {
                "fr": "Je ne comprends pas bien. Dites-moi simplement ce que vous voulez commander.",
                "ar": "لا أفهم جيداً. أخبرني ببساطة ماذا تريد أن تطلب.",
                "en": "I'm not understanding well. Just tell me what you'd like to order.",
            },
            "generic": {
                "fr": "Une erreur est survenue. Que souhaitez-vous commander ?",
                "ar": "حدث خطأ. ماذا تريد أن تطلب؟",
                "en": "An error occurred. What would you like to order?",
            },
        }

        if reason == "low_score":
            self._unclear_count += 1
            if self._unclear_count >= MAX_UNCLEAR_TURNS:
                self._unclear_count = 0
                reason = "unclear"
            else:
                reason = "low_score"
                msgs = {
                    "fr": "Je n'ai pas bien saisi. Pouvez-vous reformuler ?",
                    "ar": "لم أفهم جيداً. هل يمكنك إعادة الصياغة؟",
                    "en": "I didn't quite catch that. Could you rephrase?",
                }
                return msgs.get(lang, msgs["fr"])
        else:
            self._unclear_count = 0

        bucket = _messages.get(reason, _messages["generic"])
        return bucket.get(lang, bucket["fr"])


# Singleton
_handler_instance = None

def get_fallback() -> FallbackHandler:
    global _handler_instance
    if _handler_instance is None:
        _handler_instance = FallbackHandler()
    return _handler_instance


if __name__ == "__main__":
    fh = FallbackHandler()
    reasons = ["empty", "llm_timeout", "llm_error", "low_score", "low_score", "generic"]
    print("=== Test fallback_handler ===")
    for r in reasons:
        msg = fh.handle(r, "fr")
        print(f"  [{r:<15}] → {msg}")
