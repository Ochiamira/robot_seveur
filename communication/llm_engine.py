"""
llm_engine.py
=============
Interface avec Ollama (API REST locale).
Gère PC dev et Raspberry Pi 5 de façon transparente.
"""

import requests
import json
import logging
from config import (
    OLLAMA_KEEP_ALIVE,
    OLLAMA_MODEL,
    OLLAMA_TIMEOUT,
    OLLAMA_URL,
    OLLAMA_WARMUP_TIMEOUT,
)

logger = logging.getLogger(__name__)


class LLMEngine:
    def __init__(
        self,
        url:     str = OLLAMA_URL,
        model:   str = OLLAMA_MODEL,
        timeout: int = OLLAMA_TIMEOUT,
    ):
        self.url     = url.rstrip("/")
        self.model   = model
        self.timeout = timeout
        self._check_connection()

    def _check_connection(self) -> None:
        """Vérifie qu'Ollama tourne et que le modèle est disponible."""
        try:
            r = requests.get(f"{self.url}/api/tags", timeout=5)
            r.raise_for_status()
            models = [m["name"] for m in r.json().get("models", [])]
            if self.model not in models and f"{self.model}:latest" not in models:
                logger.warning(
                    f"⚠️  Modèle '{self.model}' non trouvé dans Ollama.\n"
                    f"   Disponibles : {models}\n"
                    f"   Lance : ollama pull {self.model}"
                )
            else:
                logger.info(f"✅ Ollama connecté | modèle : {self.model}")
        except requests.exceptions.ConnectionError:
            logger.error(
                f"❌ Ollama introuvable à {self.url}\n"
                f"   Lance : ollama serve"
            )
        except Exception as e:
            logger.warning(f"⚠️  Vérification Ollama : {e}")

    def generate(
        self,
        messages:     list[dict],
        temperature:  float = 0.3,   # bas = réponses cohérentes et concises
        max_tokens:   int   = 150,   # court pour le TTS
        lang:          str   = "fr",
    ) -> str:
        """
        Envoie les messages au LLM et retourne la réponse.

        Args:
            messages    : liste de dicts {role, content}
            temperature : 0.0 = déterministe, 1.0 = créatif
            max_tokens  : longueur max de la réponse

        Returns:
            Texte de la réponse du LLM, ou message d'erreur.
        """
        payload = {
            "model":    self.model,
            "messages": messages,
            "stream":   False,
            "keep_alive": OLLAMA_KEEP_ALIVE,
            "options": {
                "temperature":   temperature,
                "num_predict":   max_tokens,
                "top_p":         0.9,
                "repeat_penalty": 1.1,
            },
        }

        try:
            response = requests.post(
                f"{self.url}/api/chat",
                json=payload,
                timeout=self.timeout,
            )
            response.raise_for_status()
            data    = response.json()
            content = data.get("message", {}).get("content", "").strip()

            if not content:
                return self._fallback_response("empty", lang)

            return content

        except requests.exceptions.Timeout:
            logger.error(f"❌ Timeout LLM ({self.timeout}s)")
            return self._fallback_response("timeout", lang)

        except requests.exceptions.ConnectionError:
            logger.error("❌ Connexion Ollama perdue")
            return self._fallback_response("connection", lang)

        except Exception as e:
            logger.error(f"❌ Erreur LLM : {e}")
            return self._fallback_response("generic", lang)

    def _fallback_response(self, error_type: str, lang: str = "fr") -> str:
        """Réponse de secours en cas d'erreur LLM."""
        fallbacks = {
            "timeout": {
                "fr": "Je suis désolé, je n'ai pas pu traiter votre demande. Pouvez-vous répéter ?",
                "ar": "عذراً، لم أتمكن من معالجة طلبك. هل يمكنك التكرار؟",
                "en": "Sorry, I couldn't process your request. Could you repeat?",
            },
            "connection": {
                "fr": "Le service est temporairement indisponible. Veuillez patienter.",
                "ar": "الخدمة غير متاحة مؤقتاً. يرجى الانتظار.",
                "en": "The service is temporarily unavailable. Please wait.",
            },
            "empty": {
                "fr": "Je n'ai pas compris. Pouvez-vous reformuler ?",
                "ar": "لم أفهم. هل يمكنك إعادة الصياغة؟",
                "en": "I didn't understand. Could you rephrase?",
            },
            "generic": {
                "fr": "Une erreur est survenue. Pouvez-vous répéter ?",
                "ar": "حدث خطأ. هل يمكنك التكرار؟",
                "en": "An error occurred. Could you repeat?",
            },
        }
        bucket = fallbacks.get(error_type, fallbacks["generic"])
        return bucket.get(lang, bucket["fr"])

    def warmup(self) -> bool:
        """Charge le modèle Ollama en RAM sans générer de réponse client."""
        try:
            response = requests.post(
                f"{self.url}/api/generate",
                json={
                    "model": self.model,
                    "prompt": "",
                    "stream": False,
                    "keep_alive": OLLAMA_KEEP_ALIVE,
                    "options": {"num_predict": 0},
                },
                timeout=OLLAMA_WARMUP_TIMEOUT,
            )
            response.raise_for_status()
            logger.info(
                "✅ Ollama préchargé | modèle=%s | maintien=%s",
                self.model,
                OLLAMA_KEEP_ALIVE,
            )
            return True
        except Exception as exc:
            # Ollama est optionnel pour les intents déterministes. Son échec
            # ne doit pas empêcher Whisper et la prise de commande de démarrer.
            logger.warning("⚠️ Préchargement Ollama impossible: %s", exc)
            return False

    def is_available(self) -> bool:
        """Vérifie si Ollama est accessible."""
        try:
            r = requests.get(f"{self.url}/api/tags", timeout=3)
            return r.status_code == 200
        except Exception:
            return False


# Singleton
_engine_instance = None

def get_engine() -> LLMEngine:
    global _engine_instance
    if _engine_instance is None:
        _engine_instance = LLMEngine()
    return _engine_instance


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    engine = get_engine()
    if engine.is_available():
        messages = [
            {"role": "system",  "content": "Tu es un assistant de restaurant. Réponds en 1 phrase."},
            {"role": "user",    "content": "Bonjour, je voudrais un café."},
        ]
        response = engine.generate(messages)
        print(f"\nRéponse LLM : {response}")
    else:
        print("❌ Ollama non disponible — lance : ollama serve")
