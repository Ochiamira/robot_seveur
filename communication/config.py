"""
config.py
=========
Configuration centrale du pipeline NLP.
Adapte les URLs selon ton environnement (PC dev vs Raspberry Pi 5).
"""

import os
from pathlib import Path

# ── Chemins ───────────────────────────────────────────────────────────────────
BASE_DIR = Path(__file__).resolve().parent
PROJECT_DIR = BASE_DIR.parent
# Le menu.json canonique vit désormais UNIQUEMENT dans l'app staff/dashboard
# (application/ws/menu.json, avec ses images) — plus de copie dupliquée ici.
# Chemin par défaut basé sur la structure du projet :
#   pfe_project/communication/<ce dossier>/
#   pfe_project/application/ws/menu.json
# Surchargeable via variable d'env MENU_PATH si ta structure de dossiers
# diffère (ex: déploiement Raspberry Pi avec un autre agencement).
MENU_PATH = Path(os.getenv("MENU_PATH", str(PROJECT_DIR / "application" / "ws" / "menu.json"))).resolve()

# Modèle STT et réglages audio. Tous les chemins restent surchargeables pour
# permettre le déploiement Windows/Linux sans modifier le code.
WHISPER_MODEL_PATH = Path(os.getenv(
    "WHISPER_MODEL_PATH",
    str(BASE_DIR / "whisper-small-nexor" / "final"),
)).resolve()
SAMPLE_RATE = int(os.getenv("SAMPLE_RATE", "16000"))
SILENCE_THRESHOLD = float(os.getenv("SILENCE_THRESHOLD", "0.01"))
SILENCE_DURATION_S = float(os.getenv("SILENCE_DURATION_S", "0.6"))
MAX_RECORD_S = float(os.getenv("MAX_RECORD_S", "10"))
MIN_SPEECH_S = float(os.getenv("MIN_SPEECH_S", "0.4"))
SPEECH_START_TIMEOUT_S = float(os.getenv("SPEECH_START_TIMEOUT_S", "4"))

# Périphérique de capture réel. Sous Windows, le moteur choisit
# automatiquement le Microphone Array via WASAPI quand aucun nom / index
# explicite n'est fourni. La fréquence 0 signifie « fréquence native du
# périphérique » (48 kHz sur le PC de développement), puis l'audio est
# rééchantillonné vers SAMPLE_RATE pour Whisper.
AUDIO_INPUT_DEVICE = os.getenv("AUDIO_INPUT_DEVICE", "").strip()
AUDIO_INPUT_HOST_API = os.getenv(
    "AUDIO_INPUT_HOST_API", "Windows WASAPI" if os.name == "nt" else ""
).strip()
AUDIO_INPUT_SAMPLE_RATE = int(os.getenv("AUDIO_INPUT_SAMPLE_RATE", "0"))

# ── Ollama ────────────────────────────────────────────────────────────────────
# PC Windows dev   : http://localhost:11434
# Raspberry Pi 5   : http://localhost:11434  (ollama tourne localement)
# Pi → serveur PC  : http://<IP_PC>:11434
OLLAMA_URL   = os.getenv("OLLAMA_URL",   "http://localhost:11434")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "mistral")   # ou "llama3.2", "phi3"

# Timeout en secondes (Raspberry Pi est plus lent)
OLLAMA_TIMEOUT = int(os.getenv("OLLAMA_TIMEOUT", "30"))
OLLAMA_WARMUP_TIMEOUT = int(os.getenv("OLLAMA_WARMUP_TIMEOUT", "120"))
OLLAMA_KEEP_ALIVE = os.getenv("OLLAMA_KEEP_ALIVE", "30m")

# ── Langues supportées ────────────────────────────────────────────────────────
SUPPORTED_LANGS  = ["fr", "ar", "en"]
DEFAULT_LANG     = "fr"

# ── Intents supportés ─────────────────────────────────────────────────────────
INTENTS = [
    "commander",          # "je veux une pizza"
    "ajouter",            # "ajoute une salade"
    "supprimer",          # "enlève le café"
    "modifier",           # "change le jus en grande taille"
    "confirmer_commande", # "c'est tout", "voilà", "confirme"
    "annuler_commande",   # "annule tout"
    "demander_total",     # "combien ça coûte ?"
    "demander_menu",      # "qu'est-ce que vous avez ?"
    "paiement",           # "je veux payer en espèces / par carte"
    "salutation",         # "bonjour"
    "au_revoir",          # "merci au revoir"
    "autre",              # hors scope
]

# ── Dialog ────────────────────────────────────────────────────────────────────
MAX_TURNS       = 10    # nombre max de tours de dialogue
MAX_ITEMS_ORDER = 10    # nombre max d'items dans une commande

# ── Logging ───────────────────────────────────────────────────────────────────
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
STAFF_APP_URL = os.getenv("STAFF_APP_URL", "http://localhost:8000").rstrip("/")
STAFF_APP_TIMEOUT = float(os.getenv("STAFF_APP_TIMEOUT", "1.5"))
STAFF_APP_RETRIES = max(0, int(os.getenv("STAFF_APP_RETRIES", "2")))
STAFF_APP_TOKEN = os.getenv("STAFF_APP_TOKEN", "").strip()
STAFF_QUEUE_MAX = max(8, int(os.getenv("STAFF_QUEUE_MAX", "128")))
STAFF_AUDIT_MAX_BYTES = max(65536, int(os.getenv("STAFF_AUDIT_MAX_BYTES", "5242880")))
STAFF_OUTBOX_MAX_BYTES = max(65536, int(os.getenv("STAFF_OUTBOX_MAX_BYTES", "5242880")))
PIPER_TIMEOUT_S = float(os.getenv("PIPER_TIMEOUT_S", "30"))
TTS_PLAYBACK_TIMEOUT_S = float(os.getenv("TTS_PLAYBACK_TIMEOUT_S", "60"))

# Les transcriptions et commandes sont des données potentiellement sensibles.
# Elles ne sont journalisées en clair que sur demande explicite.
LOG_SENSITIVE_DATA = os.getenv("LOG_SENSITIVE_DATA", "0").lower() in {"1", "true", "yes"}
SESSION_LOG_DIR = Path(os.getenv("SESSION_LOG_DIR", str(BASE_DIR / "logs"))).resolve()
