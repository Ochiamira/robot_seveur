"""
test_full_pipeline.py
======================
Pipeline complet : fichier(s) .wav -> Whisper (STT) -> NLP (langue, intent,
entités) -> dialog_manager (réponse) -> résultats sauvegardés en JSON.

Tout se configure ci-dessous, section CONFIG — pas d'arguments CLI.
"""

import glob
import json
import os
import sys
import time
import warnings
from datetime import datetime

import numpy as np

# ═══════════════════════════════════════════════════════════════════
# CONFIG — modifie ces valeurs selon ton besoin
# Tout est en chemins RELATIFS au dossier de ce script (portable
# Windows/Linux/Raspberry Pi), et surchargeable via variables d'env
# pour ne jamais avoir à retoucher ce fichier selon la machine.
# ═══════════════════════════════════════════════════════════════════

import os as _os
from pathlib import Path

BASE_DIR = Path(__file__).parent
STT_DIR = BASE_DIR.parent / "stt"

MODEL_PATH = _os.getenv(
    "NEXOR_WHISPER_MODEL", str(STT_DIR / "whisper-small-nexor" / "final")
)

MODE = _os.getenv("NEXOR_TEST_MODE", "dir")        # "file" | "dir"

AUDIO_FILE = _os.getenv(
    "NEXOR_AUDIO_FILE", str(STT_DIR / "test_audio" / "utt_00185_Denise.wav")
)
AUDIO_DIR = _os.getenv("NEXOR_AUDIO_DIR", str(STT_DIR / "test_audio"))

OUTPUT_JSON = _os.getenv("NEXOR_OUTPUT_JSON", str(BASE_DIR / "resultats_pipeline.json"))

# Chemin vers le dossier contenant les modules NLP. Par défaut : ce dossier
# lui-même (tous les modules NLP vivent à côté de ce script).
NLP_PIPELINE_PATH = _os.getenv("NEXOR_NLP_PATH", str(BASE_DIR))

# ID de table utilisé pour ce test — requis pour confirmer une commande.
# Le backend (ws/main.py)
# doit tourner sur STAFF_APP_URL (voir config.py, défaut http://localhost:8000)
# AVANT de lancer ce script pour voir les résultats apparaître en direct.
TABLE_ID = _os.getenv("NEXOR_TABLE_ID", "T1")

# Si True : chaque réponse du robot est aussi synthétisée et jouée sur les
# haut-parleurs (Piper TTS, voir tts_engine.py). Nécessite d'avoir lancé
# 'python tts_engine.py --setup' au moins une fois avant. Si les voix ne
# sont pas prêtes, le test continue quand même (juste un avertissement),
# jamais bloquant.
ENABLE_TTS = _os.getenv("NEXOR_ENABLE_TTS", "0") == "1"

SAMPLE_RATE = 16000

# Si True : chaque fichier audio redémarre une conversation neuve (DialogManager()).
# Si False : tous les fichiers d'un dossier s'enchaînent dans LA MÊME conversation
# (utile pour tester un vrai scénario multi-tours : salutation -> commande -> total -> confirmation).
RESET_DIALOG_PER_FILE = _os.getenv("NEXOR_RESET_PER_FILE", "1") == "1"

# Mode simulation : si le modèle Whisper OU le dossier audio est introuvable,
# on ne plante pas — on rejoue une liste de phrases synthétiques (avec bruit
# STT réaliste : hésitations, fautes) directement dans le pipeline NLP+dialog,
# pour pouvoir valider tout le pipeline EN AVAL du STT (langue, intent,
# entités, dialog_manager, notifications staff) même sans modèle/audio sous
# la main. Le vrai mode audio reste inchangé et prioritaire dès que
# MODEL_PATH et AUDIO_DIR/AUDIO_FILE existent réellement.
SIMULATE_STT_IF_MISSING = _os.getenv("NEXOR_SIMULATE_STT", "1") == "1"

_SIMULATED_TURNS = [
    ("bonjour", "fr"),
    ("euh je voudrai commander un couscous agneau sil vous plait", "fr"),
    ("ajoute aussi un café espresso", "fr"),
    ("combien ça fait ?", "fr"),
    ("c'est tout merci", "fr"),
    ("hello", "en"),
    ("i'd like to order a tagine please", "en"),
    ("that's all thank you", "en"),
    ("مرحبا", "ar"),
    ("نحب نطلب كسكسي بالحوت من فضلك", "ar"),
    ("قداش تكلف", "ar"),
    ("اكتفينا شكرا", "ar"),
]

# ═══════════════════════════════════════════════════════════════════

sys.path.insert(0, NLP_PIPELINE_PATH)


def load_whisper(model_path: str):
    from whisper_runtime import load_whisper as load_shared_whisper
    processor, model, device = load_shared_whisper(model_path)
    print(f"[whisper] Modèle chargé depuis '{model_path}' sur {device}.\n")
    return processor, model, device


def transcribe_array(audio: np.ndarray, processor, model, device) -> tuple[str, str | None]:
    from whisper_runtime import transcribe_audio
    return transcribe_audio(audio, processor, model, device, SAMPLE_RATE)


def load_audio_file(path: str) -> np.ndarray:
    import soundfile as sf

    audio, sr = sf.read(path, dtype="float32")
    if audio.ndim > 1:
        audio = audio.mean(axis=1)

    if sr != SAMPLE_RATE:
        from scipy.signal import resample
        n_samples = int(len(audio) * SAMPLE_RATE / sr)
        audio = resample(audio, n_samples).astype(np.float32)

    return audio


def load_tts():
    """
    Charge le moteur TTS si ENABLE_TTS=True. Ne bloque JAMAIS le test :
    si Piper/les voix ne sont pas prêts, on avertit et on continue sans
    audio plutôt que de planter tout le pipeline pour ça.
    """
    if not ENABLE_TTS:
        return None
    try:
        from tts_engine import get_engine
        engine = get_engine()
        print("[tts] Moteur Piper chargé — les réponses seront lues à voix haute.\n")
        return engine
    except Exception as e:
        print(f"⚠️  TTS indisponible ({e}). Le test continue SANS audio. "
              f"Vérifie 'python tts_engine.py --setup' si tu veux l'activer.\n")
        return None


def run_nlp(text: str, dm, tts=None, whisper_lang: str | None = None):
    """Fait passer le texte transcrit dans le pipeline NLP complet."""
    from preprocessing_nlp import normalize
    from language_detector import detect as detect_lang
    from intent_classifier import classify as classify_intent
    from entity_extractor import extract as extract_entities

    if not text or not text.strip():
        return {
            "lang": None, "clean_text": "", "intent": None, "score": 0.0,
            "items": [], "response": None,
        }

    lang = detect_lang(text, whisper_lang=whisper_lang)
    clean = normalize(text, lang)
    intent, score = classify_intent(clean, lang)
    entities = extract_entities(clean, lang)

    items_summary = [
        {
            "nom":       it["item"]["nom"].get(lang, it["item"]["nom"]["fr"]),
            "quantity":  it["quantity"],
            "size":      it["size"],
            "modifiers": it["modifiers"],
        }
        for it in entities.get("items", [])
    ]

    response = dm.process(text, whisper_lang)

    if tts is not None and response:
        try:
            tts.speak(response, lang)
        except Exception as e:
            import traceback

            print("="*80)
            traceback.print_exc()
            print("="*80)

    return {
        "lang":       lang,
        "clean_text": clean,
        "intent":     intent,
        "score":      round(score, 3),
        "items":      items_summary,
        "response":   response,
        "dialog_finished": dm.state.finished,
    }


def process_one_file(path: str, processor, model, device, dm, tts=None) -> dict:
    print(f"\n📁 {os.path.basename(path)}")

    t0 = time.perf_counter()
    audio = load_audio_file(path)
    duration = len(audio) / SAMPLE_RATE
    text, whisper_lang = transcribe_array(audio, processor, model, device)
    stt_latency = time.perf_counter() - t0

    print(f"   🎙️  STT ({stt_latency:.2f}s) : {text}")

    t1 = time.perf_counter()
    nlp_result = run_nlp(text, dm, tts, whisper_lang)
    nlp_latency = time.perf_counter() - t1

    print(f"   🌐 Langue    : {nlp_result['lang']}")
    print(f"   🎯 Intent    : {nlp_result['intent']} (score={nlp_result['score']})")
    if nlp_result["items"]:
        for it in nlp_result["items"]:
            print(f"      - {it['nom']} × {it['quantity']}")
    print(f"   🤖 Réponse   : {nlp_result['response']}")

    return {
        "fichier":         os.path.basename(path),
        "chemin":          path,
        "duree_audio_s":   round(duration, 2),
        "stt_latence_s":   round(stt_latency, 3),
        "stt_text":        text,
        "nlp_latence_s":   round(nlp_latency, 3),
        **nlp_result,
        "timestamp":       datetime.now().isoformat(),
    }


def save_json(results: list, path: str):
    data = {
        "model_whisper":  MODEL_PATH,
        "mode":           MODE,
        "generated_at":   datetime.now().isoformat(),
        "n_results":      len(results),
        "results":        results,
    }
    Path(path).resolve().parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    print(f"\n💾 Résultats sauvegardés dans : {path}")


def check_backend_reachable():
    """Avertit si le backend staff (ws/main.py) n'est pas joignable, pour
    ne pas découvrir seulement à la fin que rien n'a été poussé vers le
    dashboard. Ne bloque JAMAIS le test — juste un avertissement."""
    try:
        import urllib.request
        sys.path.insert(0, NLP_PIPELINE_PATH)
        from config import STAFF_APP_URL
        urllib.request.urlopen(f"{STAFF_APP_URL}/api/state", timeout=2)
        print(f"✅ Backend staff joignable sur {STAFF_APP_URL} — le dashboard va recevoir les résultats en direct.\n")
    except Exception as e:
        print(f"⚠️  Backend staff INJOIGNABLE ({e}). Le test va quand même tourner "
              f"mais les événements temps réel n'apparaîtront pas sur le dashboard. "
              f"Les événements critiques restent persistés pour reprise. "
              f"Lance 'uvicorn main:app' dans ws/ avant "
              f"de relancer si tu veux voir les résultats en direct.\n")


def _real_audio_available() -> bool:
    """Vérifie que le modèle Whisper ET l'audio source existent réellement,
    pour décider si on peut lancer le vrai pipeline audio->STT ou s'il faut
    basculer en simulation."""
    model_ok = Path(MODEL_PATH).exists()
    if MODE == "file":
        audio_ok = Path(AUDIO_FILE).exists()
    else:
        audio_ok = Path(AUDIO_DIR).is_dir() and bool(glob.glob(os.path.join(AUDIO_DIR, "*.wav")))
    return model_ok and audio_ok


def process_one_simulated(text: str, lang_hint: str, dm, tts=None) -> dict:
    """Équivalent de process_one_file() mais SANS Whisper/audio : le texte
    tient lieu de transcription STT déjà faite (utilisé quand le modèle
    Whisper fine-tuné ou les .wav de test ne sont pas disponibles sur cette
    machine). Traverse exactement le même pipeline NLP + dialog_manager en
    aval — seule l'étape STT elle-même est court-circuitée."""
    print(f"\n🗣️  (simulation STT) '{text}' [{lang_hint}]")

    t0 = time.perf_counter()
    stt_latency = time.perf_counter() - t0  # ~0, pas de vrai STT ici

    t1 = time.perf_counter()
    nlp_result = run_nlp(text, dm, tts, lang_hint)
    nlp_latency = time.perf_counter() - t1

    print(f"   🌐 Langue    : {nlp_result['lang']}")
    print(f"   🎯 Intent    : {nlp_result['intent']} (score={nlp_result['score']})")
    if nlp_result["items"]:
        for it in nlp_result["items"]:
            print(f"      - {it['nom']} × {it['quantity']}")
    print(f"   🤖 Réponse   : {nlp_result['response']}")

    return {
        "fichier":         None,
        "chemin":          None,
        "simulated":       True,
        "duree_audio_s":   0.0,
        "stt_latence_s":   round(stt_latency, 3),
        "stt_text":        text,
        "nlp_latence_s":   round(nlp_latency, 3),
        **nlp_result,
        "timestamp":       datetime.now().isoformat(),
    }


def main():
    warnings.filterwarnings("ignore")
    check_backend_reachable()

    from dialog_manager import DialogManager
    dm = DialogManager(table_id=TABLE_ID)

    results = []
    use_real_audio = _real_audio_available()

    if not use_real_audio and not SIMULATE_STT_IF_MISSING:
        print(f"❌ Modèle Whisper ({MODEL_PATH}) ou audio ({AUDIO_DIR if MODE == 'dir' else AUDIO_FILE}) "
              f"introuvable, et NEXOR_SIMULATE_STT=0 -> arrêt.")
        return

    if use_real_audio:
        processor, model, device = load_whisper(MODEL_PATH)
        tts = load_tts()

        if MODE == "file":
            results.append(process_one_file(AUDIO_FILE, processor, model, device, dm, tts))

        elif MODE == "dir":
            wavs = sorted(glob.glob(os.path.join(AUDIO_DIR, "*.wav")))
            print(f"🔎 {len(wavs)} fichier(s) trouvé(s)")

            for path in wavs:
                r = process_one_file(path, processor, model, device, dm, tts)
                results.append(r)

                if RESET_DIALOG_PER_FILE or dm.state.finished:
                    dm = DialogManager(table_id=TABLE_ID)

        else:
            print(f"❌ MODE inconnu : '{MODE}' — utilise 'file' ou 'dir'")
            return

    else:
        print(f"⚠️  Modèle Whisper ou audio réel introuvable "
              f"(MODEL_PATH={MODEL_PATH}) -> MODE SIMULATION activé "
              f"(pipeline NLP + dialog_manager testé avec des transcriptions "
              f"synthétiques à la place du vrai STT).\n")
        tts = load_tts()

        for text, lang_hint in _SIMULATED_TURNS:
            r = process_one_simulated(text, lang_hint, dm, tts)
            results.append(r)

            if RESET_DIALOG_PER_FILE and dm.state.finished:
                dm = DialogManager(table_id=TABLE_ID)

    if results:
        save_json(results, OUTPUT_JSON)
    else:
        print("⚠️  Aucun résultat à sauvegarder.")

    # Filet de sécurité final avant la sortie du script.
    from staff_app_client import flush as _flush_staff_pushes
    _flush_staff_pushes()


if __name__ == "__main__":
    main()
