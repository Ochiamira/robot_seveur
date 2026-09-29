"""
stt_nlp_bridge.py
==================
Pont entre le Whisper Small fine-tuné (STT) et le pipeline NLP+LLM NEXOR.

Deux modes d'entrée audio :
  - Micro en direct  : appuie sur Entrée pour démarrer l'enregistrement,
                        ré-appuie sur Entrée pour l'arrêter et transcrire.
  - Fichier audio     : transcrit un .wav/.mp3 déjà enregistré (utile pour
                        rejouer un même test plusieurs fois, ou benchmarker).

⚠️ À ADAPTER avant utilisation :
  - MODEL_PATH : chemin vers ton dossier de modèle Whisper fine-tuné
    (celui produit par ton script d'entraînement Kaggle, rapatrié en local).
  - SAMPLE_RATE : doit correspondre à ce sur quoi le modèle a été entraîné
    (16000 Hz est la valeur standard Whisper — laisse tel quel sauf raison
    contraire).

Prérequis (déjà présents sur ta machine si tu as fine-tuné Whisper) :
    pip install transformers torch sounddevice numpy scipy

Usage :
    python stt_nlp_bridge.py                      # mode micro en direct
    python stt_nlp_bridge.py --file audio.wav      # transcrit un fichier
    python stt_nlp_bridge.py --debug               # affiche chaque étage NLP
"""

import argparse
import os
import sys
import time
import warnings
from pathlib import Path

import numpy as np

# ── Config à adapter ────────────────────────────────────────────────────────
from config import MAX_RECORD_S, SAMPLE_RATE, SILENCE_THRESHOLD, WHISPER_MODEL_PATH
from audio_vad import filter_speech_audio, validate_vad_settings
from whisper_runtime import load_whisper as _load_shared_whisper
from whisper_runtime import transcribe_audio

MODEL_PATH = str(WHISPER_MODEL_PATH)
MAX_RECORD_SECONDS = MAX_RECORD_S


# ── Chargement paresseux du modèle Whisper (lourd) ─────────────────────────
def load_whisper(model_path: str):
    """Charge le modèle via le runtime partagé et mis en cache."""
    return _load_shared_whisper(model_path)


def transcribe(audio: np.ndarray, processor, model, device) -> tuple[str, str | None]:
    """Transcrit un signal audio via le runtime Whisper partagé."""
    return transcribe_audio(audio, processor, model, device)


# ── Entrée micro ─────────────────────────────────────────────────────────
def _enter_pressed() -> bool:
    if os.name == "nt":
        import msvcrt
        pressed = False
        while msvcrt.kbhit():
            if msvcrt.getwch() in {"\r", "\n"}:
                pressed = True
        return pressed
    import select
    ready, _, _ = select.select([sys.stdin], [], [], 0)
    if ready:
        sys.stdin.readline()
        return True
    return False


def record_from_mic(threshold: float = SILENCE_THRESHOLD) -> np.ndarray:
    """Enregistre depuis le micro. Entrée pour démarrer, Entrée pour arrêter."""
    import sounddevice as sd

    input("  🎤 Appuie sur Entrée pour COMMENCER à parler...")
    print("  🔴 Enregistrement... (Entrée pour arrêter, ou attends "
          f"{MAX_RECORD_SECONDS}s max)")

    frames = []
    stream = sd.InputStream(
        samplerate=SAMPLE_RATE, channels=1, dtype="float32"
    )
    try:
        stream.start()
        start = time.monotonic()
        while not _enter_pressed() and (time.monotonic() - start) < MAX_RECORD_SECONDS:
            data, overflowed = stream.read(1024)
            if overflowed:
                warnings.warn("Débordement du tampon microphone", RuntimeWarning)
            frames.append(data.copy())
    finally:
        stream.stop()
        stream.close()
    print("  ⏹️  Enregistrement terminé.")

    if not frames:
        return np.array([], dtype=np.float32)
    audio = np.concatenate(frames, axis=0).flatten()
    return filter_speech_audio(audio, threshold=threshold)


def load_from_file(path: str) -> np.ndarray:
    """Charge un fichier audio (.wav natif ; .mp3 nécessite ffmpeg via pydub)."""
    import soundfile as sf

    audio, sr = sf.read(path, dtype="float32")
    if audio.ndim > 1:
        audio = audio.mean(axis=1)  # stéréo -> mono

    if sr != SAMPLE_RATE:
        from scipy.signal import resample
        n_samples = int(len(audio) * SAMPLE_RATE / sr)
        audio = resample(audio, n_samples).astype(np.float32)

    return audio


# ── Pont vers le pipeline NLP ───────────────────────────────────────────────
def run_nlp_turn(text: str, dm, debug: bool = False, whisper_lang: str = None):
    """Fait passer le texte transcrit dans le pipeline NLP complet."""
    from preprocessing_nlp import normalize
    from language_detector import detect as detect_lang
    from intent_classifier import classify as classify_intent
    from entity_extractor import extract as extract_entities

    if not text or not text.strip():
        print("  ⚠️  Transcription vide — rien à traiter.")
        return

    # 1. Langue : Whisper ne la fournit pas ici -> détection par mots-clés
    lang = detect_lang(text, whisper_lang=whisper_lang)

    # 2. Nettoyage
    clean = normalize(text, lang)

    if debug:
        intent, score = classify_intent(clean, lang)
        entities = extract_entities(clean, lang)
        print(f"  📝 Brut Whisper   : {text}")
        print(f"  🧹 Normalisé      : {clean}")
        print(f"  🌐 Langue détectée: {lang}")
        print(f"  🎯 Intent         : {intent} (score={score:.2f})")
        if entities["items"]:
            for it in entities["items"]:
                print(f"     - {it['item']['nom'].get(lang, it['item']['nom']['fr'])} "
                      f"× {it['quantity']}"
                      f"{' (' + it['size'] + ')' if it['size'] else ''}"
                      f"{' [' + ','.join(it['modifiers']) + ']' if it['modifiers'] else ''}")

    # 3. Orchestrateur complet (gère aussi le LLM si besoin)
    t0 = time.perf_counter()
    response = dm.process(clean, whisper_lang=lang)
    latency = time.perf_counter() - t0

    print(f"  🎤 [{lang}] {text}")
    print(f"  🤖 ({latency:.2f}s) {response}\n")

    if dm.state.finished:
        print("  ✅ Commande terminée.\n")


# ── Point d'entrée ───────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--file", help="Transcrire un fichier audio au lieu du micro")
    parser.add_argument("--debug", action="store_true", help="Afficher chaque étage NLP")
    parser.add_argument("--model", default=MODEL_PATH, help="Chemin du modèle Whisper fine-tuné")
    parser.add_argument("--table", default=os.getenv("TABLE_ID"), help="Identifiant obligatoire de la table")
    parser.add_argument("--threshold", type=float, default=SILENCE_THRESHOLD, help="Seuil RMS du filtre vocal")
    args = parser.parse_args()

    if not args.table or not args.table.strip():
        parser.error("--table est obligatoire (ou définissez TABLE_ID).")
    try:
        validate_vad_settings(args.threshold, 0.4, SAMPLE_RATE)
    except ValueError as exc:
        parser.error(str(exc))

    warnings.filterwarnings("ignore")

    try:
        processor, model, device = load_whisper(args.model)
    except Exception as exc:
        parser.error(str(exc))

    from dialog_manager import DialogManager
    dm = DialogManager(table_id=args.table.strip())

    if args.file:
        # ── Mode fichier unique ──────────────────────────────────────────
        print(f"\n[stt_nlp_bridge] Transcription de {args.file}...")
        audio = load_from_file(args.file)
        audio = filter_speech_audio(audio, threshold=args.threshold)
        if audio.size == 0:
            parser.error("Le fichier audio est vide ou ne contient pas assez de parole.")
        text, whisper_lang = transcribe(audio, processor, model, device)
        run_nlp_turn(text, dm, debug=args.debug, whisper_lang=whisper_lang)
        return

    # ── Mode micro en direct (boucle de conversation) ───────────────────
    print("\n🚀 NEXOR — Liaison STT → NLP (mode micro)")
    print("   CTRL+C pour quitter, tape 'reset' + Entrée pendant une pause pour redémarrer\n")

    try:
        while True:
            audio = record_from_mic(threshold=args.threshold)
            if audio.size == 0:
                continue

            t0 = time.perf_counter()
            text, whisper_lang = transcribe(audio, processor, model, device)
            stt_latency = time.perf_counter() - t0
            print(f"  ⏱️  STT : {stt_latency*1000:.0f}ms")

            run_nlp_turn(text, dm, debug=args.debug, whisper_lang=whisper_lang)

            if dm.state.finished:
                dm = DialogManager(table_id=args.table.strip())
                print("  ♻️  Nouvelle session démarrée automatiquement.\n")

    except KeyboardInterrupt:
        print("\n\n👋 Session terminée.")


if __name__ == "__main__":
    main()
