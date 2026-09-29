"""
test_whisper_simple.py
=======================
Teste ton modèle Whisper Small fine-tuné, SEUL, sans le pipeline NLP.
Les résultats sont sauvegardés dans un fichier JSON.

Tout se configure ci-dessous, dans la section CONFIG — pas d'arguments
en ligne de commande.
"""

import glob
import json
import os
import time
import warnings
from datetime import datetime
from pathlib import Path

import numpy as np

# ═══════════════════════════════════════════════════════════════════
# CONFIG — modifie ces valeurs selon ton besoin
# ═══════════════════════════════════════════════════════════════════

STT_DIR = Path(os.getenv(
    "STT_BASE_DIR", str(Path(__file__).resolve().parents[1] / "stt")
)).resolve()
MODEL_PATH = str(Path(os.getenv(
    "WHISPER_MODEL_PATH", str(STT_DIR / "whisper-small-nexor" / "final")
)).resolve())

MODE = os.getenv("WHISPER_TEST_MODE", "dir")        # "file" | "dir" | "mic"

AUDIO_FILE = str(Path(os.getenv(
    "WHISPER_TEST_AUDIO", str(STT_DIR / "test_audio" / "salutation_0000.wav")
)).resolve())
AUDIO_DIR = str(Path(os.getenv(
    "WHISPER_TEST_AUDIO_DIR", str(STT_DIR / "test_audio")
)).resolve())

OUTPUT_JSON = str(Path(os.getenv(
    "WHISPER_TEST_OUTPUT", str(STT_DIR / "resultats_stt.json")
)).resolve())

SAMPLE_RATE = 16000
MAX_RECORD_SECONDS = 15

# ═══════════════════════════════════════════════════════════════════


def load_model(model_path: str):
    import torch
    from transformers import WhisperForConditionalGeneration, WhisperProcessor

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[whisper] Chargement du modèle depuis '{model_path}' sur {device}...")

    processor = WhisperProcessor.from_pretrained(model_path)
    model = WhisperForConditionalGeneration.from_pretrained(model_path).to(device)
    model.eval()
    model.generation_config.forced_decoder_ids = None  # neutralisé (bug connu)

    print("[whisper] Modèle chargé avec succès.\n")
    return processor, model, device


def transcribe_array(audio: np.ndarray, processor, model, device) -> str:
    import torch

    if audio.dtype != np.float32:
        audio = audio.astype(np.float32)

    inputs = processor(
        audio, sampling_rate=SAMPLE_RATE, return_tensors="pt",
        return_attention_mask=True,
    )
    input_features = inputs.input_features.to(device)
    attention_mask = inputs.attention_mask.to(device)

    with torch.no_grad():
        predicted_ids = model.generate(
            input_features, attention_mask=attention_mask, max_new_tokens=128
        )

    return processor.batch_decode(predicted_ids, skip_special_tokens=True)[0].strip()


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


def transcribe_file(path: str, processor, model, device) -> dict:
    t0 = time.perf_counter()
    audio = load_audio_file(path)
    duration = len(audio) / SAMPLE_RATE
    text = transcribe_array(audio, processor, model, device)
    latency = time.perf_counter() - t0

    print(f"📁 {os.path.basename(path)}")
    print(f"   Durée audio : {duration:.1f}s  |  Latence transcription : {latency:.2f}s")
    print(f"   📝 Transcription : {text}\n")

    return {
        "fichier":          os.path.basename(path),
        "chemin":           path,
        "duree_audio_s":    round(duration, 2),
        "latence_s":        round(latency, 3),
        "transcription":    text,
        "timestamp":        datetime.now().isoformat(),
    }


def record_and_transcribe(processor, model, device) -> dict:
    import sounddevice as sd

    input("🎤 Appuie sur Entrée pour COMMENCER à parler...")
    print(f"🔴 Enregistrement... (Entrée pour arrêter, {MAX_RECORD_SECONDS}s max)")

    frames = []
    stream = sd.InputStream(samplerate=SAMPLE_RATE, channels=1, dtype="float32")
    stream.start()

    start = time.time()
    while not _enter_pressed() and (time.time() - start) < MAX_RECORD_SECONDS:
        data, _ = stream.read(1024)
        frames.append(data.copy())

    stream.stop()
    stream.close()
    print("⏹️  Enregistrement terminé, transcription en cours...")

    if not frames:
        print("⚠️  Rien enregistré.\n")
        return None

    audio = np.concatenate(frames, axis=0).flatten()
    if len(audio) < int(0.4 * SAMPLE_RATE) or float(np.max(np.abs(audio))) < 0.005:
        print("⚠️  Aucun signal vocal exploitable.\n")
        return None
    duration = len(audio) / SAMPLE_RATE
    t0 = time.perf_counter()
    text = transcribe_array(audio, processor, model, device)
    latency = time.perf_counter() - t0

    print(f"📝 Transcription ({latency:.2f}s) : {text}\n")

    return {
        "fichier":          "micro_live",
        "chemin":           None,
        "duree_audio_s":    round(duration, 2),
        "latence_s":        round(latency, 3),
        "transcription":    text,
        "timestamp":        datetime.now().isoformat(),
    }


def _enter_pressed() -> bool:
    if os.name == "nt":
        import msvcrt
        if not msvcrt.kbhit():
            return False
        while msvcrt.kbhit():
            if msvcrt.getwch() in {"\r", "\n"}:
                return True
        return False

    import select
    import sys
    ready, _, _ = select.select([sys.stdin], [], [], 0)
    if ready:
        sys.stdin.readline()
        return True
    return False


def save_json(results: list, path: str):
    data = {
        "model_path":   MODEL_PATH,
        "mode":         MODE,
        "generated_at": datetime.now().isoformat(),
        "n_results":    len(results),
        "results":      results,
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    print(f"💾 Résultats sauvegardés dans : {path}")


def main():
    warnings.filterwarnings("ignore")
    processor, model, device = load_model(MODEL_PATH)

    results = []

    if MODE == "file":
        results.append(transcribe_file(AUDIO_FILE, processor, model, device))

    elif MODE == "dir":
        wavs = sorted(glob.glob(os.path.join(AUDIO_DIR, "*.wav")))
        if not wavs:
            print(f"⚠️  Aucun .wav trouvé dans {AUDIO_DIR}")
            return
        print(f"🔎 {len(wavs)} fichier(s) trouvé(s)\n")
        for path in wavs:
            results.append(transcribe_file(path, processor, model, device))

    elif MODE == "mic":
        print("🚀 Mode micro — CTRL+C pour quitter et sauvegarder\n")
        try:
            while True:
                r = record_and_transcribe(processor, model, device)
                if r:
                    results.append(r)
        except KeyboardInterrupt:
            print("\n👋 Arrêt demandé.")

    else:
        print(f"❌ MODE inconnu : '{MODE}' — utilise 'file', 'dir' ou 'mic'")
        return

    if results:
        save_json(results, OUTPUT_JSON)
    else:
        print("⚠️  Aucun résultat à sauvegarder.")


if __name__ == "__main__":
    main()
