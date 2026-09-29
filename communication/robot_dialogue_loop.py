"""
robot_dialogue_loop.py
=======================
LA vraie boucle du robot, en conditions réelles — différente de
test_full_pipeline.py (qui rejoue des fichiers .wav déjà enregistrés) :
ici le micro écoute en continu, détecte quand le client a fini de parler,
transcrit, fait répondre dialog_manager, puis fait parler le robot.

    micro (VAD simple) → Whisper (STT) → dialog_manager.process()
        → Piper TTS (voix) → haut-parleur

C'est ICI — et seulement ici — que l'état "listening" peut être poussé
vers l'écran client : ni dialog_manager.py ni tts_engine.py ne savent
quand le micro enregistre réellement (documenté dans leurs docstrings
respectifs). Ce script est le "chef d'orchestre" qui connaît le cycle
complet écoute → traite → parle → réécoute.

⚠️ Ce que ce script NE fait PAS (hors scope, dépend de ton robot/ROS2) :
  - Savoir QUAND s'approcher d'une table (ça, c'est vision_bridge.py /
    l'orchestrateur vision qui le décide)
  - Saluer le client de façon PROACTIVE dès l'arrivée, dans SA langue —
    aujourd'hui le robot attend que le client parle en premier pour
    détecter la langue (voir la note greet_or_wait() plus bas). Si tu
    veux un vrai "bonjour" proactif multilingue, il faut trancher CETTE
    décision avant de le coder (voir mon message).

Installation (en plus de dialog_manager.py / tts_engine.py déjà en place) :
    pip install sounddevice numpy torch transformers soundfile --break-system-packages
"""

import logging
import os
import sys
import threading
import time
import warnings
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)

# ═══════════════════════════════════════════════════════════════════
# CONFIG
# ═══════════════════════════════════════════════════════════════════

NLP_PIPELINE_PATH = str(Path(__file__).resolve().parent)
if NLP_PIPELINE_PATH not in sys.path:
    sys.path.insert(0, NLP_PIPELINE_PATH)

from config import (
    AUDIO_INPUT_DEVICE,
    AUDIO_INPUT_HOST_API,
    AUDIO_INPUT_SAMPLE_RATE,
    MAX_RECORD_S,
    MIN_SPEECH_S,
    SAMPLE_RATE,
    SILENCE_DURATION_S,
    SILENCE_THRESHOLD,
    SPEECH_START_TIMEOUT_S,
    WHISPER_MODEL_PATH,
)

MODEL_PATH = str(WHISPER_MODEL_PATH)
_dialogue_runtime_lock = threading.Lock()


def _resolve_input_device(sd):
    """Retourne un index PortAudio stable, en préférant WASAPI sous Windows."""
    configured = AUDIO_INPUT_DEVICE.strip()
    if configured:
        try:
            return int(configured)
        except ValueError:
            pass

    devices = sd.query_devices()
    hostapis = sd.query_hostapis()
    candidates = []
    needle = configured.casefold()
    preferred_api = AUDIO_INPUT_HOST_API.casefold()

    for index, info in enumerate(devices):
        if int(info.get("max_input_channels", 0)) < 1:
            continue
        name = str(info.get("name", ""))
        if needle and needle not in name.casefold():
            continue
        host_index = int(info.get("hostapi", -1))
        host_name = (
            str(hostapis[host_index].get("name", ""))
            if 0 <= host_index < len(hostapis) else ""
        )
        score = 0
        if preferred_api and preferred_api in host_name.casefold():
            score += 100
        if "microphone array" in name.casefold():
            score += 20
        if "wasapi" in host_name.casefold():
            score += 10
        candidates.append((score, index))

    if configured and not candidates:
        raise RuntimeError(f"Périphérique audio d'entrée introuvable: {configured}")
    if candidates and (configured or os.name == "nt"):
        return max(candidates)[1]
    return None


def input_audio_settings(sd=None) -> tuple[int | None, int, str]:
    """Résout le périphérique et sa fréquence native de façon vérifiable."""
    if sd is None:
        import sounddevice as sd
    device = _resolve_input_device(sd)
    info = sd.query_devices(device, "input")
    capture_rate = AUDIO_INPUT_SAMPLE_RATE or int(round(info["default_samplerate"]))
    if capture_rate <= 0:
        raise RuntimeError("Fréquence native du microphone invalide")
    hostapis = sd.query_hostapis()
    host_index = int(info.get("hostapi", -1))
    host_name = (
        str(hostapis[host_index].get("name", ""))
        if 0 <= host_index < len(hostapis) else "inconnue"
    )
    label = f"{info['name']} / {host_name} / {capture_rate} Hz"
    return device, capture_rate, label


def validate_audio_input() -> str:
    """Échoue avant d'accepter un dialogue si le microphone est inutilisable."""
    import sounddevice as sd
    device, capture_rate, label = input_audio_settings(sd)
    sd.check_input_settings(
        device=device,
        channels=1,
        dtype="float32",
        samplerate=capture_rate,
    )
    return label


def _resample_audio(audio: np.ndarray, source_rate: int, target_rate: int) -> np.ndarray:
    signal = np.asarray(audio, dtype=np.float32).reshape(-1)
    if signal.size == 0 or source_rate == target_rate:
        return signal
    target_size = max(1, int(round(signal.size * target_rate / source_rate)))
    source_x = np.arange(signal.size, dtype=np.float64)
    target_x = np.linspace(0.0, signal.size - 1, target_size, dtype=np.float64)
    return np.interp(target_x, source_x, signal).astype(np.float32)

# ── Détection de fin de parole (VAD simple, par énergie RMS) ─────────────
# Pas de dépendance lourde (pas de webrtcvad) — suffisant pour un
# environnement de restaurant raisonnablement calme. À CALIBRER chez toi :
# lance calibrate_silence_threshold() ci-dessous pour mesurer ton propre
# bruit de fond avant la première utilisation réelle.
from whisper_runtime import load_whisper as _load_shared_whisper
from whisper_runtime import transcribe_audio
from audio_vad import select_speech_chunks, validate_vad_settings


def calibrate_silence_threshold(duration_s: float = 3.0) -> float:
    """
    Mesure le bruit de fond ambiant pendant duration_s secondes et suggère
    un SILENCE_THRESHOLD adapté. À lancer UNE FOIS dans l'environnement
    réel (salle de restaurant), pas dans un bureau silencieux — le bruit
    ambiant d'un restaurant est bien plus élevé.

    Usage :
        python robot_dialogue_loop.py --calibrate
    """
    import sounddevice as sd
    device, capture_rate, label = input_audio_settings(sd)
    print(f"[CALIBRATION] Entrée : {label}")
    print(f"[CALIBRATION] Silence pendant {duration_s}s, laisse l'environnement ambiant tel quel...")
    audio = sd.rec(
        int(duration_s * capture_rate), samplerate=capture_rate,
        channels=1, dtype="float32", device=device,
    )
    sd.wait()
    rms = float(np.sqrt(np.mean(audio ** 2)))
    suggested = round(rms * 3, 4)   # marge x3 au-dessus du bruit de fond mesuré
    print(f"[CALIBRATION] RMS bruit de fond mesuré : {rms:.4f}")
    print(f"[CALIBRATION] SILENCE_THRESHOLD suggéré : {suggested} (copie cette valeur dans la config)")
    return suggested


def select_speech_audio(
    chunks: list[np.ndarray],
    threshold: float,
    sample_rate: int = SAMPLE_RATE,
    min_speech_s: float = MIN_SPEECH_S,
    pre_roll_chunks: int = 2,
) -> np.ndarray:
    """Extrait la zone de parole et refuse le silence/bruit trop court."""
    return select_speech_chunks(
        chunks, threshold, sample_rate, min_speech_s, pre_roll_chunks
    )


def record_until_silence(
    sample_rate: int = SAMPLE_RATE,
    threshold: float = None,
) -> np.ndarray:
    """
    Enregistre depuis le micro jusqu'à détecter un silence prolongé APRÈS
    de la parole. Retourne un tableau vide si rien d'exploitable n'a été
    capté (silence total, ou bruit trop court) — le code appelant doit
    gérer ce cas en réessayant, PAS en plantant.
    """
    import sounddevice as sd

    threshold = SILENCE_THRESHOLD if threshold is None else threshold
    device, capture_rate, label = input_audio_settings(sd)
    validate_vad_settings(threshold, MIN_SPEECH_S, capture_rate)
    logger.info("[mic] Entrée active: %s", label)
    chunks = []
    speech_started = False
    consecutive_voice = 0
    silence_start = None
    t0 = time.time()

    def callback(indata, frames, time_info, status):
        if status:
            logger.debug(f"[mic] {status}")
        chunks.append(indata.copy())

    blocksize = max(1, int(capture_rate * 0.03))
    with sd.InputStream(
        samplerate=capture_rate, channels=1, dtype="float32",
        blocksize=blocksize, callback=callback, device=device,
    ):
        while True:
            time.sleep(0.05)
            if not chunks:
                continue

            rms = float(np.sqrt(np.mean(chunks[-1] ** 2)))

            if rms > threshold:
                consecutive_voice += 1
                if consecutive_voice >= 3:
                    speech_started = True
                silence_start = None
            elif speech_started:
                consecutive_voice = 0
                if silence_start is None:
                    silence_start = time.time()
                elif time.time() - silence_start >= SILENCE_DURATION_S:
                    break

            elapsed = time.time() - t0
            if not speech_started and elapsed >= SPEECH_START_TIMEOUT_S:
                break
            if elapsed >= MAX_RECORD_S:
                logger.warning("[mic] MAX_RECORD_S atteint — coupure de sécurité")
                break

    if not speech_started:
        return np.array([], dtype=np.float32)
    selected = select_speech_audio(chunks, threshold, capture_rate)
    return _resample_audio(selected, capture_rate, sample_rate)


def load_whisper(model_path: str):
    return _load_shared_whisper(model_path)


def prepare_dialogue_runtime():
    """Précharge STT, NLP, LLM et TTS sans ouvrir le microphone."""
    from intent_classifier import warmup_intent_classifier
    from llm_engine import get_engine as get_llm_engine
    from tts_engine import get_engine

    # Empêche le thread de préchargement et le thread d'arrivée de créer
    # simultanément deux moteurs TTS si le robot arrive très rapidement.
    with _dialogue_runtime_lock:
        llm = get_llm_engine()
        # Ollama vit dans un processus séparé : son chargement peut se faire
        # en parallèle du chargement PyTorch local de Whisper / DistilBERT.
        with ThreadPoolExecutor(max_workers=2) as pool:
            ollama_future = pool.submit(llm.warmup)
            processor, model, device = load_whisper(MODEL_PATH)
            tts = get_engine()
            intent_ready = warmup_intent_classifier()
            ollama_ready = ollama_future.result()
        logger.info(
            "[PRELOAD] Whisper=%s TTS=ok DistilBERT=%s Ollama=%s",
            device,
            "ok" if intent_ready else "absent",
            "ok" if ollama_ready else "indisponible",
        )
    return processor, model, device, tts


def transcribe_array(
    audio: np.ndarray,
    processor,
    model,
    device,
    language_hint: str | None = None,
) -> tuple[str, str | None]:
    return transcribe_audio(
        audio,
        processor,
        model,
        device,
        language_hint=language_hint,
    )


def run_table(
    table_id: str,
    max_empty_retries: int = 3,
    silence_threshold: float = SILENCE_THRESHOLD,
    transcription_language: str = "auto",
    on_ready=None,
) -> dict:
    """
    Boucle complète pour UNE table, du premier tour jusqu'à la commande
    confirmée. À appeler quand le robot
    arrive physiquement devant la table (déclenché par ta navigation/
    vision, hors scope de ce fichier).
    """
    from dialog_manager import DialogManager
    from staff_app_client import flush, notify_dialog_state

    try:
        processor, model, device, tts = prepare_dialogue_runtime()
        dm = DialogManager(table_id=table_id)
    except Exception as exc:
        logger.exception("Initialisation du pipeline de communication impossible")
        raise RuntimeError(f"Initialisation impossible: {exc}") from exc

    if on_ready is not None:
        on_ready()

    print(f"\n🤖 Robot prêt devant la table {table_id}. En écoute...\n")

    greeting = os.environ.get(
        "NEXOR_DIALOG_GREETING",
        "Bonjour, je suis prêt à prendre votre commande. Vous pouvez parler.",
    ).strip()
    if greeting:
        try:
            tts.speak(greeting, os.environ.get("NEXOR_DIALOG_GREETING_LANG", "fr"))
        except Exception as exc:
            logger.error("[TTS] Message d'accueil impossible: %s", exc)

    empty_streak = 0
    try:
        while not dm.state.finished:
            notify_dialog_state("listening", text="", lang=dm.state.lang)
            print("🎤 (écoute...)")

            try:
                capture_t0 = time.perf_counter()
                audio = record_until_silence(threshold=silence_threshold)
                logger.info(
                    "[LATENCE] Capture/VAD: %.2fs",
                    time.perf_counter() - capture_t0,
                )
            except Exception as exc:
                logger.error("[mic] Échec de capture: %s", exc)
                return {
                    "status": "audio_error",
                    "confirmed": False,
                    "table_id": table_id,
                    "order_id": None,
                }

            if audio.size == 0:
                empty_streak += 1
                print(f"   (rien entendu — {empty_streak}/{max_empty_retries})")
                if empty_streak >= max_empty_retries:
                    print("   Personne ne répond, fin de la session pour cette table.")
                    return {
                        "status": "no_response",
                        "confirmed": False,
                        "table_id": table_id,
                        "order_id": None,
                    }
                continue
            empty_streak = 0

            try:
                stt_t0 = time.perf_counter()
                language_hint = (
                    None if transcription_language == "auto"
                    else transcription_language
                )
                text, whisper_lang = transcribe_array(
                    audio,
                    processor,
                    model,
                    device,
                    language_hint=language_hint,
                )
                logger.info(
                    "[LATENCE] STT Whisper: %.2fs",
                    time.perf_counter() - stt_t0,
                )
            except Exception as exc:
                logger.error("[STT] Échec de transcription: %s", exc)
                continue
            if not text:
                empty_streak += 1
                continue
            print(f"   🗣️  Client : {text}")

        # dialog_manager.process() pousse déjà "processing" puis "speaking"
        # (texte) tout seul — voir son docstring. On n'a rien à faire ici
        # pour ça, juste appeler process().
            nlp_t0 = time.perf_counter()
            response = dm.process(text, whisper_lang=whisper_lang)
            logger.info(
                "[LATENCE] NLP/LLM: %.2fs",
                time.perf_counter() - nlp_t0,
            )
            print(f"   🤖 Robot  : {response}")

        # tts.speak() pousse "speaking" (audio réel) puis "idle" à la fin —
        # c'est lui qui gère le vrai cycle audio, dialog_manager ne le sait pas.
            try:
                tts_t0 = time.perf_counter()
                tts.speak(response, dm.state.lang)
                logger.info(
                    "[LATENCE] TTS + lecture: %.2fs",
                    time.perf_counter() - tts_t0,
                )
            except Exception as e:
                logger.error(f"[TTS] Échec de la synthèse vocale : {e}")
            # Le dialogue continue même sans voix (dégradation, pas de blocage) —
            # mais NOTE : sans TTS le client n'entend rien, donc en pratique le
            # dialogue va probablement mal se dérouler à partir d'ici. On log
                # fort plutôt que de planter, pour ne pas perdre toute la session.

            # DialogManager a deja place la commande confirmee dans la file
            # fiable de l'application. Le TTS vient de se terminer : le robot
            # peut maintenant quitter la table sans couper sa reponse finale.
            if dm.state.confirmed:
                print(f"\n✅ Commande finalisée pour la table {table_id}.\n")
                return {
                    "status": "confirmed",
                    "confirmed": True,
                    "table_id": table_id,
                    "order_id": dm.state.order_id,
                    "lang": dm.state.lang,
                }
    finally:
        notify_dialog_state("idle", text="", lang=dm.state.lang)
        flush(timeout=2.0)

    return {
        "status": "conversation_ended",
        "confirmed": False,
        "table_id": table_id,
        "order_id": None,
        "lang": dm.state.lang,
    }


if __name__ == "__main__":
    import argparse

    warnings.filterwarnings("ignore")
    logging.basicConfig(level=logging.INFO)

    parser = argparse.ArgumentParser(description="Boucle de dialogue temps réel NEXOR")
    parser.add_argument("--table", default="T1", help="ID de la table (ex: T1, T4)")
    parser.add_argument("--calibrate", action="store_true",
                         help="Mesure le bruit ambiant et suggère un SILENCE_THRESHOLD")
    parser.add_argument("--threshold", type=float, default=SILENCE_THRESHOLD,
                        help="Seuil RMS VAD (sinon SILENCE_THRESHOLD/config env)")
    parser.add_argument(
        "--language",
        choices=("auto", "fr", "ar", "en"),
        default="auto",
        help="Langue STT imposée, ou auto pour la détection Whisper",
    )
    args = parser.parse_args()

    if args.calibrate:
        calibrate_silence_threshold()
    else:
        run_table(
            args.table,
            silence_threshold=args.threshold,
            transcription_language=args.language,
        )
