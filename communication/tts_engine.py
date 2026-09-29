"""
tts_engine.py
=============
Synthèse vocale (Piper TTS, hors ligne) — DERNIÈRE étape du pipeline :
prend le texte déjà généré par dialog_manager.py (INCHANGÉ, transcription
texte identique) et le transforme en voix claire, jouée directement sur
les haut-parleurs.

Piper choisi pour : voix neuronale claire (pas robotique comme eSpeak),
100% hors ligne (cohérent avec le déploiement Raspberry Pi 5, pas de
dépendance réseau en salle), assez léger pour tourner en temps réel sur
CPU ARM.

⚠️ Limite assumée : aucune voix TTS neuronale n'existe en dialecte
tunisien à ce jour — la voix arabe est en arabe standard (MSA). Même
limite que côté STT (Whisper a nécessité un fine-tuning pour comprendre
le dialecte), mais dans l'autre sens : PARLER le dialecte de façon
claire n'a pas d'outil accessible aujourd'hui.

Ce module NE TOUCHE JAMAIS AU TEXTE — dialog_manager.py reste l'unique
source de vérité sur ce que le robot "dit". tts_engine.py se contente de
le transformer en voix, sans reformuler ni tronquer.

Installation :
    pip install piper-tts sounddevice --break-system-packages
    python tts_engine.py --setup          # télécharge les 3 voix (fr/en/ar)
    python tts_engine.py --text "Bonjour" --lang fr   # test
"""

import logging
import re
import subprocess
import sys
import tempfile
import time
import unicodedata
import wave
from pathlib import Path
from typing import Optional

from config import PIPER_TIMEOUT_S, TTS_PLAYBACK_TIMEOUT_S

logger = logging.getLogger(__name__)

VOICES_DIR = Path(__file__).parent / "voices"

# Nom de voix Piper par langue (format officiel du dépôt de voix Piper).
# ar_JO-kareem-medium = seule voix arabe correcte disponible à ce jour,
# en arabe standard (MSA) — pas de dialecte tunisien, voir note ci-dessus.
VOICE_NAMES = {
    "fr": "fr_FR-siwis-medium",
    "en": "en_US-lessac-medium",
    "ar": "ar_JO-kareem-medium",
}


def prepare_spoken_text(text: str, lang: str) -> str:
    """Prépare une copie du texte pour Piper, sans modifier le texte affiché.

    La normalisation NFC conserve correctement les accents. Le symbole de
    quantité ``×`` est remplacé par un mot naturel, car les phonétiseurs Piper
    ne le prononcent pas de manière fiable selon la voix utilisée.
    """
    spoken = unicodedata.normalize("NFC", text)
    multiplication_words = {
        "fr": " fois ",
        "en": " times ",
        "ar": " في ",
    }
    spoken = spoken.replace("×", multiplication_words.get(lang, " "))
    return re.sub(r"\s+", " ", spoken).strip()


def setup_voices(langs=("fr", "en", "ar"), force: bool = False) -> None:
    """
    Télécharge les voix manquantes via le mécanisme OFFICIEL de Piper
    (piper.download_voices), plutôt que des URLs codées en dur — reste à
    jour automatiquement si Piper change son dépôt de voix.
    À lancer UNE FOIS avant la première utilisation (idéalement sur le PC
    de dev, avec une bonne connexion — chaque voix fait quelques dizaines
    de Mo), puis copier voices/ tel quel sur le Raspberry Pi 5.
    """
    VOICES_DIR.mkdir(exist_ok=True)
    for lang in langs:
        name = VOICE_NAMES[lang]
        onnx = VOICES_DIR / f"{name}.onnx"
        if onnx.exists() and not force:
            print(f"[TTS] {lang} ({name}) déjà présent, ignoré.")
            continue
        print(f"[TTS] Téléchargement voix '{name}' ({lang})...")
        cmd = [sys.executable, "-m", "piper.download_voices",
               "--download-dir", str(VOICES_DIR), name]
        if force:
            cmd.insert(-1, "--force-redownload")
        subprocess.run(cmd, check=True)
    print(f"[TTS] Voix prêtes dans {VOICES_DIR}")


class TTSEngine:
    """
    Moteur TTS multilingue (Piper). Point d'entrée principal : speak(text, lang)
    — synthétise ET joue immédiatement sur les haut-parleurs, de façon
    SYNCHRONE (bloque jusqu'à la fin — le robot ne doit pas réécouter le
    client avant d'avoir fini de parler).
    """

    def __init__(self, voices_dir: Path = VOICES_DIR, push_dashboard_state: bool = True):
        self.voices_dir = voices_dir
        self.push_dashboard_state = push_dashboard_state
        self._voice_paths_cache = {}

    def _voice_path(self, lang: str) -> Path:
        if lang not in self._voice_paths_cache:
            name = VOICE_NAMES.get(lang, VOICE_NAMES["fr"])
            path = self.voices_dir / f"{name}.onnx"
            if not path.exists():
                raise FileNotFoundError(
                    f"Voix Piper manquante pour '{lang}' : {path}\n"
                    f"Lance : python tts_engine.py --setup"
                )
            self._voice_paths_cache[lang] = path
        return self._voice_paths_cache[lang]

    def synthesize(self, text: str, lang: str) -> tuple[bytes, int, int, int]:
        """
        Génère l'audio pour ce texte, SANS le jouer. Retourne
        (pcm_bytes, sample_rate, n_channels, sample_width) — lu depuis le
        WAV que Piper écrit sur stdout, pour ne jamais supposer un format
        fixe en dur (dépend de la voix utilisée).
        """
        if not text or not text.strip():
            raise ValueError("Texte vide — rien à synthétiser")

        voice = self._voice_path(lang)
        spoken_text = prepare_spoken_text(text, lang)

        # Écrit dans un FICHIER temporaire plutôt que de capturer stdout :
        # sur Windows, un flux binaire écrit sur stdout par le processus
        # enfant peut être corrompu (conversion de fins de ligne \n -> \r\n
        # si le flux n'est pas ouvert en mode binaire côté enfant) — observé
        # en test réel (EOFError au parsing du WAV). Un fichier n'a pas ce
        # problème, sur aucun OS.
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            tmp_path = Path(tmp.name)

        try:
            subprocess.run(
                [
                    sys.executable,
                    "-X",
                    "utf8",
                    "-m",
                    "piper",
                    "-m",
                    str(voice),
                    "-f",
                    str(tmp_path),
                ],
                input=spoken_text,
                text=True,
                encoding="utf-8",
                errors="strict",
                capture_output=True,
                check=True,
                timeout=PIPER_TIMEOUT_S,
            )
            with wave.open(str(tmp_path), "rb") as wf:
                sr, ch, sw = wf.getframerate(), wf.getnchannels(), wf.getsampwidth()
                pcm = wf.readframes(wf.getnframes())
        finally:
            tmp_path.unlink(missing_ok=True)
        return pcm, sr, ch, sw

    def speak(self, text: str, lang: str) -> None:
        """
        Synthétise ET joue immédiatement (bloquant).

        Pousse "speaking" (via staff_app_client.notify_dialog_state) au
        vrai début de la lecture audio, et "idle" à la vraie fin — c'est
        précisément ce que dialog_manager.py ne peut PAS faire lui-même
        (il ne sait pas quand l'audio finit de jouer, voir son docstring
        process()). Le "speaking" poussé par dialog_manager.py au moment
        où le texte est prêt reste utile (affiche le sous-titre tout de
        suite) ; celui-ci confirme le début réel de la voix et surtout
        ramène l'écran à "idle" à la fin.
        """
        import numpy as np
        import sounddevice as sd

        if self.push_dashboard_state:
            from staff_app_client import notify_dialog_state
            notify_dialog_state("speaking", text=text, lang=lang)

        try:
            pcm, sr, ch, sw = self.synthesize(text, lang)
            dtype = {1: np.uint8, 2: np.int16, 4: np.int32}.get(sw, np.int16)
            audio = np.frombuffer(pcm, dtype=dtype)
            if ch > 1:
                audio = audio.reshape(-1, ch)
            sd.play(audio, samplerate=sr)
            deadline = time.monotonic() + TTS_PLAYBACK_TIMEOUT_S
            while sd.get_stream().active:
                if time.monotonic() >= deadline:
                    sd.stop()
                    raise TimeoutError("La lecture audio TTS a dépassé le délai maximal.")
                time.sleep(0.05)
        except FileNotFoundError as e:
            logger.error(f"[TTS] {e}")
            raise
        except subprocess.CalledProcessError as e:
            stderr = e.stderr
            if isinstance(stderr, bytes):
                stderr = stderr.decode("utf-8", errors="replace")
            logger.error(f"[TTS] Piper a échoué : {stderr or ''}")
            raise
        except subprocess.TimeoutExpired as e:
            logger.error("[TTS] Piper a dépassé le délai maximal")
            raise TimeoutError("Piper n'a pas répondu dans le délai maximal.") from e
        finally:
            if self.push_dashboard_state:
                from staff_app_client import notify_dialog_state
                notify_dialog_state("idle", text="", lang=lang)


_engine: Optional[TTSEngine] = None


def get_engine() -> TTSEngine:
    global _engine
    if _engine is None:
        _engine = TTSEngine()
    return _engine


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Test du moteur TTS NEXOR (Piper)")
    parser.add_argument("--setup", action="store_true", help="Télécharge les 3 voix (fr/en/ar)")
    parser.add_argument("--force", action="store_true", help="Force le retéléchargement (avec --setup)")
    parser.add_argument("--text", default=None, help="Texte à synthétiser")
    parser.add_argument("--lang", default="fr", choices=["fr", "en", "ar"])
    parser.add_argument("--save", default=None, help="Sauvegarder en .wav au lieu de jouer sur les haut-parleurs")
    args = parser.parse_args()

    if args.setup:
        setup_voices(force=args.force)

    if args.text:
        engine = TTSEngine(push_dashboard_state=False)
        if args.save:
            pcm, sr, ch, sw = engine.synthesize(args.text, args.lang)
            with wave.open(args.save, "wb") as f:
                f.setnchannels(ch); f.setsampwidth(sw); f.setframerate(sr)
                f.writeframes(pcm)
            print(f"[TTS] Sauvegardé → {args.save} ({sr} Hz, {ch} canal/aux, {sw*8} bits)")
        else:
            print(f"[TTS] Lecture ({args.lang}) : {args.text}")
            engine.speak(args.text, args.lang)
