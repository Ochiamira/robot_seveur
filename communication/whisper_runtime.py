"""Chargement et inférence Whisper partagés par tous les points d'entrée."""

from __future__ import annotations

import threading
from pathlib import Path

import numpy as np

from config import SAMPLE_RATE

_cache = {}
_cache_lock = threading.Lock()
SUPPORTED_LANGUAGES = {"fr", "ar", "en"}


def load_whisper(model_path):
    import torch
    from transformers import WhisperForConditionalGeneration, WhisperProcessor

    resolved = str(Path(model_path).expanduser().resolve())
    if not Path(resolved).is_dir():
        raise FileNotFoundError(
            f"Modèle Whisper introuvable: {resolved}. Définissez WHISPER_MODEL_PATH."
        )
    with _cache_lock:
        if resolved in _cache:
            return _cache[resolved]
        device = "cuda" if torch.cuda.is_available() else "cpu"
        processor = WhisperProcessor.from_pretrained(resolved, local_files_only=True)
        model = WhisperForConditionalGeneration.from_pretrained(
            resolved, local_files_only=True, low_cpu_mem_usage=True
        ).to(device)
        model.eval()
        model.generation_config.forced_decoder_ids = None
        _cache[resolved] = (processor, model, device)
        return _cache[resolved]


def detect_language(input_features, model) -> str | None:
    """Utilise l'API de détection Whisper, pas les tokens de texte générés."""
    import torch

    with torch.no_grad():
        language_ids = model.detect_language(input_features=input_features)
    token_id = int(language_ids[0].detach().cpu().item())
    mapping = getattr(model.generation_config, "lang_to_id", {}) or {}
    inverse = {int(value): key[2:-2] for key, value in mapping.items()}
    language = inverse.get(token_id)
    return language if language in SUPPORTED_LANGUAGES else None


def transcribe_audio(
    audio,
    processor,
    model,
    device,
    sample_rate=SAMPLE_RATE,
    language_hint: str | None = None,
):
    """Transcrit un signal audio, avec verrouillage optionnel de la langue.

    ``language_hint`` accepte ``fr``, ``ar`` ou ``en``. Sans indication, la
    langue est détectée par Whisper. Dans les deux cas, la langue retenue est
    transmise à ``generate()`` afin d'éviter qu'une seconde détection implicite
    produise une transcription dans une autre langue.
    """
    import torch

    audio = np.asarray(audio, dtype=np.float32).reshape(-1)
    if audio.size == 0:
        return "", None
    if not np.isfinite(audio).all():
        raise ValueError("Le signal audio contient des valeurs non finies.")
    batch = processor(
        audio, sampling_rate=sample_rate, return_tensors="pt",
        return_attention_mask=True,
    )
    features = batch.input_features.to(device)
    attention_mask = getattr(batch, "attention_mask", None)
    if attention_mask is not None:
        attention_mask = attention_mask.to(device)
    if language_hint is not None:
        language_hint = language_hint.strip().lower()
        if language_hint not in SUPPORTED_LANGUAGES:
            raise ValueError(
                "language_hint doit être 'fr', 'ar', 'en' ou None."
            )
    language = language_hint or detect_language(features, model)
    generate_kwargs = {
        "max_length": 448,
        "task": "transcribe",
        "attention_mask": attention_mask,
    }
    if language:
        generate_kwargs["language"] = language
    with torch.no_grad():
        predicted_ids = model.generate(features, **generate_kwargs)
    text = processor.batch_decode(
        predicted_ids, skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )[0].strip()
    from language_detector import detect
    return text, detect(text, whisper_lang=language)
