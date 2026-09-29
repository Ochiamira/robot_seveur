"""Filtrage d'énergie commun aux entrées micro et fichier."""

import numpy as np

from config import MIN_SPEECH_S, SAMPLE_RATE, SILENCE_THRESHOLD


def validate_vad_settings(threshold, min_speech_s, sample_rate):
    if not 0 < float(threshold) < 1:
        raise ValueError("Le seuil VAD doit être strictement compris entre 0 et 1.")
    if float(min_speech_s) <= 0:
        raise ValueError("La durée minimale de parole doit être positive.")
    if int(sample_rate) <= 0:
        raise ValueError("La fréquence audio doit être positive.")


def select_speech_chunks(chunks, threshold=SILENCE_THRESHOLD,
                         sample_rate=SAMPLE_RATE, min_speech_s=MIN_SPEECH_S,
                         pre_roll_chunks=2):
    validate_vad_settings(threshold, min_speech_s, sample_rate)
    if not chunks:
        return np.array([], dtype=np.float32)
    normalized = [np.asarray(chunk, dtype=np.float32).reshape(-1) for chunk in chunks]
    energies = [float(np.sqrt(np.mean(np.square(chunk, dtype=np.float64))))
                if chunk.size else 0.0 for chunk in normalized]
    voiced = [index for index, rms in enumerate(energies) if rms > threshold]
    if not voiced:
        return np.array([], dtype=np.float32)
    voiced_samples = sum(normalized[index].size for index in voiced)
    if voiced_samples / sample_rate < min_speech_s:
        return np.array([], dtype=np.float32)
    start = max(0, voiced[0] - pre_roll_chunks)
    end = voiced[-1] + 1
    return np.concatenate(normalized[start:end]).astype(np.float32)


def filter_speech_audio(audio, threshold=SILENCE_THRESHOLD,
                        sample_rate=SAMPLE_RATE, min_speech_s=MIN_SPEECH_S,
                        frame_ms=30):
    signal = np.asarray(audio, dtype=np.float32).reshape(-1)
    if signal.size == 0:
        return signal
    frame_size = max(1, int(sample_rate * frame_ms / 1000))
    chunks = [signal[i:i + frame_size] for i in range(0, signal.size, frame_size)]
    return select_speech_chunks(chunks, threshold, sample_rate, min_speech_s)
