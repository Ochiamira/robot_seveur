"""
session_logger.py
=================
Journalise chaque tour de dialogue en JSON.
Utile pour déboguer, auditer et améliorer le pipeline.
Chaque session = un fichier JSON horodaté.
"""

import json
import logging
import os
import re
import threading
import uuid
from datetime import datetime
from pathlib import Path

from config import LOG_SENSITIVE_DATA, SESSION_LOG_DIR

LOG_DIR = SESSION_LOG_DIR
logger  = logging.getLogger(__name__)
_write_lock = threading.Lock()


class SessionLogger:
    def __init__(self, session_id: str = None):
        raw_id = session_id or f"{datetime.now():%Y%m%d_%H%M%S_%f}_{uuid.uuid4().hex[:8]}"
        self.session_id = re.sub(r"[^A-Za-z0-9_.-]", "_", str(raw_id))[:120]
        self.turns: list[dict] = []
        self.start_time = datetime.now().isoformat()
        LOG_DIR.mkdir(parents=True, exist_ok=True)
        self.log_path = LOG_DIR / f"session_{self.session_id}.json"

    def log_turn(
        self,
        turn:       int,
        stt_text:   str,
        lang:       str,
        intent:     str,
        score:      float,
        entities:   list,
        response:   str,
        order:      list,
        duration_ms: float = 0.0,
    ) -> None:
        """Journalise un tour de dialogue."""
        entry = {
            "turn":        turn,
            "timestamp":   datetime.now().isoformat(),
            "stt_text":    stt_text if LOG_SENSITIVE_DATA else "[redacted]",
            "lang":        lang,
            "intent":      intent,
            "score":       round(score, 3),
            "entities":    entities if LOG_SENSITIVE_DATA else [],
            "response":    response if LOG_SENSITIVE_DATA else "[redacted]",
            "order_state": order if LOG_SENSITIVE_DATA else [],
            "duration_ms": round(duration_ms, 1),
        }
        self.turns.append(entry)
        self._flush()
        logger.debug(f"Tour {turn} journalisé → {self.log_path}")

    def log_error(self, error: str, context: dict = None) -> None:
        """Journalise une erreur."""
        entry = {
            "type":      "error",
            "timestamp": datetime.now().isoformat(),
            "error":     error if LOG_SENSITIVE_DATA else "[redacted error]",
            "context":   (context or {}) if LOG_SENSITIVE_DATA else {},
        }
        self.turns.append(entry)
        self._flush()

    def _flush(self) -> None:
        """Écrit le fichier JSON à chaque tour (pas de perte si crash)."""
        data = {
            "session_id":  self.session_id,
            "start_time":  self.start_time,
            "turns":       self.turns,
        }
        try:
            temp_path = self.log_path.with_suffix(f".{uuid.uuid4().hex}.tmp")
            with _write_lock:
                with open(temp_path, "w", encoding="utf-8") as f:
                    json.dump(data, f, indent=2, ensure_ascii=False)
                    f.flush()
                    os.fsync(f.fileno())
                os.replace(temp_path, self.log_path)
                try:
                    os.chmod(self.log_path, 0o600)
                except OSError:
                    pass
        except Exception as e:
            try:
                temp_path.unlink(missing_ok=True)
            except Exception:
                pass
            logger.warning(f"Impossible d'écrire le log : {e}")

    def summary(self) -> dict:
        """Retourne un résumé de la session."""
        intents = [t.get("intent") for t in self.turns if "intent" in t]
        return {
            "session_id":  self.session_id,
            "n_turns":     len(self.turns),
            "intents":     intents,
            "log_path":    str(self.log_path),
        }


if __name__ == "__main__":
    sl = SessionLogger("test_001")
    sl.log_turn(
        turn=1, stt_text="je voudrais un café",
        lang="fr", intent="commander", score=0.8,
        entities=["Café espresso"], response="J'ai ajouté 1× Café espresso.",
        order=["Café espresso × 1 = 1.500 TND"],
    )
    sl.log_turn(
        turn=2, stt_text="c'est tout",
        lang="fr", intent="confirmer_commande", score=1.0,
        entities=[], response="Commande confirmée !",
        order=["Café espresso × 1 = 1.500 TND"],
    )
    print(f"✅ Session loggée : {sl.log_path}")
    print(json.dumps(sl.summary(), indent=2, ensure_ascii=False))
