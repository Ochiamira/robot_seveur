#!/usr/bin/env python3
"""Service HTTP local déclenchant le dialogue vocal à l'arrivée du robot."""

from __future__ import annotations

import json
import logging
import os
import re
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import urlparse

import requests

from robot_dialogue_loop import (
    prepare_dialogue_runtime,
    run_table,
    validate_audio_input,
)


LOGGER = logging.getLogger("nexor.dialogue_service")
ACTIVE_STATES = {"starting", "running", "returning_home"}
NAVIGATION_BRIDGE_URL = os.environ.get(
    "NEXOR_NAVIGATION_URL", "http://localhost:8090"
).rstrip("/")


class DialogueController:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._runtime_ready = threading.Event()
        self._prepare_thread = None
        self._state = {
            "state": "idle",
            "table_id": None,
            "canonical_id": None,
            "message": "service prêt",
            "started_at": None,
            "updated_at": time.time(),
        }

    def snapshot(self) -> dict:
        with self._lock:
            return dict(self._state)

    def _update(self, **changes) -> None:
        with self._lock:
            self._state.update(changes)
            self._state["updated_at"] = time.time()

    @staticmethod
    def _normalize_table(value) -> str:
        match = re.fullmatch(r"T?([1-4])", str(value).strip(), re.IGNORECASE)
        if not match:
            raise ValueError("table_id doit être compris entre 1 et 4")
        return f"T{match.group(1)}"

    def prepare(self, payload: dict) -> tuple[bool, str, int]:
        """Précharge les poids sans démarrer l'écoute ni le dialogue."""
        raw_table = payload.get("table_id")
        table_id = None
        if raw_table is not None:
            try:
                table_id = self._normalize_table(raw_table)
            except (TypeError, ValueError) as exc:
                return False, str(exc), 400
        raw_canonical_id = payload.get("canonical_id")
        canonical_id = None
        if raw_canonical_id is not None:
            try:
                canonical_id = int(raw_canonical_id)
            except (TypeError, ValueError):
                return False, "canonical_id doit être un entier", 400

        try:
            audio_input = validate_audio_input()
        except Exception as exc:
            LOGGER.exception("Microphone indisponible")
            return False, f"microphone indisponible: {exc}", 503

        with self._lock:
            if self._state["state"] in ACTIVE_STATES:
                return True, "dialogue déjà actif", 202
            if self._runtime_ready.is_set():
                self._state.update({
                    "state": "ready",
                    "table_id": table_id or self._state.get("table_id"),
                    "canonical_id": (
                        canonical_id if canonical_id is not None
                        else self._state.get("canonical_id")
                    ),
                    "message": f"poids prêts; entrée={audio_input}",
                    "updated_at": time.time(),
                })
                return True, "communication déjà préchargée", 200
            if self._prepare_thread is not None and self._prepare_thread.is_alive():
                if table_id is not None:
                    self._state["table_id"] = table_id
                    self._state["canonical_id"] = canonical_id
                    self._state["updated_at"] = time.time()
                return True, "préchargement déjà en cours", 202

            self._state.update({
                "state": "preparing",
                "table_id": table_id,
                "canonical_id": canonical_id,
                "message": f"chargement Whisper/TTS; entrée={audio_input}",
                "started_at": time.time(),
                "updated_at": time.time(),
            })
            thread = threading.Thread(
                target=self._prepare_runtime,
                daemon=True,
                name="nexor-dialogue-preload",
            )
            self._prepare_thread = thread

        thread.start()
        return True, "préchargement mis en file", 202

    def _prepare_runtime(self) -> None:
        LOGGER.info("Préchargement de Whisper et TTS")
        try:
            prepare_dialogue_runtime()
        except Exception as exc:
            LOGGER.exception("Préchargement de la communication en échec")
            self._update(state="error", message=f"préchargement impossible: {exc}")
            return
        self._runtime_ready.set()
        with self._lock:
            if self._state["state"] == "preparing":
                self._state.update({
                    "state": "ready",
                    "message": "Whisper et TTS préchargés",
                    "updated_at": time.time(),
                })
        LOGGER.info("Communication préchargée et prête")

    def start(self, payload: dict) -> tuple[bool, str, int]:
        try:
            table_id = self._normalize_table(payload.get("table_id"))
        except (TypeError, ValueError) as exc:
            return False, str(exc), 400

        try:
            canonical_id = int(payload.get("canonical_id"))
        except (TypeError, ValueError):
            return False, "canonical_id doit être un entier", 400
        try:
            audio_input = validate_audio_input()
        except Exception as exc:
            LOGGER.exception("Microphone indisponible")
            return False, f"microphone indisponible: {exc}", 503

        with self._lock:
            if self._state["state"] in ACTIVE_STATES:
                if self._state["table_id"] == table_id:
                    return True, "dialogue déjà actif pour cette table", 202
                return False, "un autre dialogue est déjà actif", 409
            self._state.update({
                "state": "starting",
                "table_id": table_id,
                "canonical_id": canonical_id,
                "message": f"initialisation Whisper; entrée={audio_input}",
                "started_at": time.time(),
                "updated_at": time.time(),
            })

        thread = threading.Thread(
            target=self._run,
            args=(table_id, canonical_id),
            daemon=True,
            name=f"nexor-dialogue-{table_id}",
        )
        thread.start()
        return True, "dialogue mis en file", 202

    def _request_return_home(self, canonical_id: int, order_id: str | None) -> None:
        response = requests.post(
            f"{NAVIGATION_BRIDGE_URL}/return-home",
            json={"canonical_id": canonical_id, "order_id": order_id},
            timeout=2.0,
        )
        response.raise_for_status()

    def _run(self, table_id: str, canonical_id: int) -> None:
        LOGGER.info("Initialisation du dialogue pour %s", table_id)

        def mark_ready() -> None:
            self._update(state="running", message="microphone en écoute")
            LOGGER.info("Dialogue vocal prêt pour %s", table_id)

        try:
            result = run_table(
                table_id,
                transcription_language=os.environ.get("NEXOR_DIALOG_LANGUAGE", "auto"),
                on_ready=mark_ready,
            )
        except Exception as exc:
            LOGGER.exception("Dialogue en échec pour %s", table_id)
            self._update(state="error", message=str(exc))
            return
        if result.get("confirmed"):
            order_id = result.get("order_id")
            self._update(
                state="returning_home",
                message=f"commande {order_id or ''} confirmée; retour home demandé",
            )
            try:
                self._request_return_home(canonical_id, order_id)
            except Exception as exc:
                LOGGER.warning(
                    "Commande confirmée, mais retour home non transmis à Nav2: %s",
                    exc,
                )
                self._update(
                    state="finished",
                    message="commande confirmée; retour home non transmis",
                )
                return
            self._update(
                state="finished",
                message="commande confirmée; retour home mis en file",
            )
            LOGGER.info(
                "Commande %s confirmée pour %s; retour home mis en file",
                order_id,
                table_id,
            )
            return

        status = result.get("status", "conversation_ended")
        self._update(state="finished", message=f"dialogue terminé: {status}")
        LOGGER.info("Dialogue terminé pour %s (%s)", table_id, status)


class DialogueHttpHandler(BaseHTTPRequestHandler):
    controller: DialogueController

    def log_message(self, _format, *args) -> None:
        return

    def _reply(self, status: int, payload: dict) -> None:
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _json_body(self) -> dict | None:
        try:
            length = int(self.headers.get("Content-Length", "0"))
            value = json.loads(self.rfile.read(length) or b"{}")
            return value if isinstance(value, dict) else None
        except (ValueError, json.JSONDecodeError):
            return None

    def do_GET(self) -> None:
        path = urlparse(self.path).path
        if path in {"/health", "/status"}:
            self._reply(200, self.controller.snapshot())
        else:
            self._reply(404, {"error": "route inconnue"})

    def do_POST(self) -> None:
        path = urlparse(self.path).path
        if path not in {"/dialog/prepare", "/dialog/start"}:
            self._reply(404, {"error": "route inconnue"})
            return
        payload = self._json_body()
        if payload is None:
            self._reply(400, {"error": "JSON invalide"})
            return
        if path == "/dialog/prepare":
            accepted, message, status = self.controller.prepare(payload)
        else:
            accepted, message, status = self.controller.start(payload)
        self._reply(status, {
            "accepted": accepted,
            "message": message,
            **self.controller.snapshot(),
        })


def main() -> None:
    logging.basicConfig(
        level=os.environ.get("LOG_LEVEL", "INFO"),
        format="[%(asctime)s] [%(levelname)s] %(message)s",
    )
    host = os.environ.get("NEXOR_DIALOG_HOST", "127.0.0.1")
    port = int(os.environ.get("NEXOR_DIALOG_PORT", "8100"))
    controller = DialogueController()
    DialogueHttpHandler.controller = controller
    server = ThreadingHTTPServer((host, port), DialogueHttpHandler)
    LOGGER.info("Service de communication sur http://%s:%s", host, port)
    if os.environ.get("NEXOR_DIALOG_PRELOAD", "1").lower() in {"1", "true", "yes"}:
        accepted, message, _status = controller.prepare({})
        if accepted:
            LOGGER.info("%s", message)
        else:
            LOGGER.error("Préchargement non démarré: %s", message)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
