"""Livraison prioritaire, vérifiable et persistante vers l'application staff."""

from __future__ import annotations

import itertools
import json
import logging
import os
import queue
import threading
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path

import requests

from config import (
    STAFF_APP_RETRIES, STAFF_APP_TIMEOUT, STAFF_APP_TOKEN, STAFF_APP_URL,
    STAFF_OUTBOX_MAX_BYTES, STAFF_QUEUE_MAX,
)
from safe_file import append_jsonl, interprocess_lock

logger = logging.getLogger(__name__)
OUTBOX_PATH = Path(__file__).resolve().parent / "staff_queue" / "delivery_outbox.jsonl"
_CRITICAL = {"/api/events/order", "/api/events/payment_intent"}
_queue: queue.PriorityQueue = queue.PriorityQueue(maxsize=STAFF_QUEUE_MAX)
_sequence = itertools.count()
_worker_lock = threading.Lock()
_state_lock = threading.Lock()
_worker = None
_outbox_loaded = False
_flush_failures = []
_http = requests.Session()


@dataclass
class DeliveryReceipt:
    event_id: str
    path: str
    persisted: bool = False
    delivered: bool = False
    error: str | None = None
    attempts: int = 0
    done: threading.Event = field(default_factory=threading.Event, repr=False)

    def wait(self, timeout: float | None = None) -> bool:
        self.done.wait(timeout)
        return self.delivered


def _headers():
    return {"Authorization": f"Bearer {STAFF_APP_TOKEN}"} if STAFF_APP_TOKEN else {}


def _order_already_exists(payload: dict) -> bool:
    if not payload.get("order_id"):
        return False
    try:
        response = _http.get(
            f"{STAFF_APP_URL}/api/orders/{payload['order_id']}/timeline",
            headers=_headers(), timeout=STAFF_APP_TIMEOUT,
        )
        return response.status_code == 200
    except requests.RequestException:
        return False


def _post(path: str, payload: dict, critical: bool = True):
    attempts = STAFF_APP_RETRIES + 1 if critical else 1
    error = None
    for attempt in range(1, attempts + 1):
        if attempt > 1 and path == "/api/events/order" and _order_already_exists(payload):
            return True, None, attempt - 1
        try:
            response = _http.post(
                f"{STAFF_APP_URL}{path}", json=payload, headers=_headers(),
                timeout=STAFF_APP_TIMEOUT,
            )
            response.raise_for_status()
            body = response.json() if response.content else {"ok": True}
            if isinstance(body, dict) and body.get("ok") is False:
                raise requests.RequestException(f"Réponse négative: {body}")
            return True, None, attempt
        except (requests.RequestException, ValueError, TypeError) as exc:
            error = str(exc)
            if attempt < attempts:
                time.sleep(0.15 * (2 ** (attempt - 1)))
    return False, error, attempts


def _persist(receipt: DeliveryReceipt, payload: dict, status: str) -> bool:
    try:
        record = {"event_id": receipt.event_id, "path": receipt.path,
                  "status": status, "timestamp": time.time()}
        if status == "pending":
            record["payload"] = payload
        if receipt.error:
            record["error"] = receipt.error
        append_jsonl(OUTBOX_PATH, record)
        _compact_outbox()
        return True
    except Exception as exc:
        logger.error("Persistance outbox impossible: %s", exc)
        return False


def _compact_outbox() -> None:
    """Réduit l'outbox aux événements encore en attente, sans perdre un append concurrent."""
    try:
        if not OUTBOX_PATH.exists() or OUTBOX_PATH.stat().st_size < STAFF_OUTBOX_MAX_BYTES:
            return
    except OSError:
        return

    lock_path = OUTBOX_PATH.with_suffix(OUTBOX_PATH.suffix + ".lock")
    temp_path = OUTBOX_PATH.with_name(f".{OUTBOX_PATH.name}.{uuid.uuid4().hex}.tmp")
    try:
        with interprocess_lock(lock_path):
            if not OUTBOX_PATH.exists() or OUTBOX_PATH.stat().st_size < STAFF_OUTBOX_MAX_BYTES:
                return
            latest = {}
            with open(OUTBOX_PATH, encoding="utf-8") as handle:
                for line in handle:
                    try:
                        record = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    event_id = record.get("event_id")
                    if event_id:
                        latest[event_id] = record

            with open(temp_path, "x", encoding="utf-8") as handle:
                for record in latest.values():
                    if record.get("status") == "pending" and "payload" in record:
                        handle.write(json.dumps(
                            record, ensure_ascii=False, separators=(",", ":")
                        ) + "\n")
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temp_path, OUTBOX_PATH)
            try:
                os.chmod(OUTBOX_PATH, 0o600)
            except OSError:
                pass
    except OSError as exc:
        logger.warning("Compactage outbox impossible: %s", exc)
    finally:
        try:
            temp_path.unlink(missing_ok=True)
        except OSError:
            pass


def _load_pending():
    if not OUTBOX_PATH.exists():
        return []
    latest = {}
    try:
        with interprocess_lock(OUTBOX_PATH.with_suffix(OUTBOX_PATH.suffix + ".lock")):
            with open(OUTBOX_PATH, encoding="utf-8") as handle:
                for line in handle:
                    try:
                        record = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    if record.get("event_id"):
                        latest[record["event_id"]] = record
    except OSError as exc:
        logger.error("Lecture outbox impossible: %s", exc)
        return []
    return [(eid, r["path"], r["payload"]) for eid, r in latest.items()
            if r.get("status") == "pending" and "payload" in r]


def _put(receipt, payload, critical):
    item = (0 if critical else 10, next(_sequence), receipt, payload, critical)
    try:
        _queue.put(item, timeout=0.5) if critical else _queue.put_nowait(item)
        return True
    except queue.Full:
        receipt.error = "file de livraison saturée"
        receipt.done.set()
        logger.error("Événement staff rejeté: %s", receipt.path)
        return False


def _worker_loop():
    while True:
        _, _, receipt, payload, critical = _queue.get()
        try:
            receipt.delivered, receipt.error, receipt.attempts = _post(
                receipt.path, payload, critical
            )
            if critical:
                _persist(receipt, payload, "delivered" if receipt.delivered else "pending")
            if not receipt.delivered:
                with _state_lock:
                    _flush_failures.append(receipt.event_id)
                logger.error("Échec livraison staff %s: %s", receipt.path, receipt.error)
        except Exception as exc:
            receipt.error = str(exc)
            with _state_lock:
                _flush_failures.append(receipt.event_id)
            logger.exception("Erreur du worker staff")
        finally:
            receipt.done.set()
            _queue.task_done()


def _ensure_worker():
    global _worker, _outbox_loaded
    with _worker_lock:
        if _worker is None or not _worker.is_alive():
            _worker = threading.Thread(target=_worker_loop, name="staff-events", daemon=True)
            _worker.start()
        if not _outbox_loaded:
            _outbox_loaded = True
            for event_id, path, payload in _load_pending():
                _put(DeliveryReceipt(event_id, path, persisted=True), payload, True)


def _enqueue(path: str, payload: dict):
    critical = path in _CRITICAL
    receipt = DeliveryReceipt(str(uuid.uuid4()), path)
    _ensure_worker()
    if critical:
        receipt.persisted = _persist(receipt, payload, "pending")
    _put(receipt, payload, critical)
    return receipt


def flush(timeout=5.0):
    deadline = time.monotonic() + timeout
    while _queue.unfinished_tasks and time.monotonic() < deadline:
        time.sleep(0.02)
    with _state_lock:
        failed = bool(_flush_failures)
        _flush_failures.clear()
    return _queue.unfinished_tasks == 0 and not failed


def notify_dialog_state(state, text="", lang="fr"):
    return _enqueue("/api/events/dialog", {"state": state, "text": text, "lang": lang})


def notify_draft_order(items, total=None):
    return _enqueue("/api/events/draft_order", {
        "items": items, "total": round(total, 3) if total is not None else None,
    })


def notify_order(order_id, table, items, total, devise="TND", lang="fr"):
    return _enqueue("/api/events/order", {
        "event": "confirmed", "order_id": order_id, "table": table,
        "items": items, "total": round(total, 3), "devise": devise, "lang": lang,
    })


def notify_payment(table, payment_method, items, total, devise="TND", lang="fr",
                   session_id=None, order_id=None):
    correlation = session_id
    if order_id:
        correlation = f"{session_id or 'session'}|order_id={order_id}"
    return _enqueue("/api/events/payment_intent", {
        "table": table, "payment_method": payment_method, "items": items,
        "total": round(total, 3), "devise": devise, "lang": lang,
        # Le backend actuel ne possède pas encore de champ order_id sur cette
        # route; on conserve donc aussi la corrélation dans session_id.
        "session_id": correlation, "order_id": order_id,
    })
