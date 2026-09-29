"""
staff_app_client.py
====================
Pousse les événements commande/paiement vers le backend FastAPI de l'app
staff (main.py), qui les diffuse en temps réel au dashboard via WebSocket.

Contrat de robustesse, NON NÉGOCIABLE : un problème réseau ou un dashboard
éteint ne doit JAMAIS ralentir ni faire planter le dialogue avec le client.
D'où : timeout court (1s) + appel dans un thread daemon séparé + aucune
exception ne remonte jamais à l'appelant (dialog_manager).

Complémentaire à staff_notifier.py (journal JSONL local, permanent, lu même
si le dashboard est éteint) plutôt qu'un remplacement : staff_notifier.py
sert d'audit trail fiable, ce module sert de notification temps réel.
"""

import logging
import queue
import threading
import time

import requests

from config import STAFF_QUEUE_MAX

logger = logging.getLogger(__name__)

# Suivi des threads de push en cours, pour pouvoir les "flush" (attendre leur
# fin) à des points de rupture naturels (fin de session, confirmation de
# commande, arrêt du script). Sans ça, un thread daemon lancé juste avant la
# fin du process peut être tué avant d'avoir terminé son POST : la commande
# a bien été confirmée au client vocalement, mais n'arrive jamais au
# dashboard — perte silencieuse, observée en test sur le dernier tour d'une
# conversation (voir test_full_pipeline.py).
_event_queue: queue.Queue = queue.Queue(maxsize=STAFF_QUEUE_MAX)
_worker_lock = threading.Lock()
_worker: threading.Thread | None = None


def _post(path: str, payload: dict) -> bool:
    """Compatibilité privée : délègue à la livraison robuste unique."""
    from staff_delivery import _post as reliable_post
    delivered, _, _ = reliable_post(path, payload, critical=True)
    return delivered


def _worker_loop() -> None:
    while True:
        path, payload = _event_queue.get()
        try:
            _post(path, payload)
        finally:
            _event_queue.task_done()


def _ensure_worker() -> None:
    global _worker
    with _worker_lock:
        if _worker is None or not _worker.is_alive():
            _worker = threading.Thread(target=_worker_loop, name="staff-events", daemon=True)
            _worker.start()


def _post_async(path: str, payload: dict) -> None:
    """Compatibilité privée : utilise aussi la file persistante/prioritaire."""
    from staff_delivery import _enqueue
    _enqueue(path, payload)


def flush(timeout: float = 5.0) -> bool:
    """
    Attend que tous les envois en cours se terminent (au plus `timeout`
    secondes au total). À appeler aux points de rupture naturels où on ne
    veut PAS perdre silencieusement le dernier événement :
      - fin de session / reset du dialogue (conversation terminée)
      - juste avant l'arrêt propre du script/robot

    Ne bloque JAMAIS le dialogue en cours : à n'appeler qu'une fois la
    réponse déjà donnée au client (ex: après dm.process() si dm.state.finished).
    """
    deadline = time.monotonic() + timeout
    while _event_queue.unfinished_tasks and time.monotonic() < deadline:
        time.sleep(0.02)
    complete = _event_queue.unfinished_tasks == 0
    if not complete:
        logger.error("Timeout: %s événement(s) staff non livré(s)", _event_queue.unfinished_tasks)
    return complete


def notify_dialog_state(state: str, text: str = "", lang: str = "fr") -> None:
    """
    Pousse un changement d'état de conversation vers /api/events/dialog,
    affiché en direct sur l'écran client (orbe animé + transcription).
    state ∈ {"idle", "listening", "processing", "speaking"} — voir
    DialogEvent dans main.py pour le contrat exact.
    """
    _post_async("/api/events/dialog", {
        "state": state,
        "text":  text,
        "lang":  lang,
    })


def notify_draft_order(items: list, total: float = None) -> None:
    """
    Pousse le panier EN COURS (avant confirmation) vers /api/events/draft_order,
    affiché en direct dans le panneau latéral de l'écran client. Purement
    d'affichage — jamais persisté côté backend (voir DraftOrderEvent).
    """
    _post_async("/api/events/draft_order", {
        "items": items,
        "total": round(total, 3) if total is not None else None,
    })


def notify_order(
    order_id: str,
    table:    str,
    items:    list,           # ["2× Couscous agneau", ...]
    total:    float,
    devise:   str = "TND",
    lang:     str = "fr",
) -> None:
    """
    Pousse une commande CONFIRMÉE vers /api/events/order (voir main.py).
    Appelé une seule fois par commande (à la confirmation) — main.py ne
    connaît que l'événement "confirmed" ; le suivi ensuite (en préparation,
    prête, servie...) est piloté par le staff depuis le dashboard, pas par
    le pipeline NLP.
    """
    _post_async("/api/events/order", {
        "event":    "confirmed",
        "order_id": order_id,
        "table":    table,
        "items":    items,
        "total":    round(total, 3) if total is not None else None,
        "devise":   devise,
        "lang":     lang,
    })


def notify_payment(
    table:          str,
    payment_method: str,      # "especes" | "carte"
    items:          list,
    total:          float,
    devise:         str = "TND",
    lang:           str = "fr",
    session_id:     str = None,
) -> None:
    """
    Signale une INTENTION de paiement (pas un encaissement réel) vers
    /api/events/payment_intent. Le robot n'a aucun moyen de savoir si la
    commande est déjà "servie" (condition exigée par db.record_payment côté
    backend) — cette alerte informe juste le staff du souhait du client,
    l'encaissement réel reste fait à la main via le dashboard.
    """
    _post_async("/api/events/payment_intent", {
        "table":          table,
        "payment_method": payment_method,
        "items":          items,
        "total":          round(total, 3),
        "devise":         devise,
        "lang":           lang,
        "session_id":     session_id,
    })


# API publique remplacée par l'implémentation persistante/prioritaire. Ces
# imports en fin de module conservent la compatibilité des anciens appelants.
from staff_delivery import (  # noqa: E402,F401
    DeliveryReceipt,
    flush,
    notify_dialog_state,
    notify_draft_order,
    notify_order,
    notify_payment,
)
