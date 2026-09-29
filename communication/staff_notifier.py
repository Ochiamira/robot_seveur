"""
staff_notifier.py
==================
Transmet les demandes qui nécessitent une intervention humaine (paiement,
pour l'instant) au personnel, via un fichier file d'attente JSONL.

Format volontairement simple et découplé : n'importe quelle app externe
(écran cuisine, app staff mobile, script de notification Slack/Telegram...)
peut lire ce fichier en le "tailant" (tail -f) ou en le pollant, sans
dépendance directe avec le pipeline NLP. Une ligne = un événement JSON.

⚠️ Le robot NEXOR n'effectue AUCUNE transaction réelle : ce module ne fait
qu'informer un humain qu'une action est attendue de sa part.
"""

import json
import logging
import os
import uuid
from datetime import datetime
from pathlib import Path

from config import STAFF_AUDIT_MAX_BYTES
from safe_file import append_jsonl

logger = logging.getLogger(__name__)

QUEUE_DIR  = Path(__file__).parent / "staff_queue"
QUEUE_PATH = QUEUE_DIR / "queue.jsonl"


def _append_event(event: dict) -> bool:
    """Ajoute une ligne JSON au fichier file d'attente. Ne lève jamais
    d'exception vers l'appelant : une notification staff ratée ne doit
    JAMAIS faire planter ou bloquer une réponse déjà donnée au client."""
    try:
        append_jsonl(QUEUE_PATH, event, max_bytes=STAFF_AUDIT_MAX_BYTES)
        return True
    except Exception as e:
        logger.warning(f"⚠️  Notification staff échouée : {e}")
        return False


def notify_payment(
    session_id:     str,
    payment_method: str,   # "cash" | "card"
    order_items:    list,  # ["1× Couscous agneau = 18.000 TND", ...]
    total:          float,
    devise:         str,
    lang:           str = "fr",
    table:          str = None,
    order_id:       str = None,
) -> dict:
    """
    Enregistre une demande de paiement dans la file d'attente staff.
    Retourne l'événement créé (utile pour logs/tests), même en cas
    d'échec d'écriture (le champ "queued" indique le succès réel).
    """
    event = {
        "event_id":       str(uuid.uuid4()),
        "type":           "payment_request",
        "timestamp":      datetime.now().isoformat(),
        "session_id":     session_id,
        "table":          table,
        "order_id":       order_id,
        "payment_method": payment_method,
        "order_items":    order_items,
        "total":          round(total, 3),
        "devise":         devise,
        "lang":           lang,
        "status":         "pending",  # à mettre à jour côté app staff ("done")
    }
    event["queued"] = _append_event(event)
    if event["queued"]:
        logger.info(f"📋 Demande de paiement enregistrée localement : {payment_method} — {total:.3f} {devise}")
    return event


def notify_order(
    order_id: str,
    table: str,
    order_items: list,
    total: float,
    devise: str,
    lang: str = "fr",
) -> dict:
    """Persiste aussi les commandes confirmées avant l'envoi HTTP."""
    event = {
        "event_id": str(uuid.uuid4()),
        "type": "confirmed_order",
        "timestamp": datetime.now().isoformat(),
        "order_id": order_id,
        "table": table,
        "order_items": order_items,
        "total": round(total, 3),
        "devise": devise,
        "lang": lang,
        "status": "pending",
    }
    event["queued"] = _append_event(event)
    return event


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    e = notify_payment(
        session_id="test_001",
        payment_method="cash",
        order_items=["1× Couscous agneau = 18.000 TND"],
        total=18.0,
        devise="TND",
        lang="fr",
    )
    print(json.dumps(e, indent=2, ensure_ascii=False))
    print(f"→ Fichier file d'attente : {QUEUE_PATH}")
