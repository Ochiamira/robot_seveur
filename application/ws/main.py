"""
main.py
=======
Serveur local NEXOR Staff App + Écran client.

Rôle : pont entre le robot (dialog_manager / ROS2 / vision_bridge) et deux
interfaces :
  - le dashboard staff (cuisine/salle)
  - l'écran client monté sur le robot (transcription, panier live, menu, QR)

Tourne sur le Raspberry Pi 5 (ou un PC du resto), accessible en WiFi local,
aucune dépendance internet.

Architecture :
    dialog_manager.py   ──POST /api/events/order────────┐
    dialog_manager.py   ──POST /api/events/dialog───────┤
    dialog_manager.py   ──POST /api/events/draft_order──┤
    dialog_manager.py   ──POST /api/events/payment_intent┤──> StateStore ──WS /ws────────> dashboard staff
    vision_bridge.py    ──POST /api/events/table_status─┤                ──WS /ws/client─> écran client
    noeud ROS2 (statut) ──POST /api/events/robot_status─┤
    noeud ROS2 (pose)   ──POST /api/events/robot_position┘

Lancer :
    pip install -r requirements.txt --break-system-packages
    python -m uvicorn main:app --host 0.0.0.0 --port 8000 --workers 1

    Dashboard staff -> http://<IP>:8000/
    Écran client    -> http://<IP>:8000/client/
"""

import asyncio
import html
import ipaddress
import io
import json
import os
import re
import secrets
import socket
import time
from decimal import Decimal
from pathlib import Path
from typing import Annotated, Literal
from urllib.parse import quote, urlparse
from urllib.request import Request as UrlRequest, urlopen

import qrcode
from fastapi import Depends, FastAPI, HTTPException, Query, Request, WebSocket, WebSocketDisconnect
from fastapi.staticfiles import StaticFiles
from fastapi.responses import StreamingResponse, HTMLResponse
from pydantic import BaseModel, ConfigDict, Field, model_validator

import db

BASE_DIR = Path(__file__).parent
FRONTEND_DIR = BASE_DIR.parent / "ui"
CLIENT_DIR = BASE_DIR.parent / "client_screen"
IMAGES_DIR = BASE_DIR / "static" / "menu_images"
MENU_PATH = BASE_DIR / "menu.json"
STAFF_APP_TOKEN = os.getenv("STAFF_APP_TOKEN", "").strip()
NAVIGATION_BRIDGE_URL = os.getenv(
    "NEXOR_NAVIGATION_URL", "http://localhost:8090"
).rstrip("/")
ALLOW_INSECURE_REMOTE = os.getenv("NEXOR_ALLOW_INSECURE_REMOTE", "0").lower() in {
    "1", "true", "yes", "on"
}

db.init_db()

app = FastAPI(title="NEXOR Staff App")


@app.middleware("http")
async def security_headers(request: Request, call_next):
    response = await call_next(request)
    response.headers["X-Content-Type-Options"] = "nosniff"
    response.headers["X-Frame-Options"] = "DENY"
    response.headers["Referrer-Policy"] = "no-referrer"
    response.headers["Content-Security-Policy"] = (
        "default-src 'self'; script-src 'self'; style-src 'self' 'unsafe-inline'; "
        "img-src 'self' data:; connect-src 'self' ws: wss:; object-src 'none'; "
        "base-uri 'none'; frame-ancestors 'none'"
    )
    return response


def _is_loopback(host: str | None) -> bool:
    if not host:
        return False
    try:
        return ipaddress.ip_address(host).is_loopback
    except ValueError:
        return host.lower() == "localhost"


def _valid_bearer(value: str | None) -> bool:
    if not STAFF_APP_TOKEN or not value or not value.startswith("Bearer "):
        return False
    return secrets.compare_digest(value[7:], STAFF_APP_TOKEN)


async def require_api_access(request: Request) -> None:
    """Autorise le local; exige le secret partagé pour tout accès distant."""
    if _is_loopback(request.client.host if request.client else None):
        return
    if STAFF_APP_TOKEN:
        if _valid_bearer(request.headers.get("authorization")):
            return
        raise HTTPException(status_code=401, detail="Jeton STAFF_APP_TOKEN invalide.")
    if ALLOW_INSECURE_REMOTE:
        return
    raise HTTPException(
        status_code=403,
        detail="Accès distant désactivé. Configurez STAFF_APP_TOKEN des deux côtés.",
    )


def _same_origin(websocket: WebSocket) -> bool:
    origin = websocket.headers.get("origin")
    if not origin:
        return True
    parsed = urlparse(origin)
    return parsed.netloc == websocket.headers.get("host")


def _websocket_authorized(websocket: WebSocket) -> bool:
    if not _same_origin(websocket):
        return False
    host = websocket.client.host if websocket.client else None
    if _is_loopback(host):
        return True
    if STAFF_APP_TOKEN:
        supplied = websocket.query_params.get("token")
        return bool(supplied and secrets.compare_digest(supplied, STAFF_APP_TOKEN))
    return ALLOW_INSECURE_REMOTE


PROTECTED = [Depends(require_api_access)]


# ═══════════════════════════════════════════════════════════════════════════
# Modèles
# ═══════════════════════════════════════════════════════════════════════════

class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)


ShortText = Annotated[str, Field(min_length=1, max_length=200)]


class OrderEvent(StrictModel):
    event: Literal["confirmed"]   # seul événement de création — le workflow ensuite passe par /api/orders/{id}/advance
    order_id: str | None = Field(default=None, min_length=1, max_length=64, pattern=r"^[A-Za-z0-9_-]+$")
    table: str = Field(min_length=1, max_length=64)
    items: list[ShortText] = Field(min_length=1, max_length=100)
    total: float = Field(gt=0, allow_inf_nan=False)
    devise: str = Field(default="TND", min_length=3, max_length=8, pattern=r"^[A-Z]+$")
    lang: Literal["fr", "en", "ar"] = "fr"


class AdvanceStatus(StrictModel):
    to_status: Literal["en_preparation", "prete", "servie", "annulee"]


class PaymentRequest(StrictModel):
    method: Literal["especes", "carte"]
    amount_paid: float | None = Field(default=None, ge=0, allow_inf_nan=False)


class PaymentIntentEvent(StrictModel):
    """
    ⚠️ Ceci n'est PAS un encaissement réel — juste le souhait exprimé par le
    client au robot ("je paie en espèces"). L'encaissement réel passe
    exclusivement par POST /api/orders/{id}/pay, déclenché par le staff
    depuis le dashboard une fois le plat "servie", avec le montant réel reçu.
    """
    table: str = Field(min_length=1, max_length=64)
    payment_method: Literal["especes", "carte"]
    items: list[ShortText] = Field(min_length=1, max_length=100)
    total: float = Field(gt=0, allow_inf_nan=False)
    devise: str = Field(default="TND", min_length=3, max_length=8, pattern=r"^[A-Z]+$")
    lang: Literal["fr", "en", "ar"] = "fr"
    session_id: str | None = Field(default=None, max_length=256)
    order_id: str | None = Field(default=None, min_length=1, max_length=64, pattern=r"^[A-Za-z0-9_-]+$")


class TableStatusEvent(StrictModel):
    """
    Poussé par vision_bridge.py (via Orchestrator.on_action), PAS par le
    robot lui-même — c'est la caméra/FSM qui sait qui est assis où.
    "attente_longue" déclenche EN PLUS une alerte staff (voir endpoint).
    """
    table: str = Field(min_length=1, max_length=64)
    status: Literal["libre", "occupee", "attente_robot", "en_service", "attente_longue"]
    message: str = Field(default="", max_length=500)
    waiting_s: float | None = Field(default=None, ge=0, allow_inf_nan=False)


class RobotStatusEvent(StrictModel):
    status: Literal["ok", "en_route", "attente", "bloque", "aide_demandee", "hors_ligne"]
    message: str = Field(default="", max_length=500)
    table: str | None = Field(default=None, min_length=1, max_length=64)


class RobotPositionEvent(StrictModel):
    x: float = Field(allow_inf_nan=False)
    y: float = Field(allow_inf_nan=False)
    theta: float = Field(default=0.0, allow_inf_nan=False)


class StaffAction(StrictModel):
    action: Literal["acquitter_alerte"]
    order_id: str = Field(min_length=1, max_length=64, pattern=r"^[A-Za-z0-9_-]+$")


class DialogEvent(StrictModel):
    """Poussé par dialog_manager.py à chaque changement d'état de la conversation (écran client)."""
    state: Literal["idle", "listening", "processing", "speaking"]
    text: str = Field(default="", max_length=2000)
    lang: Literal["fr", "en", "ar"] = "fr"


def _dispatch_ready_order(order: dict) -> dict:
    """Demande a Nav2 de livrer une commande devenue prete."""
    table_name = str(order.get("table_name", "")).strip()
    match = re.fullmatch(r"T?([1-4])", table_name, re.IGNORECASE)
    if not match:
        raise ValueError(f"Table de livraison invalide: {table_name or 'absente'}")

    payload = json.dumps({
        "table_id": int(match.group(1)),
        "order_id": order["order_id"],
    }).encode("utf-8")
    request = UrlRequest(
        f"{NAVIGATION_BRIDGE_URL}/deliver-order",
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urlopen(request, timeout=2.0) as response:
        result = json.loads(response.read().decode("utf-8"))
    if not result.get("accepted"):
        raise RuntimeError(result.get("message", "livraison refusee"))
    return result


class DraftOrderEvent(StrictModel):
    """Panier en cours de construction, avant confirmation — purement d'affichage, non persisté."""
    items: list[ShortText] = Field(default_factory=list, max_length=100)
    total: float | None = Field(default=None, ge=0, allow_inf_nan=False)

    @model_validator(mode="after")
    def validate_total(self):
        if self.items and (self.total is None or self.total <= 0):
            raise ValueError("Un panier non vide doit avoir un total positif.")
        return self


def _normalized_name(value: str) -> str:
    return " ".join(value.casefold().split())


def _validated_order_total(evt: OrderEvent) -> float:
    """Recalcule le total depuis le menu canonique partagé avec communication."""
    menu = _load_menu()
    expected_currency = menu.get("restaurant", {}).get("devise", "TND")
    if evt.devise != expected_currency:
        raise HTTPException(status_code=422, detail="Devise différente de celle du menu.")

    prices: dict[str, Decimal] = {}
    for menu_item in menu.get("items", []):
        if not menu_item.get("disponible", True):
            continue
        price = Decimal(str(menu_item["prix"]))
        names = menu_item.get("nom", {})
        if isinstance(names, str):
            names = {"fr": names}
        for name in names.values():
            if name:
                prices[_normalized_name(str(name))] = price

    total = Decimal("0")
    total_units = 0
    for label in evt.items:
        match = re.fullmatch(r"\s*([1-9]\d{0,2})\s*[xX×]\s*(.+?)\s*", label)
        if not match:
            raise HTTPException(status_code=422, detail=f"Article mal formaté : {label}")
        quantity = int(match.group(1))
        raw_name = match.group(2)
        key = _normalized_name(raw_name)
        if key not in prices:
            # communication ajoute éventuellement taille/modificateurs en suffixe.
            raw_name = re.sub(r"\s+\([^()]{1,120}\)\s*$", "", raw_name)
            key = _normalized_name(raw_name)
        if key not in prices:
            raise HTTPException(status_code=422, detail=f"Article absent du menu : {raw_name}")
        total_units += quantity
        if total_units > 100:
            raise HTTPException(status_code=422, detail="Commande limitée à 100 articles.")
        total += prices[key] * quantity

    total = total.quantize(Decimal("0.001"))
    supplied = Decimal(str(evt.total)).quantize(Decimal("0.001"))
    if total != supplied:
        raise HTTPException(
            status_code=422,
            detail=f"Total incorrect : attendu {total:.3f} {expected_currency}.",
        )
    return float(total)


def _payment_order_id(evt: PaymentIntentEvent) -> str | None:
    if evt.order_id:
        return evt.order_id
    if evt.session_id:
        match = re.search(r"(?:^|\|)order_id=([A-Za-z0-9_-]{1,64})(?:$|\|)", evt.session_id)
        if match:
            return match.group(1)
    return None


# ═══════════════════════════════════════════════════════════════════════════
# État en mémoire (partagé, un seul resto / un seul robot)
# ═══════════════════════════════════════════════════════════════════════════

class StateStore:
    def __init__(self):
        self._clients: set[WebSocket] = set()
        self._client_screens: set[WebSocket] = set()
        self._lock = asyncio.Lock()
        self._suppressed_draft: tuple[tuple[str, ...], int] | None = None

    async def snapshot(self) -> dict:
        defaults = {
            "robot_status": {"status": "hors_ligne", "message": "", "table": None, "ts": time.time()},
            "robot_position": {"x": 0.5, "y": 0.5, "theta": 0.0, "ts": time.time()},
            "tables": {},
        }
        orders, alerts, robot_status, robot_position, tables = await asyncio.gather(
            asyncio.to_thread(db.list_active_orders),
            asyncio.to_thread(db.list_alerts),
            asyncio.to_thread(db.load_state, "robot_status", defaults["robot_status"]),
            asyncio.to_thread(db.load_state, "robot_position", defaults["robot_position"]),
            asyncio.to_thread(db.load_state, "tables", defaults["tables"]),
        )
        return {
            "type": "snapshot",
            "orders": orders,
            "robot_status": robot_status,
            "robot_position": robot_position,
            "tables": tables,
            "alerts": alerts,
        }

    async def client_snapshot(self) -> dict:
        dialog, draft = await asyncio.gather(
            asyncio.to_thread(
                db.load_state,
                "dialog_state",
                {"state": "idle", "text": "", "lang": "fr", "ts": time.time()},
            ),
            asyncio.to_thread(
                db.load_state,
                "draft_order",
                {"items": [], "total": None, "ts": time.time()},
            ),
        )
        return {
            "type": "snapshot",
            "dialog_state": dialog,
            "draft_order": draft,
        }

    def remember_confirmed_draft(self, items: list[str], total: float) -> None:
        self._suppressed_draft = (tuple(items), round(total * 1000))

    def suppresses_draft(self, items: list[str], total: float | None) -> bool:
        if not items:
            self._suppressed_draft = None
            return False
        signature = (tuple(items), round((total or 0) * 1000))
        if signature == self._suppressed_draft:
            return True
        self._suppressed_draft = None
        return False

    async def register(self, ws: WebSocket):
        async with self._lock:
            self._clients.add(ws)

    async def unregister(self, ws: WebSocket):
        async with self._lock:
            self._clients.discard(ws)

    async def register_client_screen(self, ws: WebSocket):
        async with self._lock:
            self._client_screens.add(ws)

    async def unregister_client_screen(self, ws: WebSocket):
        async with self._lock:
            self._client_screens.discard(ws)

    async def broadcast(self, message: dict):
        async with self._lock:
            clients = list(self._clients)
        results = await asyncio.gather(
            *(asyncio.wait_for(ws.send_json(message), timeout=1.0) for ws in clients),
            return_exceptions=True,
        )
        dead = [ws for ws, result in zip(clients, results) if isinstance(result, Exception)]
        if dead:
            async with self._lock:
                self._clients.difference_update(dead)

    async def broadcast_client_screen(self, message: dict):
        async with self._lock:
            clients = list(self._client_screens)
        results = await asyncio.gather(
            *(asyncio.wait_for(ws.send_json(message), timeout=1.0) for ws in clients),
            return_exceptions=True,
        )
        dead = [ws for ws, result in zip(clients, results) if isinstance(result, Exception)]
        if dead:
            async with self._lock:
                self._client_screens.difference_update(dead)


store = StateStore()


# ═══════════════════════════════════════════════════════════════════════════
# WebSocket — le dashboard s'y connecte pour recevoir le flux temps réel
# ═══════════════════════════════════════════════════════════════════════════

@app.websocket("/ws")
async def ws_endpoint(websocket: WebSocket):
    if not _websocket_authorized(websocket):
        await websocket.close(code=4401)
        return
    await websocket.accept()
    await store.register(websocket)
    try:
        await websocket.send_json(await store.snapshot())
        while True:
            await websocket.receive_text()
    except (WebSocketDisconnect, RuntimeError):
        pass
    finally:
        await store.unregister(websocket)


@app.websocket("/ws/client")
async def ws_client_endpoint(websocket: WebSocket):
    """Canal dédié à l'écran client — ne reçoit ni les alertes ni les données des autres tables."""
    if not _websocket_authorized(websocket):
        await websocket.close(code=4401)
        return
    await websocket.accept()
    await store.register_client_screen(websocket)
    try:
        await websocket.send_json(await store.client_snapshot())
        while True:
            await websocket.receive_text()
    except (WebSocketDisconnect, RuntimeError):
        pass
    finally:
        await store.unregister_client_screen(websocket)


# ═══════════════════════════════════════════════════════════════════════════
# Ingestion — appelé par dialog_manager.py et par le noeud ROS2
# ═══════════════════════════════════════════════════════════════════════════

@app.post("/api/events/order", dependencies=PROTECTED)
async def post_order_event(evt: OrderEvent):
    """Crée une commande confirmée; un rejeu identique est sans effet."""
    canonical_total = await asyncio.to_thread(_validated_order_total, evt)
    try:
        order, created = await asyncio.to_thread(
            db.create_order,
            evt.table,
            evt.items,
            canonical_total,
            evt.devise,
            evt.lang,
            evt.order_id,
        )
    except db.OrderConflictError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc

    if created:
        await store.broadcast({"type": "order", "event": "created", "order": order})
        empty_draft = {"items": [], "total": None, "ts": time.time()}
        await asyncio.to_thread(db.save_state, "draft_order", empty_draft)
        store.remember_confirmed_draft(evt.items, canonical_total)
        await store.broadcast_client_screen({"type": "order_confirmed", "order": order})

    return {"ok": True, "order_id": order["order_id"], "created": created}


@app.post("/api/orders/{order_id}/advance", dependencies=PROTECTED)
async def advance_order(order_id: str, body: AdvanceStatus):
    """Fait avancer une commande dans le workflow (confirmee -> en_preparation -> prete -> servie).
    Pour l'encaissement (servie -> payee), utiliser /api/orders/{order_id}/pay."""
    ok, error, order = await asyncio.to_thread(db.advance_status, order_id, body.to_status)
    if not ok:
        raise HTTPException(status_code=400, detail=error)
    await store.broadcast({"type": "order", "event": "updated", "order": order})

    delivery = None
    if order["status"] == "prete":
        try:
            delivery = await asyncio.to_thread(_dispatch_ready_order, order)
        except Exception as exc:
            delivery = {"accepted": False, "message": str(exc)}
            robot_status = {
                "status": "bloque",
                "message": f"Livraison non transmise: {exc}",
                "table": order.get("table_name"),
                "ts": time.time(),
            }
        else:
            robot_status = {
                "status": "en_route",
                "message": f"Livraison de la commande {order_id}",
                "table": order.get("table_name"),
                "ts": time.time(),
            }
        await asyncio.to_thread(db.save_state, "robot_status", robot_status)
        await store.broadcast({"type": "robot_status", "robot_status": robot_status})

    return {"ok": True, "order": order, "delivery": delivery}


@app.post("/api/orders/{order_id}/pay", dependencies=PROTECTED)
async def pay_order(order_id: str, body: PaymentRequest):
    """Encaisse une commande servie (calcule la monnaie à rendre pour les espèces)."""
    ok, error, order = await asyncio.to_thread(db.record_payment, order_id, body.method, body.amount_paid)
    if not ok:
        raise HTTPException(status_code=400, detail=error)
    await store.broadcast({"type": "order", "event": "updated", "order": order})
    return {"ok": True, "order": order}


@app.get("/api/orders/history", dependencies=PROTECTED)
async def orders_history(limit: int = Query(default=100, ge=1, le=500)):
    """Historique complet — toutes les commandes, y compris payées/annulées."""
    return await asyncio.to_thread(db.list_history, limit)


@app.get("/api/orders/{order_id}/timeline", dependencies=PROTECTED)
async def order_timeline(order_id: str):
    """Timeline d'une commande : chaque changement de statut avec son horodatage."""
    order = await asyncio.to_thread(db.get_order, order_id)
    if order is None:
        raise HTTPException(status_code=404, detail="Commande introuvable.")
    timeline = await asyncio.to_thread(db.get_timeline, order_id)
    return {"order": order, "timeline": timeline}


@app.get("/api/stats/today", dependencies=PROTECTED)
async def stats_today():
    return await asyncio.to_thread(db.stats_today)


@app.post("/api/events/table_status", dependencies=PROTECTED)
async def post_table_status(evt: TableStatusEvent):
    """
    Appelé par vision_bridge.py (traduction des Action de l'Orchestrator
    vision). Un statut "attente_longue" crée EN PLUS une alerte staff,
    comme un robot bloqué — ça demande une action humaine.
    """
    tables = await asyncio.to_thread(db.load_state, "tables", {})
    tables[evt.table] = {
        "status": evt.status,
        "message": evt.message,
        "waiting_s": evt.waiting_s,
        "ts": time.time(),
    }
    await asyncio.to_thread(db.save_state, "tables", tables)
    await store.broadcast({"type": "table_status", "table": evt.table, "info": tables[evt.table]})

    if evt.status == "attente_longue":
        candidate = {
            "type": "table_waiting",
            "table": evt.table,
            "message": evt.message or f"Table {evt.table} attend depuis trop longtemps",
            "waiting_s": evt.waiting_s,
            "ts": time.time(),
        }
        alert, created = await asyncio.to_thread(
            db.create_alert, candidate, f"table_waiting:{evt.table}"
        )
        if created:
            await store.broadcast({"type": "alert", "alert": alert})

    return {"ok": True}


@app.post("/api/events/payment_intent", dependencies=PROTECTED)
async def post_payment_intent(evt: PaymentIntentEvent):
    """
    Le robot n'encaisse JAMAIS réellement. Cette route reçoit uniquement
    l'INTENTION de paiement exprimée à voix haute par le client, et la
    transforme en alerte pour le staff — au même titre qu'un robot bloqué,
    car ça demande une action humaine (aller encaisser à cette table).
    L'encaissement réel se fait via POST /api/orders/{id}/pay, uniquement
    quand la commande est passée au statut "servie".
    """
    order_id = _payment_order_id(evt)
    if not order_id:
        raise HTTPException(status_code=422, detail="order_id requis pour corréler le paiement.")
    order = await asyncio.to_thread(db.get_order, order_id)
    if order is None:
        raise HTTPException(status_code=409, detail="Commande de paiement introuvable.")
    if order["table_name"] != evt.table:
        raise HTTPException(status_code=409, detail="La table ne correspond pas à la commande.")
    if round(order["total"] * 1000) != round(evt.total * 1000):
        raise HTTPException(status_code=409, detail="Le total ne correspond pas à la commande.")
    if order["devise"] != evt.devise:
        raise HTTPException(status_code=409, detail="La devise ne correspond pas à la commande.")

    candidate = {
        "type": "payment_intent",
        "order_id": order_id,
        "table": evt.table,
        "payment_method": evt.payment_method,
        "items": evt.items,
        "total": evt.total,
        "devise": evt.devise,
        "session_id": evt.session_id,
        "ts": time.time(),
    }
    alert, created = await asyncio.to_thread(
        db.create_alert, candidate, f"payment:{order_id}:{evt.payment_method}"
    )
    if created:
        await store.broadcast({"type": "alert", "alert": alert})
    return {"ok": True, "alert_id": alert["id"], "created": created}


@app.post("/api/events/robot_status", dependencies=PROTECTED)
async def post_robot_status(evt: RobotStatusEvent):
    robot_status = {"status": evt.status, "message": evt.message, "table": evt.table, "ts": time.time()}
    await asyncio.to_thread(db.save_state, "robot_status", robot_status)

    # Les statuts "bloque" et "aide_demandee" génèrent une alerte persistante
    if evt.status in ("bloque", "aide_demandee"):
        candidate = {
            "type": "robot_status",
            "status": evt.status,
            "message": evt.message,
            "table": evt.table,
            "ts": time.time(),
        }
        alert, created = await asyncio.to_thread(
            db.create_alert,
            candidate,
            f"robot:{evt.status}:{evt.table or '-'}",
        )
        if created:
            await store.broadcast({"type": "alert", "alert": alert})

    await store.broadcast({"type": "robot_status", "robot_status": robot_status})
    return {"ok": True}


@app.post("/api/events/robot_position", dependencies=PROTECTED)
async def post_robot_position(evt: RobotPositionEvent):
    robot_position = {"x": evt.x, "y": evt.y, "theta": evt.theta, "ts": time.time()}
    await asyncio.to_thread(db.save_state, "robot_position", robot_position)
    await store.broadcast({"type": "robot_position", "robot_position": robot_position})
    return {"ok": True}


@app.post("/api/events/dialog", dependencies=PROTECTED)
async def post_dialog_event(evt: DialogEvent):
    """État de la conversation pour l'écran client (idle/listening/processing/speaking) + texte."""
    dialog_state = {"state": evt.state, "text": evt.text, "lang": evt.lang, "ts": time.time()}
    await asyncio.to_thread(db.save_state, "dialog_state", dialog_state)
    await store.broadcast_client_screen({"type": "dialog_state", "dialog_state": dialog_state})
    return {"ok": True}


@app.post("/api/events/draft_order", dependencies=PROTECTED)
async def post_draft_order_event(evt: DraftOrderEvent):
    """Panier en cours de construction (avant confirmation), pour affichage live sur l'écran client.
    À vider (items: []) une fois la commande confirmée ou annulée."""
    if store.suppresses_draft(evt.items, evt.total):
        return {"ok": True, "suppressed": True}
    draft_order = {"items": evt.items, "total": evt.total, "ts": time.time()}
    await asyncio.to_thread(db.save_state, "draft_order", draft_order)
    await store.broadcast_client_screen({"type": "draft_order", "draft_order": draft_order})
    return {"ok": True, "suppressed": False}


# ═══════════════════════════════════════════════════════════════════════════
# Actions staff (depuis le dashboard)
# ═══════════════════════════════════════════════════════════════════════════

@app.post("/api/staff_action", dependencies=PROTECTED)
async def staff_action(action: StaffAction):
    """Conservé pour l'action non liée au workflow commande (acquitter une alerte).
    Pour faire avancer une commande, utiliser /api/orders/{order_id}/advance."""
    if action.action == "acquitter_alerte":
        deleted = await asyncio.to_thread(db.acknowledge_alert, action.order_id)
        if not deleted:
            raise HTTPException(status_code=404, detail="Alerte introuvable.")
        await store.broadcast({"type": "alert_cleared", "alert_id": action.order_id})

    return {"ok": True}


@app.get("/api/state", dependencies=PROTECTED)
async def get_state():
    return await store.snapshot()


@app.get("/api/client_state", dependencies=PROTECTED)
async def get_client_state():
    return await store.client_snapshot()


# ═══════════════════════════════════════════════════════════════════════════
# Menu + QR code
# ═══════════════════════════════════════════════════════════════════════════

def _load_menu() -> dict:
    """Charge le menu.json canonique (schéma unifié, partagé avec le pipeline NLP
    vocal via menu_loader.py : items à plat avec id/categorie/disponible/etc.)."""
    with open(MENU_PATH, encoding="utf-8") as f:
        return json.load(f)


def _menu_view(menu: dict) -> dict:
    """Reformate le menu canonique (items à plat) en catégories imbriquées avec
    leurs items, pour l'affichage web (écran client + page /menu du QR code).
    Exclut automatiquement les plats marqués disponible=false."""
    items_by_cat: dict[str, list] = {}
    for it in menu.get("items", []):
        if not it.get("disponible", True):
            continue
        items_by_cat.setdefault(it["categorie"], []).append({
            "nom": it["nom"],
            "prix": it["prix"],
            "desc": it.get("description", {}),
            "image": (
                Path(str(it["image"])).name
                if it.get("image") and re.fullmatch(r"[A-Za-z0-9_.-]+", Path(str(it["image"])).name)
                else None
            ),
        })

    categories = []
    for cat in menu.get("categories", []):
        cat_items = items_by_cat.get(cat["id"], [])
        if not cat_items:
            continue   # catégorie vide (tout indisponible) -> pas affichée
        categories.append({"name": cat["nom"], "icon": cat.get("icon", "🍽"), "items": cat_items})

    return {"categories": categories, "menu_url": menu.get("menu_url")}


def _get_local_ip() -> str:
    """Détecte l'IP locale de la machine sur le réseau (sans envoyer aucune donnée,
    juste pour savoir quelle interface le système utiliserait). Fonctionne même
    sans accès internet, car aucune connexion réelle n'est établie."""
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        s.connect(("8.8.8.8", 80))
        return s.getsockname()[0]
    except Exception:
        return "127.0.0.1"
    finally:
        s.close()


@app.get("/api/menu")
async def get_menu():
    """Contenu du menu (catégories/plats/prix), utilisé par l'écran client.
    Vue filtrée/imbriquée dérivée du menu.json canonique (voir _menu_view)."""
    menu = await asyncio.to_thread(_load_menu)
    return _menu_view(menu)


@app.get("/menu", response_class=HTMLResponse)
async def menu_page(lang: str = "fr"):
    """Page HTML simple du menu — c'est la cible du QR code (consultable depuis le téléphone du client,
    sur le même réseau WiFi que le robot). ?lang=fr|en|ar pour choisir la langue."""
    if lang not in ("fr", "en", "ar"):
        lang = "fr"
    menu = _menu_view(_load_menu())
    sections = ""
    for cat in menu["categories"]:
        rows = ""
        for it in cat["items"]:
            names = it["nom"] if isinstance(it["nom"], dict) else {"fr": str(it["nom"])}
            nom = html.escape(str(names.get(lang, names.get("fr", ""))))
            descriptions = it.get("desc") or {}
            if not isinstance(descriptions, dict):
                descriptions = {"fr": str(descriptions)}
            desc = html.escape(str(descriptions.get(lang, "")))
            desc_html = f"<small>{desc}</small>" if desc else ""
            img_html = (
                f'<img src="/images/{quote(it["image"])}" alt="">'
                if it.get("image") else ""
            )
            rows += (
                f"<li>{img_html}"
                f"<div class='li-text'><span>{nom}</span><b>{it['prix']:.2f} TND</b>{desc_html}</div></li>"
            )
        names = cat["name"] if isinstance(cat["name"], dict) else {"fr": str(cat["name"])}
        cat_name = html.escape(str(names.get(lang, names.get("fr", ""))))
        icon = html.escape(str(cat.get("icon", "")))
        sections += f"<h2>{icon} {cat_name}</h2><ul>{rows}</ul>"

    dir_attr = ' dir="rtl"' if lang == "ar" else ""
    return f"""<!DOCTYPE html><html lang="{lang}"{dir_attr}><head><meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1">
    <title>NEXOR · Menu</title>
    <style>
      body {{ font-family: -apple-system, sans-serif; background:#FBF3E7; color:#4A392E; margin:0; padding:24px; }}
      h1 {{ color:#C98A2E; }}
      h2 {{ border-bottom:1px solid #EAD9C2; padding-bottom:6px; margin-top:28px; }}
      ul {{ list-style:none; padding:0; }}
      li {{ display:flex; gap:14px; align-items:center; padding:10px 0; border-bottom:1px solid #F4E8D8; }}
      li img {{ width:52px; height:52px; border-radius:10px; object-fit:cover; flex-shrink:0; }}
      .li-text {{ flex:1; display:flex; flex-wrap:wrap; justify-content:space-between; gap:4px; }}
      li b {{ color:#C98A2E; }}
      li small {{ flex-basis:100%; color:#8A7660; }}
    </style></head><body>
    <div style="background:#FFFFFF; border-radius:16px; padding:8px 24px 24px; box-shadow:0 2px 8px rgba(74,57,46,0.08);">
    <h1>🍽 NEXOR — Menu</h1>
    {sections}
    </div>
    </body></html>"""


def _resolve_menu_url(menu: dict, request: Request) -> str:
    """URL utilisée par le QR code : celle configurée dans menu.json si elle a été
    personnalisée, sinon l'IP locale de la machine détectée automatiquement."""
    configured = menu.get("menu_url")
    if configured and "192.168.1.50" not in configured:
        return configured
    ip = _get_local_ip()
    port = request.url.port or 8000
    scheme = "https" if request.url.scheme == "https" else "http"
    return f"{scheme}://{ip}:{port}/menu"


@app.get("/api/menu_qr_url")
async def menu_qr_url(request: Request):
    """Renvoie l'URL exacte encodée dans le QR code — utile pour vérifier/déboguer
    sans avoir à scanner (affichée en petit sous le QR sur l'écran client)."""
    menu = _load_menu()
    return {"url": _resolve_menu_url(menu, request)}


@app.get("/api/menu_qr")
async def menu_qr(request: Request):
    """QR code (PNG) pointant vers /menu — généré localement, aucune dépendance internet.

    Si menu_url dans menu.json est encore la valeur d'exemple (192.168.1.50) ou absente,
    l'IP locale de la machine est détectée automatiquement — plus besoin de l'éditer à
    la main à chaque changement de réseau."""
    menu = _load_menu()
    url = _resolve_menu_url(menu, request)

    def _make_png() -> bytes:
        qr = qrcode.QRCode(box_size=8, border=2)
        qr.add_data(url)
        qr.make(fit=True)
        img = qr.make_image(fill_color="#4A392E", back_color="#FBF3E7")
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        return buf.getvalue()

    png_bytes = await asyncio.to_thread(_make_png)
    return StreamingResponse(io.BytesIO(png_bytes), media_type="image/png")


# ═══════════════════════════════════════════════════════════════════════════
# Sert le dashboard staff (racine) et l'écran client (/client)
# ═══════════════════════════════════════════════════════════════════════════

IMAGES_DIR.mkdir(parents=True, exist_ok=True)   # ne casse jamais le démarrage si le dossier a été supprimé
app.mount("/images", StaticFiles(directory=str(IMAGES_DIR)), name="menu_images")
app.mount("/client", StaticFiles(directory=str(CLIENT_DIR), html=True), name="client_screen")
app.mount("/", StaticFiles(directory=str(FRONTEND_DIR), html=True), name="frontend")
