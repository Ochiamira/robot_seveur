"""
db.py
=====
Persistance SQLite pour les commandes NEXOR.

Remplace le dict en mémoire de la V1 : les commandes, leur statut et
l'historique complet des transitions (timeline) survivent au redémarrage
du serveur.

Workflow des statuts :
    confirmee -> en_preparation -> prete -> servie -> payee
    (annulee possible depuis confirmee ou en_preparation)

Chaque transition est journalisée dans order_events -> ça donne la
timeline d'une commande gratuitement (créée à quelle heure, prête à
quelle heure, etc.).

Paiement :
    Le passage servie -> payee ne se fait plus via advance_status() mais
    via record_payment(), qui exige une méthode de paiement (espèces/carte)
    et calcule la monnaie à rendre pour les paiements en espèces.
"""

from __future__ import annotations

import json
import os
import sqlite3
import time
import uuid
from contextlib import contextmanager
from datetime import datetime, timedelta
from decimal import Decimal, InvalidOperation, ROUND_HALF_UP
from pathlib import Path
from typing import Iterator

DB_PATH = Path(os.getenv("NEXOR_DB_PATH", str(Path(__file__).parent / "nexor_staff.db")))

STATUS_FLOW = {
    "confirmee":      ["en_preparation", "annulee"],
    "en_preparation": ["prete", "annulee"],
    "prete":          ["servie"],
    "servie":         [],   # -> payee uniquement via record_payment()
    "payee":          [],
    "annulee":        [],
}

PAYMENT_METHODS = {"especes", "carte"}

ACTIVE_STATUSES = ("confirmee", "en_preparation", "prete", "servie")


class OrderConflictError(ValueError):
    """Un order_id existant a été réutilisé avec un contenu différent."""


def _connect():
    conn = sqlite3.connect(
        DB_PATH, timeout=5.0, isolation_level=None, check_same_thread=False
    )
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    conn.execute("PRAGMA busy_timeout = 5000")
    return conn


@contextmanager
def _connection() -> Iterator[sqlite3.Connection]:
    conn = _connect()
    try:
        yield conn
    finally:
        conn.close()


@contextmanager
def _transaction() -> Iterator[sqlite3.Connection]:
    with _connection() as conn:
        conn.execute("BEGIN IMMEDIATE")
        try:
            yield conn
        except Exception:
            conn.rollback()
            raise
        else:
            conn.commit()


def _to_millimes(value: float | int | str | Decimal) -> int:
    try:
        amount = Decimal(str(value))
    except (InvalidOperation, ValueError, TypeError) as exc:
        raise ValueError("Montant invalide.") from exc
    if not amount.is_finite() or amount < 0:
        raise ValueError("Le montant doit être fini et positif ou nul.")
    rounded = amount.quantize(Decimal("0.001"), rounding=ROUND_HALF_UP)
    return int(rounded * 1000)


def _from_millimes(value: int | None) -> float | None:
    return None if value is None else value / 1000.0


def init_db():
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    with _connection() as conn:
        conn.execute("PRAGMA journal_mode = WAL")
        conn.executescript("""
            BEGIN IMMEDIATE;
            CREATE TABLE IF NOT EXISTS orders (
                order_id       TEXT PRIMARY KEY,
                table_name     TEXT,
                items          TEXT NOT NULL,
                total          REAL,
                devise         TEXT DEFAULT 'TND',
                status         TEXT NOT NULL,
                lang           TEXT DEFAULT 'fr',
                created_ts     REAL NOT NULL,
                updated_ts     REAL NOT NULL,
                payment_method TEXT,
                amount_paid    REAL,
                change_due     REAL
            );

            CREATE TABLE IF NOT EXISTS order_events (
                id        INTEGER PRIMARY KEY AUTOINCREMENT,
                order_id  TEXT NOT NULL,
                status    TEXT NOT NULL,
                ts        REAL NOT NULL,
                FOREIGN KEY (order_id) REFERENCES orders(order_id) ON DELETE CASCADE
            );

            CREATE TABLE IF NOT EXISTS alerts (
                id          TEXT PRIMARY KEY,
                dedupe_key  TEXT UNIQUE,
                payload     TEXT NOT NULL,
                created_ts  REAL NOT NULL
            );

            CREATE TABLE IF NOT EXISTS app_state (
                key         TEXT PRIMARY KEY,
                value       TEXT NOT NULL,
                updated_ts  REAL NOT NULL
            );

            CREATE INDEX IF NOT EXISTS idx_orders_created ON orders(created_ts);
            CREATE INDEX IF NOT EXISTS idx_events_order ON order_events(order_id);
            CREATE INDEX IF NOT EXISTS idx_alerts_created ON alerts(created_ts);
            COMMIT;
        """)

        existing_cols = {row["name"] for row in conn.execute("PRAGMA table_info(orders)")}
        migrations = (
            ("payment_method", "TEXT"),
            ("amount_paid", "REAL"),
            ("change_due", "REAL"),
            ("total_millimes", "INTEGER"),
            ("amount_paid_millimes", "INTEGER"),
            ("change_due_millimes", "INTEGER"),
        )
        conn.execute("BEGIN IMMEDIATE")
        try:
            for column, column_type in migrations:
                if column not in existing_cols:
                    conn.execute(f"ALTER TABLE orders ADD COLUMN {column} {column_type}")
            conn.execute(
                "UPDATE orders SET total_millimes = ROUND(total * 1000) "
                "WHERE total_millimes IS NULL AND total IS NOT NULL"
            )
            conn.execute(
                "UPDATE orders SET amount_paid_millimes = ROUND(amount_paid * 1000) "
                "WHERE amount_paid_millimes IS NULL AND amount_paid IS NOT NULL"
            )
            conn.execute(
                "UPDATE orders SET change_due_millimes = ROUND(change_due * 1000) "
                "WHERE change_due_millimes IS NULL AND change_due IS NOT NULL"
            )
        except Exception:
            conn.rollback()
            raise
        else:
            conn.commit()


# ═══════════════════════════════════════════════════════════════════════
# Écriture
# ═══════════════════════════════════════════════════════════════════════

def create_order(
    table: str,
    items: list[str],
    total: float,
    devise: str = "TND",
    lang: str = "fr",
    order_id: str | None = None,
) -> tuple[dict, bool]:
    """Crée une commande idempotente et retourne ``(commande, créée)``."""
    order_id = order_id or uuid.uuid4().hex
    total_millimes = _to_millimes(total)
    encoded_items = json.dumps(items, ensure_ascii=False, separators=(",", ":"))
    now = time.time()

    with _transaction() as conn:
        existing = conn.execute(
            "SELECT * FROM orders WHERE order_id = ?", (order_id,)
        ).fetchone()
        if existing is not None:
            existing_data = _row_to_dict(existing)
            same_payload = (
                existing_data["table_name"] == table
                and existing_data["items"] == items
                and _to_millimes(existing_data["total"]) == total_millimes
                and existing_data["devise"] == devise
                and existing_data["lang"] == lang
            )
            if not same_payload:
                raise OrderConflictError(
                    f"L'identifiant {order_id} appartient déjà à une autre commande."
                )
            return existing_data, False

        conn.execute(
            """
            INSERT INTO orders (
                order_id, table_name, items, total, total_millimes, devise,
                status, lang, created_ts, updated_ts
            ) VALUES (?, ?, ?, ?, ?, ?, 'confirmee', ?, ?, ?)
            """,
            (
                order_id,
                table,
                encoded_items,
                _from_millimes(total_millimes),
                total_millimes,
                devise,
                lang,
                now,
                now,
            ),
        )
        conn.execute(
            "INSERT INTO order_events (order_id, status, ts) VALUES (?, 'confirmee', ?)",
            (order_id, now),
        )
        created = conn.execute(
            "SELECT * FROM orders WHERE order_id = ?", (order_id,)
        ).fetchone()
        return _row_to_dict(created), True


def advance_status(order_id: str, to_status: str) -> tuple[bool, str, dict | None]:
    """Fait avancer atomiquement une commande."""
    with _transaction() as conn:
        row = conn.execute(
            "SELECT * FROM orders WHERE order_id = ?", (order_id,)
        ).fetchone()
        if row is None:
            return False, "Commande introuvable.", None
        current = row["status"]
        if to_status not in STATUS_FLOW.get(current, []):
            return False, f"Transition {current} -> {to_status} non autorisée.", None

        now = time.time()
        updated = conn.execute(
            "UPDATE orders SET status = ?, updated_ts = ? "
            "WHERE order_id = ? AND status = ?",
            (to_status, now, order_id, current),
        )
        if updated.rowcount != 1:
            return False, "La commande a été modifiée simultanément. Réessayez.", None
        conn.execute(
            "INSERT INTO order_events (order_id, status, ts) VALUES (?, ?, ?)",
            (order_id, to_status, now),
        )
        result = conn.execute(
            "SELECT * FROM orders WHERE order_id = ?", (order_id,)
        ).fetchone()
        return True, "", _row_to_dict(result)


def record_payment(order_id: str, method: str, amount_paid: float | None = None) -> tuple[bool, str, dict | None]:
    """Encaisse atomiquement une commande servie."""
    with _transaction() as conn:
        row = conn.execute(
            "SELECT * FROM orders WHERE order_id = ?", (order_id,)
        ).fetchone()
        if row is None:
            return False, "Commande introuvable.", None
        if row["status"] != "servie":
            return False, (
                "La commande doit être 'servie' avant d'être encaissée "
                f"(statut actuel : {row['status']})."
            ), None
        if method not in PAYMENT_METHODS:
            return False, f"Méthode de paiement inconnue : {method}.", None

        total_millimes = row["total_millimes"]
        if total_millimes is None:
            total_millimes = _to_millimes(row["total"] or 0)

        if method == "especes":
            if amount_paid is None:
                return False, "Le montant reçu est requis pour un paiement en espèces.", None
            paid_millimes = _to_millimes(amount_paid)
            if paid_millimes < total_millimes:
                return False, "Montant reçu insuffisant.", None
            change_millimes = paid_millimes - total_millimes
        else:
            paid_millimes = total_millimes
            change_millimes = 0

        now = time.time()
        updated = conn.execute(
            """
            UPDATE orders
            SET status = 'payee', updated_ts = ?, payment_method = ?,
                amount_paid = ?, amount_paid_millimes = ?,
                change_due = ?, change_due_millimes = ?
            WHERE order_id = ? AND status = 'servie'
            """,
            (
                now,
                method,
                _from_millimes(paid_millimes),
                paid_millimes,
                _from_millimes(change_millimes),
                change_millimes,
                order_id,
            ),
        )
        if updated.rowcount != 1:
            return False, "La commande a été encaissée simultanément.", None
        conn.execute(
            "INSERT INTO order_events (order_id, status, ts) VALUES (?, 'payee', ?)",
            (order_id, now),
        )
        result = conn.execute(
            "SELECT * FROM orders WHERE order_id = ?", (order_id,)
        ).fetchone()
        return True, "", _row_to_dict(result)


# ═══════════════════════════════════════════════════════════════════════
# Lecture
# ═══════════════════════════════════════════════════════════════════════

def _row_to_dict(row: sqlite3.Row) -> dict:
    data = dict(row)
    try:
        data["items"] = json.loads(data["items"])
    except (TypeError, json.JSONDecodeError):
        data["items"] = []
    total_millimes = data.pop("total_millimes", None)
    paid_millimes = data.pop("amount_paid_millimes", None)
    change_millimes = data.pop("change_due_millimes", None)
    if total_millimes is not None:
        data["total"] = _from_millimes(total_millimes)
    if paid_millimes is not None:
        data["amount_paid"] = _from_millimes(paid_millimes)
    if change_millimes is not None:
        data["change_due"] = _from_millimes(change_millimes)
    return data


def get_order(order_id: str) -> dict | None:
    with _connection() as conn:
        row = conn.execute("SELECT * FROM orders WHERE order_id = ?", (order_id,)).fetchone()
    return _row_to_dict(row) if row else None


def list_active_orders() -> list[dict]:
    """Commandes pas encore terminées (pour le dashboard temps réel)."""
    placeholders = ",".join("?" * len(ACTIVE_STATUSES))
    with _connection() as conn:
        rows = conn.execute(
            f"SELECT * FROM orders WHERE status IN ({placeholders}) ORDER BY created_ts DESC",
            ACTIVE_STATUSES,
        ).fetchall()
    return [_row_to_dict(r) for r in rows]


def list_history(limit: int = 100, date_from: float | None = None, date_to: float | None = None) -> list[dict]:
    """Historique complet (toutes commandes, y compris payées/annulées)."""
    limit = max(1, min(int(limit), 500))
    query = "SELECT * FROM orders WHERE 1=1"
    params: list = []
    if date_from is not None:
        query += " AND created_ts >= ?"
        params.append(date_from)
    if date_to is not None:
        query += " AND created_ts <= ?"
        params.append(date_to)
    query += " ORDER BY created_ts DESC LIMIT ?"
    params.append(limit)

    with _connection() as conn:
        rows = conn.execute(query, params).fetchall()
    return [_row_to_dict(r) for r in rows]


def get_timeline(order_id: str) -> list[dict]:
    with _connection() as conn:
        rows = conn.execute(
            "SELECT status, ts FROM order_events WHERE order_id = ? ORDER BY id ASC",
            (order_id,),
        ).fetchall()
    return [dict(r) for r in rows]


def stats_today() -> dict:
    """Statistiques du jour civil local; le CA ne compte que les encaissements."""
    now = datetime.now().astimezone()
    start = now.replace(hour=0, minute=0, second=0, microsecond=0)
    end = start + timedelta(days=1)
    with _connection() as conn:
        orders_count = conn.execute(
            "SELECT COUNT(*) FROM orders WHERE created_ts >= ? AND created_ts < ?",
            (start.timestamp(), end.timestamp()),
        ).fetchone()[0]
        paid_total = conn.execute(
            "SELECT COALESCE(SUM(total_millimes), 0) FROM orders "
            "WHERE created_ts >= ? AND created_ts < ? AND status = 'payee'",
            (start.timestamp(), end.timestamp()),
        ).fetchone()[0]
        paid = conn.execute(
            "SELECT payment_method, COUNT(*) AS n, "
            "COALESCE(SUM(total_millimes), 0) AS total_millimes FROM orders "
            "WHERE created_ts >= ? AND created_ts < ? AND status = 'payee' "
            "GROUP BY payment_method",
            (start.timestamp(), end.timestamp()),
        ).fetchall()
    payments = [
        {
            "payment_method": row["payment_method"],
            "n": row["n"],
            "total": _from_millimes(row["total_millimes"]),
        }
        for row in paid
    ]
    return {
        "date": start.date().isoformat(),
        "n_commandes": orders_count,
        "ca_encaisse": _from_millimes(paid_total),
        "encaissements": payments,
    }


def create_alert(payload: dict, dedupe_key: str | None = None) -> tuple[dict, bool]:
    """Persiste une alerte et déduplique les répétitions non acquittées."""
    alert = dict(payload)
    alert.setdefault("id", uuid.uuid4().hex)
    alert.setdefault("ts", time.time())
    encoded = json.dumps(alert, ensure_ascii=False, separators=(",", ":"))
    with _transaction() as conn:
        try:
            conn.execute(
                "INSERT INTO alerts (id, dedupe_key, payload, created_ts) VALUES (?, ?, ?, ?)",
                (alert["id"], dedupe_key, encoded, alert["ts"]),
            )
            return alert, True
        except sqlite3.IntegrityError:
            if dedupe_key is None:
                raise
            row = conn.execute(
                "SELECT payload FROM alerts WHERE dedupe_key = ?", (dedupe_key,)
            ).fetchone()
            if row is None:
                raise
            return json.loads(row["payload"]), False


def list_alerts(limit: int = 1000) -> list[dict]:
    limit = max(1, min(int(limit), 1000))
    with _connection() as conn:
        rows = conn.execute(
            "SELECT payload FROM alerts ORDER BY created_ts ASC LIMIT ?", (limit,)
        ).fetchall()
    return [json.loads(row["payload"]) for row in rows]


def acknowledge_alert(alert_id: str) -> bool:
    with _transaction() as conn:
        deleted = conn.execute("DELETE FROM alerts WHERE id = ?", (alert_id,))
        return deleted.rowcount == 1


def save_state(key: str, value: dict) -> None:
    encoded = json.dumps(value, ensure_ascii=False, separators=(",", ":"))
    with _transaction() as conn:
        conn.execute(
            "INSERT INTO app_state (key, value, updated_ts) VALUES (?, ?, ?) "
            "ON CONFLICT(key) DO UPDATE SET value = excluded.value, "
            "updated_ts = excluded.updated_ts",
            (key, encoded, time.time()),
        )


def load_state(key: str, default: dict) -> dict:
    with _connection() as conn:
        row = conn.execute("SELECT value FROM app_state WHERE key = ?", (key,)).fetchone()
    if row is None:
        return dict(default)
    try:
        value = json.loads(row["value"])
    except (TypeError, json.JSONDecodeError):
        return dict(default)
    return value if isinstance(value, dict) else dict(default)
