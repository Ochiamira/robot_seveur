"""
simulate.py
===========
Génère des événements fictifs (commandes, statut robot, position) pour
tester le dashboard SANS robot ni ROS2 branché.

Usage :
    # dans un terminal : lance le serveur
    python -m uvicorn main:app --host 0.0.0.0 --port 8000 --workers 1

    # dans un autre terminal :
    python simulate.py
"""

import math
import os
import random
import time
import requests

BASE = "http://localhost:8000"
TOKEN = os.getenv("STAFF_APP_TOKEN", "").strip()

ORDERS = [
    {"table": "T4", "items": ["1× Couscous agneau", "1× Café espresso"], "total": 20.5},
    {"table": "T7", "items": ["2× Tagine poulet", "1× Jus d'orange frais"], "total": 32.0},
    {"table": "T2", "items": ["1× Salade tunisienne"], "total": 8.5},
]


def send(path, payload):
    headers = {"Authorization": f"Bearer {TOKEN}"} if TOKEN else {}
    r = requests.post(f"{BASE}{path}", json=payload, headers=headers, timeout=5)
    r.raise_for_status()
    return r.json()


def main():
    print("🧪 Simulation démarrée — Ctrl+C pour arrêter")
    t = 0.0
    order_ids = []

    while True:
        # Nouvelle commande de temps en temps
        if random.random() < 0.15:
            o = random.choice(ORDERS)
            res = send("/api/events/order", {"event": "confirmed", **o})
            order_ids.append(res["order_id"])
            print(f"🧾 Commande confirmée : {o['table']} — {o['items']}")

        # Position du robot : trajectoire circulaire simulée
        x = 0.5 + 0.35 * math.cos(t)
        y = 0.5 + 0.35 * math.sin(t)
        send("/api/events/robot_position", {"x": x, "y": y, "theta": t})
        t += 0.1

        # Statut robot aléatoire, avec alerte occasionnelle
        if random.random() < 0.03:
            status, msg = random.choice([
                ("bloque", "Obstacle détecté devant la table T3"),
                ("aide_demandee", "Client demande assistance humaine"),
            ])
            send("/api/events/robot_status", {"status": status, "message": msg, "table": "T3"})
            print(f"🚨 Alerte : {status} — {msg}")
        else:
            send("/api/events/robot_status", {"status": "en_route", "message": ""})

        time.sleep(1)


if __name__ == "__main__":
    main()
