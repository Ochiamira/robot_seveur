"""
vision_bridge.py
=================
Traduit les Action de l'Orchestrator (vision NEXOR) en appels HTTP vers
le backend staff (main.py), pour que le dashboard affiche l'état des
tables en temps réel.

Usage :
    from orchestrator import Orchestrator
    from vision_bridge import on_action

    orch = Orchestrator(ros2_node=..., on_action=on_action)

Le format de table_id CHANGE de nature à cette frontière, volontairement :
- Côté vision (orchestrator.py, state_machine.py, customer_manager.py) :
  table_id est un ENTIER (3, 7, ...) — on ne touche à AUCUN de ces fichiers
  pour ce projet, ils fonctionnent déjà et sont testés.
- Côté dashboard/NLP (main.py, dialog_manager.py, db.py) : table_id est
  une CHAÎNE ("T3", "T7", ...) — plus lisible pour un humain sur l'écran.
La conversion se fait UNIQUEMENT ici, à la frontière, à un seul endroit
du code — si un jour la convention change, un seul fichier à modifier.
"""

import logging
import os
import threading
import time
from typing import Optional

import requests

from orchestrator import Action, ActionType

logger = logging.getLogger(__name__)

STAFF_APP_URL = "http://localhost:8000"
NAVIGATION_BRIDGE_URL = os.environ.get(
    "NEXOR_NAVIGATION_URL", "http://localhost:8090")
COMMUNICATION_BRIDGE_URL = os.environ.get(
    "NEXOR_COMMUNICATION_URL", "http://localhost:8100")

# Référence optionnelle vers le CustomerManager de la session en cours —
# nécessaire pour vérifier, avant de relayer ALERT_TABLE_FREE, qu'il ne
# reste PAS d'autres personnes assises à la même table (scénario groupe).
# Défini une seule fois au démarrage du pipeline via set_customer_manager().
_customer_manager = None
_orchestrator = None
_navigation_watchers = set()
_navigation_lock = threading.Lock()


def set_customer_manager(cm) -> None:
    """
    À appeler une fois, juste après la création du CustomerManager, dans
    run_vision_pipeline.py :

        cm = CustomerManager(...)
        vision_bridge.set_customer_manager(cm)

    Sans cet appel, vision_bridge continue de fonctionner (dégrade
    proprement), mais ne peut plus détecter qu'un groupe reste assis
    quand une seule personne part -> ALERT_TABLE_FREE serait alors
    relayée à chaque départ individuel, même table encore occupée.
    """
    global _customer_manager
    _customer_manager = cm


def set_orchestrator(orchestrator) -> None:
    """Connecte le retour d'arrivee Nav2 a l'Orchestrator actif."""
    global _orchestrator
    _orchestrator = orchestrator


def _start_dialogue(table_id: int, canonical_id: int) -> None:
    """Déclenche la boucle vocale Windows après l'arrivée physique."""
    try:
        response = requests.post(
            f"{COMMUNICATION_BRIDGE_URL}/dialog/start",
            json={"table_id": table_id, "canonical_id": canonical_id},
            timeout=2.0,
        )
        response.raise_for_status()
        logger.info(
            "[DIALOGUE] Déclenchement accepté: client #%s, table T%s",
            canonical_id,
            table_id,
        )
    except Exception as exc:
        logger.warning("[DIALOGUE] Service de communication injoignable: %s", exc)


def _start_dialogue_async(table_id: int, canonical_id: int) -> None:
    threading.Thread(
        target=_start_dialogue,
        args=(table_id, canonical_id),
        daemon=True,
        name=f"nexor-dialogue-table-{table_id}",
    ).start()


def _prepare_dialogue(table_id: int, canonical_id: int) -> None:
    """Précharge Whisper/TTS pendant que le robot se déplace."""
    try:
        response = requests.post(
            f"{COMMUNICATION_BRIDGE_URL}/dialog/prepare",
            json={"table_id": table_id, "canonical_id": canonical_id},
            timeout=2.0,
        )
        response.raise_for_status()
        logger.info(
            "[DIALOGUE] Préchargement demandé: client #%s, table T%s",
            canonical_id,
            table_id,
        )
    except Exception as exc:
        logger.warning("[DIALOGUE] Préchargement indisponible: %s", exc)


def _prepare_dialogue_async(table_id: Optional[int], canonical_id: int) -> None:
    if table_id is None:
        return
    threading.Thread(
        target=_prepare_dialogue,
        args=(table_id, canonical_id),
        daemon=True,
        name=f"nexor-dialogue-preload-table-{table_id}",
    ).start()


def _send_navigation(table_id: int, canonical_id: int,
                     priority: float = 0.0) -> None:
    """Envoie l'objectif fixe puis surveille Nav2 sans bloquer la vision."""
    watcher_key = (table_id, canonical_id)
    with _navigation_lock:
        if watcher_key in _navigation_watchers:
            return
        _navigation_watchers.add(watcher_key)

    try:
        response = requests.post(
            f"{NAVIGATION_BRIDGE_URL}/navigate",
            json={"table_id": table_id, "canonical_id": canonical_id,
                  "priority": priority},
            timeout=2.0,
        )
        response.raise_for_status()
        logger.info("[NAV2] Objectif envoye: client #%s -> table %s",
                    canonical_id, table_id)

        deadline = time.monotonic() + 300.0
        while time.monotonic() < deadline:
            time.sleep(0.5)
            status_response = requests.get(
                f"{NAVIGATION_BRIDGE_URL}/status", timeout=1.5)
            status_response.raise_for_status()
            status = status_response.json()
            if (status.get("table_id") != table_id
                    or status.get("canonical_id") != canonical_id):
                continue
            state = status.get("state")
            if state == "succeeded":
                logger.info("[NAV2] Robot arrive: client #%s, table %s",
                            canonical_id, table_id)
                if _orchestrator is not None:
                    _orchestrator.notify_robot_arrived(canonical_id)
                _start_dialogue_async(table_id, canonical_id)
                return
            if state in {"aborted", "canceled", "rejected", "error"}:
                logger.warning("[NAV2] Navigation terminee avec etat=%s: %s",
                               state, status.get("message", ""))
                return
        logger.warning("[NAV2] Delai de navigation depasse pour table %s",
                       table_id)
    except Exception as exc:
        logger.warning("[NAV2] Passerelle navigation injoignable: %s", exc)
    finally:
        with _navigation_lock:
            _navigation_watchers.discard(watcher_key)


def _navigate_async(table_id: Optional[int], canonical_id: int,
                    priority: float = 0.0) -> None:
    if table_id is None:
        logger.warning("[NAV2] Objectif ignore: aucun ID de table fixe")
        return
    threading.Thread(
        target=_send_navigation,
        args=(table_id, canonical_id, priority),
        daemon=True,
        name=f"nexor-nav-table-{table_id}",
    ).start()


def _post_navigation_cancel() -> None:
    try:
        requests.post(f"{NAVIGATION_BRIDGE_URL}/cancel", timeout=1.5)
    except Exception as exc:
        logger.debug("[NAV2] Annulation non transmise: %s", exc)


def _cancel_navigation_async() -> None:
    threading.Thread(
        target=_post_navigation_cancel,
        daemon=True,
        name="nexor-nav-cancel",
    ).start()


def _table_still_occupied(table_id: int) -> bool:
    """
    True s'il reste au moins une personne encore attablée à table_id
    (statut différent de PARTI). Utilisé pour éviter une fausse alerte
    "table libérée" quand une seule personne d'un groupe se lève.
    Si aucun CustomerManager n'a été fourni (set_customer_manager jamais
    appelé), on ne peut pas vérifier -> on suppose prudemment que non
    (comportement identique à avant ce correctif, pas de régression).
    """
    if _customer_manager is None:
        return False
    for record in _customer_manager.records_with_table().values():
        if record.table_id == table_id and record.statut != "PARTI":
            return True
    return False


def _table_str(table_id: Optional[int]) -> Optional[str]:
    """Convertit l'entier vision (3) en identifiant dashboard ("T3")."""
    if table_id is None:
        return None
    return f"T{table_id}"


def _post(path: str, payload: dict) -> None:
    try:
        requests.post(f"{STAFF_APP_URL}{path}", json=payload, timeout=1)
    except Exception as e:
        # Dashboard éteint ou hors réseau = cas normal, pas une erreur vision.
        logger.debug(f"Staff app injoignable ({path}) : {e}")


def _post_async(path: str, payload: dict) -> None:
    threading.Thread(target=_post, args=(path, payload), daemon=True).start()


def _push_table_status(table_id: Optional[int], status: str,
                        message: str = "", waiting_s: float = None) -> None:
    table = _table_str(table_id)
    if table is None:
        return  # pas encore de table assignée -> rien à afficher côté dashboard
    _post_async("/api/events/table_status", {
        "table": table,
        "status": status,
        "message": message,
        "waiting_s": waiting_s,
    })


def on_action(action: Action) -> None:
    """
    Callback à passer à Orchestrator(on_action=...).

    Ne prend AUCUNE décision métier — l'Orchestrator a déjà décidé, ce
    module se contente de traduire cette décision en mise à jour dashboard.
    Best-effort, non-bloquant, jamais d'exception qui remonte : l'Orchestrator
    encadre déjà l'appel dans un try/except (voir _execute()), mais on
    double la ceinture ici pour ne jamais perturber la boucle vision.
    """
    try:
        if action.action_type == ActionType.PLAY_SEATED:
            # Client vient de s'asseoir -> table occupée, robot pas encore là
            _push_table_status(action.table_id, "occupee",
                                message="Client assis, en attente du robot")
            _prepare_dialogue_async(action.table_id, action.canonical_id)

        elif action.action_type == ActionType.SEND_ROBOT:
            waiting_s = action.payload.get("time_waiting_s")
            _push_table_status(action.table_id, "attente_robot",
                                message="Robot envoyé vers cette table",
                                waiting_s=waiting_s)
            _prepare_dialogue_async(action.table_id, action.canonical_id)
            _navigate_async(action.table_id, action.canonical_id,
                            action.priority)

        elif action.action_type == ActionType.ALERT_WAITING_LONG:
            waiting_s = action.payload.get("waiting_s")
            _push_table_status(action.table_id, "attente_longue",
                                message=action.payload.get("message", ""),
                                waiting_s=waiting_s)
            # /api/events/table_status crée AUSSI une alerte staff pour ce
            # statut précis côté main.py -> rien d'autre à faire ici.

        elif action.action_type == ActionType.PLAY_SERVED:
            _push_table_status(action.table_id, "en_service",
                                message="Client pris en charge par le robot")

        elif action.action_type == ActionType.ALERT_TABLE_FREE:
            # ⚠️ Correctif scénario groupe : orchestrator.py déclenche cette
            # action dès qu'UNE SEULE personne part, sans savoir si d'autres
            # restent assises à la même table (il raisonne par personne, pas
            # par table). On vérifie ici avant de relayer au dashboard —
            # sinon un groupe de 3 dont un seul se lève ferait croire au
            # staff que toute la table est libre.
            if action.table_id is not None and _table_still_occupied(action.table_id):
                logger.info(
                    f"[vision_bridge] ALERT_TABLE_FREE ignorée pour table "
                    f"{action.table_id} : d'autres personnes du groupe y sont encore.")
            else:
                _push_table_status(action.table_id, "libre",
                                    message=action.payload.get("message", ""))

        elif action.action_type == ActionType.CANCEL_ROBOT:
            _cancel_navigation_async()

        # Les autres ActionType (PLAY_WELCOME — pas encore de table assignée,
        # CANCEL_ROBOT, LOG_DEPARTURE, NOTIFY_ROS2 générique) ne concernent
        # pas l'affichage "état des tables" du dashboard -> ignorés ici.

    except Exception as e:
        logger.warning(f"[vision_bridge] Erreur traitement action {action}: {e}")


if __name__ == "__main__":
    # Test standalone SANS caméra ni vraie Orchestrator : simule quelques
    # Action à la main, comme le ferait l'Orchestrator, pour vérifier que
    # les appels HTTP partent bien (lance main.py avant, sur le port 8000).
    import time

    logging.basicConfig(level=logging.INFO)

    print("Test 1 — client assis à la table 3")
    on_action(Action(action_type=ActionType.PLAY_SEATED, canonical_id=1, table_id=3))

    time.sleep(0.5)
    print("Test 2 — robot envoyé vers la table 3")
    on_action(Action(action_type=ActionType.SEND_ROBOT, canonical_id=1, table_id=3,
                      payload={"time_waiting_s": 5.2}))

    time.sleep(0.5)
    print("Test 3 — client attend trop longtemps à la table 3 (doit créer une alerte)")
    on_action(Action(action_type=ActionType.ALERT_WAITING_LONG, canonical_id=1, table_id=3,
                      payload={"message": "Client #1 attend depuis 65s", "waiting_s": 65.0}))

    time.sleep(0.5)
    print("Test 4 — client servi à la table 3")
    on_action(Action(action_type=ActionType.PLAY_SERVED, canonical_id=1, table_id=3))

    time.sleep(0.5)
    print("Test 5 — table 3 libérée")
    on_action(Action(action_type=ActionType.ALERT_TABLE_FREE, canonical_id=1, table_id=3,
                      payload={"message": "Table 3 libérée (client #1)"}))

    time.sleep(1.5)  # laisser les threads daemon HTTP finir avant que le process ne meure
    print("\n→ Vérifie http://localhost:8000/api/state : table 'T3' doit apparaître avec status='libre'")
