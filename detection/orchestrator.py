"""
=============================================================
  NEXOR Vision — Orchestrator  (11-orchestrator.py)
  ─────────────────────────────────────────────────────────
  Rôle UNIQUE : lire les données (CustomerRecord) et le
  statut (StateMachine) et décider des ACTIONS à exécuter.

  ❌ Ne stocke AUCUNE donnée client (c'est CustomerManager)
  ❌ Ne fait AUCUNE transition d'état (c'est StateMachine)

  ✅ Décide d'envoyer le robot vers une table
  ✅ Ordonnance les robots si plusieurs clients attendent
  ✅ Publie les commandes ROS2 (navigation, TTS, affichage)
  ✅ Gère les cooldowns (pas de double envoi robot)
  ✅ Génère les recommandations / messages vocaux
  ✅ Logue toutes les actions pour audit
  ✅ Gère les alertes (client attend trop longtemps)

  Intégration :
    orchestrator = Orchestrator(ros2_node=node)   # ou None si pas ROS2
    # à chaque frame, après StateMachine :
    orchestrator.process(record, state_machine)
    # en fin de boucle :
    orchestrator.tick()   # actions périodiques (alertes, timeouts)
=============================================================
"""

import time
import json
import math
import logging
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional, Dict, List, Callable, Any
from enum import Enum
from collections import defaultdict


# ─────────────────────────────────────────────────────────────
#  LOGGER
# ─────────────────────────────────────────────────────────────

logging.basicConfig(
    level   = logging.INFO,
    format  = "[%(asctime)s] [ORCH] %(levelname)s — %(message)s",
    datefmt = "%H:%M:%S",
)
log = logging.getLogger("orchestrator")


# ─────────────────────────────────────────────────────────────
#  TYPES D'ACTIONS
# ─────────────────────────────────────────────────────────────

class ActionType(str, Enum):
    SEND_ROBOT          = "SEND_ROBOT"          # envoyer le robot à une table
    CANCEL_ROBOT        = "CANCEL_ROBOT"        # annuler une mission robot
    PLAY_WELCOME        = "PLAY_WELCOME"        # message vocal "bienvenue"
    PLAY_SEATED         = "PLAY_SEATED"         # message vocal "je viens vous voir"
    PLAY_SERVED         = "PLAY_SERVED"         # message vocal "merci"
    DISPLAY_MENU        = "DISPLAY_MENU"        # afficher le menu sur l'écran robot
    ALERT_WAITING_LONG  = "ALERT_WAITING_LONG"  # alerte staff : client attend trop
    ALERT_TABLE_FREE    = "ALERT_TABLE_FREE"    # alerte staff : table libérée
    LOG_DEPARTURE       = "LOG_DEPARTURE"       # log client parti (stats)
    NOTIFY_ROS2         = "NOTIFY_ROS2"         # publication ROS2 générique


# ─────────────────────────────────────────────────────────────
#  ACTION — une décision concrète
# ─────────────────────────────────────────────────────────────

@dataclass
class Action:
    """
    Représente une action décidée par l'Orchestrator.
    Immuable après création.
    """
    action_type    : ActionType
    canonical_id   : int
    table_id       : Optional[int]    = None
    position       : Optional[tuple]  = None    # (cx, cy) pixels
    priority       : float            = 0.0
    payload        : Dict             = field(default_factory=dict)
    timestamp      : float            = field(default_factory=time.time)

    def __str__(self):
        tbl = f" table={self.table_id}" if self.table_id else ""
        return (f"Action[{self.action_type.value}] "
                f"client=#{self.canonical_id}{tbl} "
                f"priority={self.priority:.1f}")


# ─────────────────────────────────────────────────────────────
#  CONFIG ORCHESTRATOR
# ─────────────────────────────────────────────────────────────

@dataclass
class OrchestratorConfig:
    """Tous les paramètres de décision de l'Orchestrator."""

    # ── Cooldowns — évite les actions répétées (secondes) ────
    cooldown_robot_s       : float = 30.0   # délai min entre 2 envois robot
    cooldown_welcome_s     : float = 5.0    # délai min entre 2 messages bienvenue
    cooldown_alert_s       : float = 60.0   # délai min entre 2 alertes staff

    # ── Seuils de priorité robot ──────────────────────────────
    # La priorité augmente avec le temps d'attente
    # priority = time_in_statut / priority_scale
    priority_scale         : float = 10.0   # 10s d'attente = priorité 1.0
    max_priority           : float = 10.0   # plafond

    # ── Alerte "client attend trop longtemps" ─────────────────
    alert_waiting_s        : float = 90.0   # alerte après N s en EN_ATTENTE

    # ── Robot : max simultané ─────────────────────────────────
    max_robots_active      : int   = 3      # missions robot simultanées max

    # ── Actions vocales activées ──────────────────────────────
    tts_enabled            : bool  = True

    # ── Publication ROS2 activée ──────────────────────────────
    ros2_enabled           : bool  = True

    # ── Topic ROS2 ────────────────────────────────────────────
    ros2_topic_navigate    : str   = "/nexor/navigate_to_table"
    ros2_topic_cancel      : str   = "/nexor/cancel_mission"
    ros2_topic_tts         : str   = "/nexor/tts"
    ros2_topic_display     : str   = "/nexor/display"
    ros2_topic_alert       : str   = "/nexor/staff_alert"


# ─────────────────────────────────────────────────────────────
#  MISSION ROBOT — suivi d'une mission en cours
# ─────────────────────────────────────────────────────────────

@dataclass
class RobotMission:
    """Une mission robot assignée à un client/table."""
    canonical_id    : int
    table_id        : Optional[int]
    position        : Optional[tuple]
    priority        : float
    started_at      : float = field(default_factory=time.time)
    status          : str   = "PENDING"   # PENDING / ACTIVE / DONE / CANCELLED

    @property
    def age_s(self) -> float:
        return time.time() - self.started_at

    def to_dict(self) -> dict:
        return {
            "canonical_id" : self.canonical_id,
            "table_id"     : self.table_id,
            "priority"     : round(self.priority, 2),
            "status"       : self.status,
            "age_s"        : round(self.age_s, 1),
        }


# ─────────────────────────────────────────────────────────────
#  STUBS ROS2 — interfaces simulées si ROS2 indisponible
# ─────────────────────────────────────────────────────────────

class ROS2Publisher:
    """
    Wrapper autour du nœud ROS2.
    Si ros2_node=None → simule les publications (mode dev/test).
    """

    def __init__(self, node=None, config: OrchestratorConfig = None):
        self._node   = node
        self._config = config or OrchestratorConfig()
        self._pubs   = {}

        if node is not None:
            self._init_publishers()
        else:
            log.info("[ROS2] Mode simulation (pas de nœud ROS2)")

    def _init_publishers(self):
        """Crée les publishers ROS2 réels."""
        try:
            from std_msgs.msg import String
            from geometry_msgs.msg import PoseStamped
            cfg = self._config

            self._pubs["navigate"] = self._node.create_publisher(
                PoseStamped, cfg.ros2_topic_navigate, 10)
            self._pubs["cancel"]   = self._node.create_publisher(
                String, cfg.ros2_topic_cancel, 10)
            self._pubs["tts"]      = self._node.create_publisher(
                String, cfg.ros2_topic_tts, 10)
            self._pubs["display"]  = self._node.create_publisher(
                String, cfg.ros2_topic_display, 10)
            self._pubs["alert"]    = self._node.create_publisher(
                String, cfg.ros2_topic_alert, 10)
            log.info("[ROS2] Publishers initialisés ✓")
        except Exception as e:
            log.warning(f"[ROS2] Impossible d'initialiser les publishers : {e}")

    def publish_navigate(self, table_id: Optional[int],
                         position: Optional[tuple], priority: float,
                         canonical_id: int):
        """Publie une commande de navigation vers une table."""
        payload = {
            "canonical_id" : canonical_id,
            "table_id"     : table_id,
            "priority"     : round(priority, 2),
            "position_px"  : list(position) if position else None,
        }
        self._publish("navigate", payload)

    def publish_cancel(self, canonical_id: int, table_id: Optional[int]):
        """Annule une mission robot."""
        self._publish("cancel", {
            "canonical_id": canonical_id,
            "table_id"    : table_id,
        })

    def publish_tts(self, text: str, canonical_id: int):
        """Déclenche la synthèse vocale du robot."""
        self._publish("tts", {"text": text, "canonical_id": canonical_id})

    def publish_display(self, content: dict, canonical_id: int):
        """Commande l'affichage sur l'écran du robot."""
        self._publish("display", {"content": content,
                                   "canonical_id": canonical_id})

    def publish_alert(self, message: str, level: str = "WARNING"):
        """Envoie une alerte au staff."""
        self._publish("alert", {"message": message, "level": level})

    def _publish(self, channel: str, data: dict):
        """Publication réelle ou simulation."""
        msg_str = json.dumps(data, ensure_ascii=False)

        if self._node is not None and channel in self._pubs:
            try:
                from std_msgs.msg import String
                msg = String()
                msg.data = msg_str
                self._pubs[channel].publish(msg)
            except Exception as e:
                log.warning(f"[ROS2] Erreur publication {channel} : {e}")
        else:
            # Mode simulation : log uniquement
            log.debug(f"[ROS2-SIM] {channel.upper()} → {msg_str[:120]}")


# ─────────────────────────────────────────────────────────────
#  ORCHESTRATOR
# ─────────────────────────────────────────────────────────────

class Orchestrator:
    """
    Décide et exécute toutes les actions du système NEXOR.

    ❌ Ne stocke PAS les données client (CustomerManager)
    ❌ Ne fait PAS les transitions d'état (StateMachine)

    ✅ Reçoit record + state_machine → décide des actions
    ✅ Gère les missions robot (envoi, annulation, priorité)
    ✅ Publie sur ROS2 (navigation, TTS, affichage, alertes)
    ✅ Respecte les cooldowns (pas de spam)
    ✅ Ordonnance si plusieurs clients EN_ATTENTE
    ✅ Logue toutes les actions
    """

    def __init__(self,
                 ros2_node   = None,
                 config      : OrchestratorConfig = None,
                 output_dir  : str = "tracking_output",
                 on_action   : Optional[Callable[[Action], None]] = None):
        """
        ros2_node  : nœud ROS2 (ou None en mode simulation)
        config     : paramètres de décision
        output_dir : dossier pour le log des actions
        on_action  : callback optionnel appelé à chaque action
                     (utile pour tests, UI, websocket, etc.)
        """
        self._cfg        = config or OrchestratorConfig()
        self._ros2       = ROS2Publisher(ros2_node, self._cfg)
        self._output_dir = Path(output_dir)
        self._on_action  = on_action

        self._output_dir.mkdir(parents=True, exist_ok=True)

        # ── État interne de l'Orchestrator ────────────────────
        # Missions robot actives : canonical_id → RobotMission
        self._active_missions : Dict[int, RobotMission] = {}

        # Cooldowns : (canonical_id, ActionType) → timestamp dernière action
        self._last_action_t  : Dict[tuple, float] = defaultdict(float)

        # Log de toutes les actions exécutées
        self._action_log     : List[Action] = []

        # Statuts précédents pour détecter les transitions
        self._prev_statuts   : Dict[int, str] = {}

        # Tables occupées par des missions actives : table_id → canonical_id
        self._table_missions : Dict[int, int] = {}

        log.info("Orchestrator initialisé ✓")

    # ─────────────────────────────────────────────────────────
    #  POINT D'ENTRÉE PRINCIPAL
    # ─────────────────────────────────────────────────────────

    def process(self, record, state_machine) -> List[Action]:
        """
        Traite un CustomerRecord et la StateMachine associée.
        Décide et exécute les actions appropriées.

        Paramètres :
            record        : CustomerRecord (depuis CustomerManager)
            state_machine : ClientStateMachine (depuis StateMachine)

        Retourne :
            Liste des Actions exécutées ce frame pour ce client.
        """
        cid     = record.canonical_id
        statut  = record.statut
        prev    = self._prev_statuts.get(cid, "")
        actions = []

        # ── Détection de transition de statut ─────────────────
        transition = (prev != statut and prev != "")
        just_entered = transition

        # ── Dispatcher par statut ─────────────────────────────

        if statut == "NOUVEAU":
            actions += self._on_nouveau(record, just_entered)

        elif statut == "CHERCHE_TABLE":
            actions += self._on_cherche_table(record, just_entered)

        elif statut == "ASSIS":
            actions += self._on_assis(record, just_entered)

        elif statut == "EN_ATTENTE":
            actions += self._on_en_attente(record, state_machine, just_entered)

        elif statut == "SERVI":
            actions += self._on_servi(record, just_entered)

        elif statut == "PARTI":
            actions += self._on_parti(record, just_entered)

        # ── Mémoriser le statut actuel ────────────────────────
        self._prev_statuts[cid] = statut

        return actions

    # ─────────────────────────────────────────────────────────
    #  HANDLERS PAR STATUT
    # ─────────────────────────────────────────────────────────

    def _on_nouveau(self, record, just_entered: bool) -> List[Action]:
        """Client vient d'être détecté pour la première fois."""
        actions = []

        if just_entered and self._cfg.tts_enabled:
            # Message de bienvenue (une seule fois)
            if self._can_act(record.canonical_id, ActionType.PLAY_WELCOME):
                a = self._execute(Action(
                    action_type  = ActionType.PLAY_WELCOME,
                    canonical_id = record.canonical_id,
                    payload      = {"text": "Bienvenue ! Je serai avec vous dans un instant."},
                ))
                self._ros2.publish_tts(
                    "Bienvenue ! Je serai avec vous dans un instant.",
                    record.canonical_id,
                )
                actions.append(a)

        return actions

    def _on_cherche_table(self, record, just_entered: bool) -> List[Action]:
        """Client cherche une table — pas d'action robot pour l'instant."""
        actions = []
        # Aucune action immédiate — on attend qu'il s'assoie
        # On pourrait ici afficher une carte du restaurant
        return actions

    def _on_assis(self, record, just_entered: bool) -> List[Action]:
        """Client vient de s'asseoir."""
        actions = []

        if just_entered:
            log.info(f"[ASSIS] Client #{record.canonical_id} "
                     f"table={record.table_id} "
                     f"pos={record.position}")

            if self._cfg.tts_enabled:
                if self._can_act(record.canonical_id, ActionType.PLAY_SEATED):
                    msg = (f"Parfait ! Je viens vous voir à la table "
                           f"{record.table_id} dans un moment.")
                    a = self._execute(Action(
                        action_type  = ActionType.PLAY_SEATED,
                        canonical_id = record.canonical_id,
                        table_id     = record.table_id,
                        payload      = {"text": msg},
                    ))
                    self._ros2.publish_tts(msg, record.canonical_id)
                    actions.append(a)

        return actions

    def _on_en_attente(self, record, state_machine,
                       just_entered: bool) -> List[Action]:
        """
        Client en attente → décision principale :
        envoyer le robot ou non.
        """
        actions = []
        cid     = record.canonical_id

        # ── Envoi du robot ────────────────────────────────────
        if self._should_send_robot(record, state_machine):
            priority = self._compute_priority(state_machine)
            mission  = self._dispatch_robot(record, priority)
            if mission is not None:
                a = self._execute(Action(
                    action_type  = ActionType.SEND_ROBOT,
                    canonical_id = cid,
                    table_id     = record.table_id,
                    position     = record.position,
                    priority     = priority,
                    payload      = {
                        "table_id"       : record.table_id,
                        "position_px"    : list(record.position),
                        "time_waiting_s" : round(
                            state_machine.time_in_statut, 1),
                        "priority"       : round(priority, 2),
                    },
                ))
                actions.append(a)

                if self._cfg.tts_enabled:
                    self._ros2.publish_tts(
                        "Le robot arrive à votre table.",
                        cid,
                    )

                if self._cfg.ros2_enabled:
                    self._ros2.publish_navigate(
                        record.table_id, record.position,
                        priority, cid,
                    )

                if self._cfg.ros2_enabled:
                    self._ros2.publish_display(
                        {"type": "menu", "table_id": record.table_id},
                        cid,
                    )

        # ── Alerte "trop longtemps en attente" ────────────────
        if (state_machine.time_in_statut >= self._cfg.alert_waiting_s
                and self._can_act(cid, ActionType.ALERT_WAITING_LONG,
                                  self._cfg.cooldown_alert_s)):
            msg = (f"⚠ Client #{cid} attend depuis "
                   f"{state_machine.time_in_statut:.0f}s "
                   f"(table {record.table_id}) — robot non arrivé !")
            a = self._execute(Action(
                action_type  = ActionType.ALERT_WAITING_LONG,
                canonical_id = cid,
                table_id     = record.table_id,
                payload      = {
                    "message"       : msg,
                    "waiting_s"     : round(state_machine.time_in_statut, 1),
                    "has_mission"   : cid in self._active_missions,
                },
            ))
            actions.append(a)
            self._ros2.publish_alert(msg, level="WARNING")
            log.warning(msg)

        return actions

    def _on_servi(self, record, just_entered: bool) -> List[Action]:
        """Client servi → clore la mission robot."""
        actions = []
        cid = record.canonical_id

        if just_entered:
            log.info(f"[SERVI] Client #{cid} table={record.table_id}")

            # Clore la mission robot si active
            if cid in self._active_missions:
                self._close_mission(cid, status="DONE")
                a = self._execute(Action(
                    action_type  = ActionType.CANCEL_ROBOT,
                    canonical_id = cid,
                    table_id     = record.table_id,
                    payload      = {"reason": "client servi"},
                ))
                actions.append(a)
                self._ros2.publish_cancel(cid, record.table_id)

            # Message vocal de confirmation
            if self._cfg.tts_enabled:
                if self._can_act(cid, ActionType.PLAY_SERVED):
                    msg = "Votre commande a bien été enregistrée. Bon appétit !"
                    a = self._execute(Action(
                        action_type  = ActionType.PLAY_SERVED,
                        canonical_id = cid,
                        payload      = {"text": msg},
                    ))
                    self._ros2.publish_tts(msg, cid)
                    actions.append(a)

        return actions

    def _on_parti(self, record, just_entered: bool) -> List[Action]:
        """Client parti → libérer les ressources."""
        actions = []
        cid = record.canonical_id

        if just_entered:
            # La caméra est embarquée sur le robot. Dès que la navigation
            # commence, elle peut perdre naturellement la table et faire
            # passer à tort le client à PARTI. Une absence visuelle seule ne
            # doit donc jamais annuler une mission déjà envoyée à Nav2.
            mission = self._active_missions.get(cid)
            if mission is not None and mission.status in ("PENDING", "ACTIVE"):
                log.info(
                    f"[PARTI] Client #{cid} invisible, mais mission "
                    f"{mission.status} vers table={mission.table_id} : "
                    "annulation ignorée"
                )
                return actions

            log.info(f"[PARTI] Client #{cid} "
                     f"présence={record.presence_s:.1f}s "
                     f"table={record.table_id}")

            # Annuler toute mission en cours
            if cid in self._active_missions:
                self._close_mission(cid, status="CANCELLED")
                a = self._execute(Action(
                    action_type  = ActionType.CANCEL_ROBOT,
                    canonical_id = cid,
                    table_id     = record.table_id,
                    payload      = {"reason": "client parti"},
                ))
                actions.append(a)
                self._ros2.publish_cancel(cid, record.table_id)

            # Alerte table libérée
            if record.table_id is not None:
                a = self._execute(Action(
                    action_type  = ActionType.ALERT_TABLE_FREE,
                    canonical_id = cid,
                    table_id     = record.table_id,
                    payload      = {
                        "message"     : (f"Table {record.table_id} libérée "
                                         f"(client #{cid})"),
                        "presence_s"  : round(record.presence_s, 1),
                    },
                ))
                actions.append(a)
                self._ros2.publish_alert(
                    f"Table {record.table_id} libérée", level="INFO")

            # Log départ
            a = self._execute(Action(
                action_type  = ActionType.LOG_DEPARTURE,
                canonical_id = cid,
                table_id     = record.table_id,
                payload      = {
                    "presence_s"  : round(record.presence_s, 1),
                    "frame_count" : record.temporal.frame_count,
                    "crossed_in"  : record.crossed_in,
                    "crossed_out" : record.crossed_out,
                },
            ))
            actions.append(a)

        return actions

    # ─────────────────────────────────────────────────────────
    #  ACTIONS PÉRIODIQUES — appeler une fois par frame
    # ─────────────────────────────────────────────────────────

    def tick(self, all_records: Dict = None):
        """
        Actions périodiques indépendantes d'un client spécifique.
        Appeler une fois par frame en fin de boucle.

        - Nettoie les missions trop vieilles
        - Ordonnance les clients EN_ATTENTE par priorité
        - Log les statistiques globales périodiquement
        """
        now = time.time()

        # ── Nettoyer missions expirées (timeout 5 min) ────────
        to_close = [
            cid for cid, m in self._active_missions.items()
            if m.age_s > 300 and m.status == "PENDING"
        ]
        for cid in to_close:
            log.warning(f"[MISSION] Timeout mission #{cid} — annulation")
            self._close_mission(cid, status="CANCELLED")

    # ─────────────────────────────────────────────────────────
    #  LOGIQUE DE DÉCISION ROBOT
    # ─────────────────────────────────────────────────────────

    def _should_send_robot(self, record, state_machine) -> bool:
        """
        Décide si le robot doit être envoyé pour ce client.
        Vérifie toutes les conditions nécessaires.
        """
        cid = record.canonical_id

        # Statut incorrect
        if record.statut != "EN_ATTENTE":
            return False

        # Déjà une mission active pour ce client
        if cid in self._active_missions:
            return False

        # Cooldown robot non écoulé
        if not self._can_act(cid, ActionType.SEND_ROBOT,
                             self._cfg.cooldown_robot_s):
            return False

        # Trop de missions simultanées
        active = sum(1 for m in self._active_missions.values()
                     if m.status in ("PENDING", "ACTIVE"))
        if active >= self._cfg.max_robots_active:
            log.debug(f"[ROBOT] Max missions atteint ({active}) — "
                      f"client #{cid} en file d'attente")
            return False

        # Table déjà servie par une autre mission
        if (record.table_id is not None
                and record.table_id in self._table_missions
                and self._table_missions[record.table_id] != cid):
            return False

        return True

    def _compute_priority(self, state_machine) -> float:
        """
        Calcule la priorité du robot pour ce client.
        Plus le client attend, plus la priorité est haute.
        """
        raw = state_machine.time_in_statut / self._cfg.priority_scale
        return min(raw, self._cfg.max_priority)

    def _dispatch_robot(self, record,
                        priority: float) -> Optional[RobotMission]:
        """Crée et enregistre une mission robot."""
        cid = record.canonical_id
        mission = RobotMission(
            canonical_id = cid,
            table_id     = record.table_id,
            position     = record.position,
            priority     = priority,
        )
        self._active_missions[cid] = mission
        if record.table_id is not None:
            self._table_missions[record.table_id] = cid

        log.info(f"[ROBOT] Mission #{cid} → table={record.table_id} "
                 f"pos={record.position} priority={priority:.2f}")
        return mission

    def _close_mission(self, canonical_id: int, status: str = "DONE"):
        """Ferme une mission robot et libère les ressources."""
        if canonical_id in self._active_missions:
            m = self._active_missions.pop(canonical_id)
            m.status = status
            if m.table_id in self._table_missions:
                del self._table_missions[m.table_id]
            log.info(f"[MISSION] Clôture #{canonical_id} "
                     f"status={status} durée={m.age_s:.1f}s")

    # ─────────────────────────────────────────────────────────
    #  NOTIFICATION EXTERNE : robot arrivé
    # ─────────────────────────────────────────────────────────

    def notify_robot_arrived(self, canonical_id: int):
        """
        À appeler depuis le nœud ROS2 quand le robot
        signale qu'il est arrivé à la table.
        Met à jour la mission et notifie la StateMachine
        via le CustomerRecord (robot_arrived=True sera lu
        par FrameContext au prochain tick de StateMachine).
        """
        if canonical_id in self._active_missions:
            self._active_missions[canonical_id].status = "ACTIVE"
            log.info(f"[ROBOT] Arrivé à destination — client #{canonical_id}")

    def is_robot_at_table(self, table_id: Optional[int]) -> bool:
        """
        Retourne True si un robot est actif sur cette table.
        Utilisé par FrameContext pour remplir robot_arrived.
        """
        if table_id is None:
            return False
        cid = self._table_missions.get(table_id)
        if cid is None:
            return False
        m = self._active_missions.get(cid)
        return m is not None and m.status == "ACTIVE"

    # ─────────────────────────────────────────────────────────
    #  COOLDOWN
    # ─────────────────────────────────────────────────────────

    def _can_act(self, canonical_id: int,
                 action_type: ActionType,
                 cooldown_s: Optional[float] = None) -> bool:
        """
        Vérifie si l'action peut être exécutée (cooldown respecté).
        """
        if cooldown_s is None:
            # Choisir le cooldown par défaut selon le type
            defaults = {
                ActionType.SEND_ROBOT         : self._cfg.cooldown_robot_s,
                ActionType.PLAY_WELCOME       : self._cfg.cooldown_welcome_s,
                ActionType.PLAY_SEATED        : self._cfg.cooldown_welcome_s,
                ActionType.PLAY_SERVED        : self._cfg.cooldown_welcome_s,
                ActionType.ALERT_WAITING_LONG : self._cfg.cooldown_alert_s,
                ActionType.ALERT_TABLE_FREE   : 5.0,
            }
            cooldown_s = defaults.get(action_type, 0.0)

        key      = (canonical_id, action_type)
        last_t   = self._last_action_t[key]
        elapsed  = time.time() - last_t
        return elapsed >= cooldown_s

    # ─────────────────────────────────────────────────────────
    #  EXÉCUTION D'UNE ACTION
    # ─────────────────────────────────────────────────────────

    def _execute(self, action: Action) -> Action:
        """
        Enregistre l'action, met à jour le cooldown,
        appelle le callback optionnel, logue.
        """
        key = (action.canonical_id, action.action_type)
        self._last_action_t[key] = time.time()
        self._action_log.append(action)

        log.info(str(action))

        if self._on_action is not None:
            try:
                self._on_action(action)
            except Exception as e:
                log.warning(f"[CALLBACK] Erreur on_action : {e}")

        return action

    # ─────────────────────────────────────────────────────────
    #  EXPORT ET STATS
    # ─────────────────────────────────────────────────────────

    def active_missions_summary(self) -> List[dict]:
        """Résumé des missions robot actives."""
        return [m.to_dict() for m in self._active_missions.values()]

    def action_log_summary(self, last_n: int = 20) -> List[dict]:
        """Les N dernières actions exécutées."""
        return [
            {
                "type"         : a.action_type.value,
                "canonical_id" : a.canonical_id,
                "table_id"     : a.table_id,
                "priority"     : round(a.priority, 2),
                "t"            : time.strftime(
                    "%H:%M:%S", time.localtime(a.timestamp)),
            }
            for a in self._action_log[-last_n:]
        ]

    def export_log(self, path: Optional[str] = None) -> str:
        """Exporte le log complet des actions en JSON."""
        if path is None:
            path = str(self._output_dir / "orchestrator_log.json")
        data = {
            "total_actions"   : len(self._action_log),
            "actions"         : [
                {
                    "type"    : a.action_type.value,
                    "cid"     : a.canonical_id,
                    "table"   : a.table_id,
                    "priority": round(a.priority, 2),
                    "payload" : a.payload,
                    "t"       : time.strftime(
                        "%H:%M:%S", time.localtime(a.timestamp)),
                }
                for a in self._action_log
            ]
        }
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        return path

    def print_summary(self):
        """Affiche un résumé console."""
        print("\n" + "=" * 65)
        print("  ORCHESTRATOR — RÉSUMÉ")
        print("=" * 65)
        print(f"  Actions exécutées  : {len(self._action_log)}")
        print(f"  Missions actives   : {len(self._active_missions)}")
        print(f"\n  Missions :")
        for m in self._active_missions.values():
            print(f"    #{m.canonical_id:<4} table={m.table_id} "
                  f"status={m.status} age={m.age_s:.1f}s "
                  f"priority={m.priority:.1f}")
        print(f"\n  Dernières actions :")
        for a in self.action_log_summary(10):
            print(f"    [{a['t']}] {a['type']:<22} client=#{a['canonical_id']}"
                  f" table={a['table_id']}")
        print("=" * 65)


# ─────────────────────────────────────────────────────────────
#  BOUCLE PRINCIPALE COMPLÈTE (exemple d'intégration)
# ─────────────────────────────────────────────────────────────
#
#  from customer_manager import CustomerManager
#  from state_machine    import ClientStateMachine, FrameContext, Statut
#  from orchestrator     import Orchestrator, OrchestratorConfig
#
#  cm   = CustomerManager(fps=FPS, output_dir=OUTPUT_DIR)
#  orch = Orchestrator(ros2_node=None, output_dir=OUTPUT_DIR)
#  state_machines: Dict[int, ClientStateMachine] = {}
#
#  while True:
#      ret, frame = cap.read()
#      frame_idx += 1
#
#      cm.mark_all_invisible()
#
#      results = yolo.track(frame, tracker="bytetrack.yaml",
#                           conf=0.35, classes=[0,1], persist=True)
#
#      tables_detected = extract_tables(results)
#
#      for each person detected:
#          cid, dist, is_new = reid.process(frame, bbox, byte_id, frame_idx)
#          kpts  = pose_model.infer(crop_person(frame, bbox))
#          pose  = analyze_pose_state(kpts)
#
#          record = cm.update(
#              canonical_id=cid, byte_id=byte_id,
#              bbox=bbox, conf=conf, frame_idx=frame_idx,
#              keypoints=kpts, pose_state=pose,
#              tables=tables_detected, line_y=line_y,
#              reid_dist=dist, is_reid_known=not is_new,
#              frame_shape=frame.shape,
#          )
#
#          if cid not in state_machines:
#              state_machines[cid] = ClientStateMachine()
#
#          ctx = FrameContext(
#              frame_idx     = frame_idx,
#              fps           = FPS,
#              pose_state    = record.pose_state,
#              keypoints     = record.pose.keypoints,
#              position      = record.position,
#              table_id      = record.table_id,
#              table_dist_px = record.table.table_dist_px,
#              reid_distance = record.reid_distance,
#              is_reid_known = record.is_reid_known,
#              robot_arrived = orch.is_robot_at_table(record.table_id),
#              visible       = record.visible,
#          )
#          new_statut = state_machines[cid].update(ctx)
#          record.statut      = new_statut.value
#          record.statut_extra= state_machines[cid].to_dict()
#
#          orch.process(record, state_machines[cid])
#
#      orch.tick(cm.all_records())
#
#      if frame_idx % 30 == 0:
#          cm.export_csv_snapshot(csv_writer, frame_idx)
#
#  orch.export_log()
#  cm.export_json_final()


# ─────────────────────────────────────────────────────────────
#  TEST STANDALONE
# ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys
    sys.path.insert(0, str(Path(__file__).parent))

    # Import des modules du projet
    try:
        from customer_manager import CustomerManager, CustomerRecord
        from state_machine    import ClientStateMachine, FrameContext, Statut
    except ImportError:
        print("[TEST] Modules customer_manager / state_machine non trouvés.")
        print("       Place ce fichier dans le même dossier que les modules.")
        sys.exit(1)

    print("=" * 65)
    print("  ORCHESTRATOR — Test standalone (simulation)")
    print("=" * 65)

    # Callback de test : affiche chaque action
    def on_action(action: Action):
        print(f"  >>> ACTION : {action}")

    cfg  = OrchestratorConfig(
        cooldown_robot_s   = 2.0,    # court pour le test
        alert_waiting_s    = 10.0,
    )
    cm   = CustomerManager(fps=25.0, output_dir="test_output")
    orch = Orchestrator(config=cfg, output_dir="test_output",
                        on_action=on_action)
    sms  : Dict[int, ClientStateMachine] = {}

    tables = [{"id": 1, "cx": 300, "cy": 400}]

    # ── Scénario client #1 ────────────────────────────────────
    print("\n[SIM] Client #1 entre dans le restaurant...")

    scenarios = [
        # (pose, position, table_id, robot, frames, label)
        ("DEBOUT", (100, 200), None,  False, 5,  "NOUVEAU"),
        ("DEBOUT", (200, 250), None,  False, 5,  "CHERCHE_TABLE"),
        ("ASSIS",  (290, 390), 1,     False, 8,  "ASSIS"),
        ("ASSIS",  (290, 390), 1,     False, 5,  "EN_ATTENTE"),
        ("ASSIS",  (290, 390), 1,     True,  3,  "ROBOT_ARRIVE"),
        ("ASSIS",  (290, 390), 1,     False, 4,  "SERVI"),
        ("?",      (290, 390), None,  False, 10, "PARTI"),
    ]

    frame_idx = 0
    for pose, pos, table_id, robot_arrived, n_frames, label in scenarios:
        print(f"\n  --- {label} ---")
        for _ in range(n_frames):
            frame_idx += 1
            cm.mark_all_invisible()

            tbl = [{"id": 1, "cx": 300, "cy": 400}] if table_id else []
            bbox = (pos[0]-30, pos[1]-50, pos[0]+30, pos[1]+50)
            visible = pose != "?"

            if visible:
                record = cm.update(
                    canonical_id = 1, byte_id = 10,
                    bbox         = bbox, conf = 0.88,
                    frame_idx    = frame_idx,
                    pose_state   = pose,
                    tables       = tbl,
                    line_y       = 250,
                    reid_dist    = 0.12,
                    is_reid_known= frame_idx > 1,
                    frame_shape  = (480, 640, 3),
                )
            else:
                record = cm.get(1)
                if record:
                    record.visible = False

            if record is None:
                continue

            if 1 not in sms:
                sms[1] = ClientStateMachine()

            if robot_arrived:
                orch.notify_robot_arrived(1)

            ctx = FrameContext(
                frame_idx     = frame_idx,
                fps           = 25.0,
                pose_state    = pose,
                position      = pos,
                table_id      = table_id,
                table_dist_px = 80.0 if table_id else 9999.0,
                reid_distance = 0.12,
                is_reid_known = frame_idx > 1,
                robot_arrived = orch.is_robot_at_table(table_id),
                visible       = visible,
            )
            new_statut = sms[1].update(ctx)
            record.statut = new_statut.value

            orch.process(record, sms[1])

        time.sleep(0.05)

    orch.tick()
    orch.print_summary()
    cm.print_summary()

    log_path = orch.export_log()
    print(f"\n[TEST] Log exporté → {log_path}")
    print("\n✅ Orchestrator OK")
