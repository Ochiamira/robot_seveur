"""
NEXOR Vision — Machine à États Client (v2)
==========================================
6 statuts précis avec déclencheurs multi-modaux :
  durée + pose RTMPose + position spatiale + Re-ID

Statuts :
  NOUVEAU       → vient de franchir la ROI d'entrée, Re-ID inconnu
  CHERCHE_TABLE → debout, se déplace, regarde autour
  ASSIS         → immobile, pose=assis, table assignée
  EN_ATTENTE    → assis depuis N secondes → déclenche le robot
  SERVI         → robot a interagi
  PARTI         → invisible depuis N secondes

Intégration :
  from state_machine import ClientStateMachine, Statut
  sm = ClientStateMachine()
  # à chaque frame :
  nouveau_statut = sm.update(record, frame_context)
"""

import time
from dataclasses import dataclass, field
from typing import Optional, Tuple
from enum import Enum


# ─────────────────────────────────────────────────────────────
#  STATUTS
# ─────────────────────────────────────────────────────────────

class Statut(str, Enum):
    NOUVEAU        = "NOUVEAU"
    CHERCHE_TABLE  = "CHERCHE_TABLE"
    ASSIS          = "ASSIS"
    EN_ATTENTE     = "EN_ATTENTE"
    SERVI          = "SERVI"
    PARTI          = "PARTI"


# ─────────────────────────────────────────────────────────────
#  PARAMÈTRES DE TRANSITION (tous modifiables)
# ─────────────────────────────────────────────────────────────

class Config:
    # Fallback when the camera first acquires a customer already seated.
    NOUVEAU_TO_ASSIS_S     = 3.0

    # ── Seuils temporels (secondes) ──────────────────────────
    NOUVEAU_TO_CHERCHE_S   = 2.0   # debout + mobile depuis N s → CHERCHE_TABLE
    CHERCHE_TO_ASSIS_S     = 3.0   # pose=ASSIS + immobile depuis N s → ASSIS
    ASSIS_TO_ATTENTE_S     = 5.0   # assis depuis N s sans robot → EN_ATTENTE
    ATTENTE_TIMEOUT_S      = 60.0  # EN_ATTENTE max avant alerte secondaire
    INVISIBLE_TO_PARTI_S   = 8.0   # client debout absent depuis N s → PARTI
    # Une personne attablée est souvent masquée par la chaise ou la table.
    # Elle ne doit donc pas être déclarée partie sur une courte perte YOLO.
    SEATED_INVISIBLE_TO_PARTI_S = 30.0

    # ── Seuils de mouvement ───────────────────────────────────
    MOUVEMENT_PX_THRESHOLD = 15    # déplacement (px) considéré comme mouvement
    IMMOBILE_WINDOW_S      = 2.0   # durée fenêtre pour évaluer l'immobilité

    # ── Seuils de pose RTMPose ────────────────────────────────
    KPT_CONF_MIN      = 0.3        # confiance keypoint minimale
    ANGLE_GENOU_ASSIS = 120        # angle genou < N° → assis
    ANGLE_GENOU_DEBOUT= 150        # angle genou > N° → debout

    # ── Table assignée ────────────────────────────────────────
    TABLE_DIST_MAX_PX = 200        # distance max pour attribuer une table

    # ── Re-ID ─────────────────────────────────────────────────
    REID_SAME_THRESH  = 0.30       # distance cosinus < seuil → même personne


# ─────────────────────────────────────────────────────────────
#  CONTEXTE FRAME — données fournies à la machine à états
# ─────────────────────────────────────────────────────────────

@dataclass
class FrameContext:
    """
    Toutes les informations disponibles à une frame donnée
    pour évaluer les transitions d'un PersonRecord.
    """
    frame_idx       : int   = 0
    fps             : float = 25.0
    timestamp_s     : Optional[float] = None  # temps source, sinon horloge réelle

    # Pose RTMPose
    pose_state      : str   = "?"       # "DEBOUT" / "ASSIS" / "MAIN_LEVEE" / "?"
    keypoints       : object = None     # np.ndarray (17,3) ou None

    # Spatial
    position        : Tuple = (0, 0)   # (cx, cy) actuel
    table_id        : Optional[int] = None
    table_dist_px   : float = 9999.0

    # Re-ID
    reid_distance   : float = 1.0      # distance cosinus du dernier match
    is_reid_known   : bool  = False    # True si Re-ID a reconnu la personne

    # Robot
    robot_arrived   : bool  = False    # True si le robot est arrivé à la table

    # Visibilité
    visible         : bool  = True


# ─────────────────────────────────────────────────────────────
#  HISTORIQUE DE MOUVEMENT
# ─────────────────────────────────────────────────────────────

class MovementHistory:
    """
    Fenêtre glissante de positions pour détecter l'immobilité.
    """
    def __init__(self, window_s: float = Config.IMMOBILE_WINDOW_S,
                 fps: float = 25.0):
        self._positions = []
        self._max_len   = max(1, int(window_s * fps))

    def push(self, cx: int, cy: int):
        self._positions.append((cx, cy))
        if len(self._positions) > self._max_len:
            self._positions.pop(0)

    def is_mobile(self) -> bool:
        """True si la personne s'est déplacée de plus de THRESHOLD px."""
        if len(self._positions) < 2:
            return False
        xs = [p[0] for p in self._positions]
        ys = [p[1] for p in self._positions]
        spread = max(max(xs)-min(xs), max(ys)-min(ys))
        return spread > Config.MOUVEMENT_PX_THRESHOLD

    def is_stationary(self) -> bool:
        return not self.is_mobile()


# ─────────────────────────────────────────────────────────────
#  MACHINE À ÉTATS
# ─────────────────────────────────────────────────────────────

class ClientStateMachine:
    """
    Gère les transitions d'un seul client.
    Une instance par PersonRecord.

    Usage :
        sm = ClientStateMachine()
        statut = sm.update(ctx)   # ctx = FrameContext
    """

    def __init__(self, fps: float = 25.0):
        self.statut          : Statut = Statut.NOUVEAU
        self._now_s          : float  = 0.0
        self._since          : float  = 0.0
        self._movement       : MovementHistory = MovementHistory(fps=fps)
        self._pose_since     : dict   = {}            # pose → horloge première détection
        self._invisible_since: Optional[float] = None
        self._robot_notified : bool   = False
        self._log            : list   = []            # historique des transitions

    # ── Propriétés utiles ─────────────────────────────────────────────────────

    @property
    def time_in_statut(self) -> float:
        """Secondes passées dans le statut actuel."""
        return max(0.0, self._now_s - self._since)

    def _transition(self, new_statut: Statut, reason: str):
        """Effectue une transition et log."""
        old = self.statut
        self.statut  = new_statut
        self._since  = self._now_s
        entry = {
            "from"   : old.value,
            "to"     : new_statut.value,
            "reason" : reason,
            "t"      : time.strftime("%H:%M:%S"),
        }
        self._log.append(entry)
        print(f"[SM] {old.value:15} → {new_statut.value:15}  ({reason})")

    # ── Helpers de conditions ─────────────────────────────────────────────────

    def _pose_stable_since(self, pose: str, ctx: FrameContext,
                           duration_s: float) -> bool:
        """
        True si la pose `pose` est détectée de façon stable
        depuis au moins `duration_s` secondes.
        """
        if ctx.pose_state != pose:
            self._pose_since.pop(pose, None)
            return False
        if pose not in self._pose_since:
            self._pose_since[pose] = self._now_s
        return (self._now_s - self._pose_since[pose]) >= duration_s

    def _main_levee(self, ctx: FrameContext) -> bool:
        """Détecte main levée via keypoints RTMPose."""
        if ctx.pose_state == "MAIN_LEVEE":
            return True
        # Vérification directe sur les keypoints si disponibles
        if ctx.keypoints is not None:
            try:
                import numpy as np
                kpts = ctx.keypoints
                # poignet (9 ou 10) au-dessus de l'épaule (5 ou 6)
                for poignet, epaule in [(9, 5), (10, 6)]:
                    if (kpts[poignet, 2] >= Config.KPT_CONF_MIN and
                            kpts[epaule, 2] >= Config.KPT_CONF_MIN):
                        if kpts[poignet, 1] < kpts[epaule, 1]:
                            return True
            except Exception:
                pass
        return False

    # ── MISE À JOUR PRINCIPALE ────────────────────────────────────────────────

    def update(self, ctx: FrameContext) -> Statut:
        """
        Évalue toutes les conditions de transition et met à jour le statut.
        Appeler une fois par frame pour chaque personne.

        Retourne le statut courant (potentiellement mis à jour).
        """
        self._now_s = (float(ctx.timestamp_s) if ctx.timestamp_s is not None
                       else time.monotonic())
        if self._since == 0.0:
            self._since = self._now_s

        # Une position absente ne doit pas contaminer l'historique mouvement.
        if ctx.visible:
            self._movement.push(*ctx.position)

        # ── Si invisible depuis trop longtemps → PARTI ────────────────────────
        if not ctx.visible:
            if self._invisible_since is None:
                self._invisible_since = self._now_s
            departure_timeout_s = (
                Config.SEATED_INVISIBLE_TO_PARTI_S
                if self.statut in (Statut.ASSIS, Statut.EN_ATTENTE, Statut.SERVI)
                else Config.INVISIBLE_TO_PARTI_S
            )
            if self._now_s - self._invisible_since >= departure_timeout_s:
                if self.statut != Statut.PARTI:
                    self._transition(
                        Statut.PARTI,
                        f"invisible depuis {departure_timeout_s}s",
                    )
            return self.statut
        self._invisible_since = None

        # ── Machine à états ───────────────────────────────────────────────────
        s = self.statut

        # ┌─────────────────────────────────────────────────────────┐
        # │  NOUVEAU                                                │
        # │  Déclencheurs de sortie :                               │
        # │    A. pose=DEBOUT + mobile depuis N s → CHERCHE_TABLE   │
        # │    B. main levée → EN_ATTENTE (urgence)                 │
        # └─────────────────────────────────────────────────────────┘
        if s == Statut.NOUVEAU:
            if self._main_levee(ctx):
                self._transition(Statut.EN_ATTENTE,
                                 "main levée détectée dès l'entrée")

            elif (self._pose_stable_since("ASSIS", ctx,
                                          Config.NOUVEAU_TO_ASSIS_S)
                  and self._movement.is_stationary()
                  and ctx.table_id is not None):
                self._transition(Statut.ASSIS,
                                 f"premiere observation assise a T{ctx.table_id}")

            elif (self._pose_stable_since("DEBOUT", ctx,
                                          Config.NOUVEAU_TO_CHERCHE_S)
                  and self._movement.is_mobile()):
                self._transition(Statut.CHERCHE_TABLE,
                                 f"debout+mobile depuis {Config.NOUVEAU_TO_CHERCHE_S}s")

        # ┌─────────────────────────────────────────────────────────┐
        # │  CHERCHE_TABLE                                          │
        # │  Déclencheurs de sortie :                               │
        # │    A. pose=ASSIS + immobile + table proche → ASSIS      │
        # │    B. main levée → EN_ATTENTE                           │
        # └─────────────────────────────────────────────────────────┘
        elif s == Statut.CHERCHE_TABLE:
            if self._main_levee(ctx):
                self._transition(Statut.EN_ATTENTE,
                                 "main levée pendant recherche table")

            elif (self._pose_stable_since("ASSIS", ctx,
                                          Config.CHERCHE_TO_ASSIS_S)
                  and self._movement.is_stationary()
                  and ctx.table_id is not None):
                self._transition(Statut.ASSIS,
                                 f"assis+immobile+table T{ctx.table_id}")

        # ┌─────────────────────────────────────────────────────────┐
        # │  ASSIS                                                  │
        # │  Déclencheurs de sortie :                               │
        # │    A. main levée → EN_ATTENTE (court-circuit)           │
        # │    B. assis depuis N s → EN_ATTENTE                     │
        # │    C. se lève + bouge → CHERCHE_TABLE (changement table)│
        # └─────────────────────────────────────────────────────────┘
        elif s == Statut.ASSIS:
            if self._main_levee(ctx):
                self._transition(Statut.EN_ATTENTE,
                                 "main levée — client appelle")

            elif self.time_in_statut >= Config.ASSIS_TO_ATTENTE_S:
                self._transition(Statut.EN_ATTENTE,
                                 f"assis depuis {self.time_in_statut:.1f}s")

            elif (ctx.pose_state == "DEBOUT"
                  and self._movement.is_mobile()
                  and not self._movement.is_stationary()):   # FIX: double is_mobile() → is_mobile() + not is_stationary()
                self._transition(Statut.CHERCHE_TABLE,
                                 "s'est levé — cherche une autre table ?")

        # ┌─────────────────────────────────────────────────────────┐
        # │  EN_ATTENTE                                             │
        # │  Déclencheurs de sortie :                               │
        # │    A. robot arrivé → SERVI                              │
        # │    B. timeout → log alerte (reste EN_ATTENTE)           │
        # └─────────────────────────────────────────────────────────┘
        elif s == Statut.EN_ATTENTE:
            if ctx.robot_arrived:
                self._transition(Statut.SERVI,
                                 "robot arrivé à la table")

            elif (self.time_in_statut >= Config.ATTENTE_TIMEOUT_S
                  and not self._robot_notified):
                self._robot_notified = True
                print(f"[SM][ALERTE] Client EN_ATTENTE depuis "
                      f"{self.time_in_statut:.0f}s — robot non arrivé !")

        # ┌─────────────────────────────────────────────────────────┐
        # │  SERVI                                                  │
        # │  Déclencheurs de sortie :                               │
        # │    A. main levée → EN_ATTENTE (commande supplémentaire) │
        # │    B. invisible → PARTI                                 │
        # └─────────────────────────────────────────────────────────┘
        elif s == Statut.SERVI:
            if self._main_levee(ctx):
                self._transition(Statut.EN_ATTENTE,
                                 "client appelle à nouveau après service")

        # ┌─────────────────────────────────────────────────────────┐
        # │  PARTI                                                  │
        # │  Peut revenir si Re-ID le reconnaît                     │
        # └─────────────────────────────────────────────────────────┘
        elif s == Statut.PARTI:
            if ctx.visible and ctx.is_reid_known:
                # Le client est revenu (Re-ID l'a reconnu)
                # → on le remet à ASSIS directement si une table est assignée
                if ctx.table_id is not None:
                    self._transition(Statut.ASSIS,
                                     "Re-ID reconnu + table disponible")
                else:
                    self._transition(Statut.CHERCHE_TABLE,
                                     "Re-ID reconnu — cherche sa table")

        return self.statut

    # ── DÉCISION ROBOT ────────────────────────────────────────────────────────

    def should_send_robot(self) -> bool:
        """
        True si le robot doit être envoyé vers ce client.
        Appelé par le nœud ROS2 de navigation.
        """
        return self.statut == Statut.EN_ATTENTE

    def robot_priority(self) -> float:
        """
        Score de priorité pour l'ordonnancement du robot (plusieurs clients).
        Plus le score est élevé, plus le client est prioritaire.
        """
        if self.statut != Statut.EN_ATTENTE:
            return 0.0
        # Plus le client attend longtemps, plus la priorité est haute
        return min(self.time_in_statut / 10.0, 10.0)

    # ── EXPORT ───────────────────────────────────────────────────────────────

    def to_dict(self) -> dict:
        return {
            "statut"         : self.statut.value,
            "time_in_statut" : round(self.time_in_statut, 2),
            "send_robot"     : self.should_send_robot(),
            "priority"       : round(self.robot_priority(), 2),
            "history"        : self._log[-5:],   # 5 dernières transitions
        }


# ─────────────────────────────────────────────────────────────
#  COULEURS ET LABELS UI
# ─────────────────────────────────────────────────────────────

# Couleurs BGR pour OpenCV
STATUT_BGR = {
    Statut.NOUVEAU       : (180, 100, 220),   # violet
    Statut.CHERCHE_TABLE : (100, 200, 100),   # vert
    Statut.ASSIS         : (60,  180, 240),   # ambre/cyan
    Statut.EN_ATTENTE    : (50,  100, 220),   # rouge-orange
    Statut.SERVI         : (200, 180,  50),   # bleu
    Statut.PARTI         : (140, 140, 140),   # gris
}

# Emojis pour le tableau OpenCV
STATUT_ICON = {
    Statut.NOUVEAU       : ">>",
    Statut.CHERCHE_TABLE : "?T",
    Statut.ASSIS         : "[]",
    Statut.EN_ATTENTE    : "!!",
    Statut.SERVI         : "OK",
    Statut.PARTI         : "--",
}


# ─────────────────────────────────────────────────────────────
#  INTÉGRATION DANS PersonRecord
# ─────────────────────────────────────────────────────────────
#
#  Dans rtmpose_tracker.py, ajouter à PersonRecord :
#
#    from state_machine import ClientStateMachine, Statut, FrameContext
#
#    @dataclass
#    class PersonRecord:
#        ...
#        state_machine : ClientStateMachine = field(
#                            default_factory=ClientStateMachine)
#        # Supprimer l'ancien champ `statut : str`
#        # et remplacer par :
#        @property
#        def statut(self):
#            return self.state_machine.statut.value
#
#  Puis dans la boucle principale, après update_position() :
#
#    ctx = FrameContext(
#        frame_idx      = frame_idx,
#        fps            = FPS,
#        pose_state     = rec.pose_state,
#        keypoints      = rec.keypoints,
#        position       = rec.position,
#        table_id       = rec.table_id,
#        table_dist_px  = rec.table_dist_px,
#        reid_distance  = reid_dist,
#        is_reid_known  = not is_new,
#        robot_arrived  = check_robot_arrived(rec.table_id),
#        visible        = rec.visible,
#    )
#    rec.state_machine.update(ctx)
#
#  Pour savoir si envoyer le robot :
#    if rec.state_machine.should_send_robot():
#        publish_to_ros2(rec.table_id, rec.position,
#                        rec.state_machine.robot_priority())
# ─────────────────────────────────────────────────────────────


# ─────────────────────────────────────────────────────────────
#  TEST STANDALONE — simulation sans caméra
# ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import time

    print("=" * 55)
    print("  TEST Machine à états — simulation client #1")
    print("=" * 55)

    sm = ClientStateMachine()

    def sim_step(pose, position, visible=True, table_id=None,
                 main_levee=False, robot=False, label=""):
        ctx = FrameContext(
            frame_idx    = 0,
            pose_state   = "MAIN_LEVEE" if main_levee else pose,
            position     = position,
            table_id     = table_id,
            visible      = visible,
            robot_arrived= robot,
        )
        ancien = sm.statut
        sm.update(ctx)
        if sm.statut != ancien or label:
            print(f"  [{label:25}] statut={sm.statut.value:<15} "
                  f"depuis={sm.time_in_statut:.1f}s "
                  f"robot={sm.should_send_robot()}")

    # ── Scénario : client entre, cherche table, s'assoit, attend, est servi ──

    print("\n[SIM] Client entre dans le restaurant...")
    for _ in range(5):
        sim_step("DEBOUT", (100, 200), label="entrée")
        time.sleep(0.5)

    print("\n[SIM] Client cherche une table, se déplace...")
    for i in range(6):
        sim_step("DEBOUT", (100 + i*10, 200 + i*5), label="déplacement")
        time.sleep(0.5)

    print("\n[SIM] Client s'assoit à une table...")
    for _ in range(7):
        sim_step("ASSIS", (250, 310), table_id=3, label="assis")
        time.sleep(0.5)

    print("\n[SIM] Client attend → robot doit être envoyé...")
    for _ in range(3):
        sim_step("ASSIS", (250, 310), table_id=3, label="en_attente")
        print(f"         priority={sm.robot_priority():.2f}")
        time.sleep(0.5)

    print("\n[SIM] Robot arrive à la table...")
    sim_step("ASSIS", (250, 310), table_id=3, robot=True, label="robot_arrivé")

    print("\n[SIM] Client lève la main (2ème commande)...")
    time.sleep(0.5)
    sim_step("ASSIS", (250, 310), main_levee=True, label="main_levée")

    print("\n[SIM] Robot revient → client servi...")
    time.sleep(0.5)
    sim_step("ASSIS", (250, 310), robot=True, label="servi_2")

    print("\n[SIM] Client quitte le restaurant...")
    for _ in range(10):
        sim_step("?", (250, 310), visible=False, label="absent")
        time.sleep(1.0)

    print("\n" + "="*55)
    print("  HISTORIQUE DES TRANSITIONS")
    print("="*55)
    for entry in sm._log:
        print(f"  {entry['t']}  {entry['from']:15} → {entry['to']:15}  ({entry['reason']})")
    print("="*55)
