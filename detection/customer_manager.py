"""
=============================================================
  NEXOR Vision — CustomerManager  (10-customer_manager.py)
  ─────────────────────────────────────────────────────────
  Rôle UNIQUE : centraliser et maintenir toutes les
  informations de chaque client.

  ❌ Ne prend AUCUNE décision
  ❌ Ne fait AUCUNE transition d'état
  ❌ N'envoie AUCUN signal ROS2 / robot

  ✅ Stocke les données brutes (ByteTrack, ReID, RTMPose,
     table, visibilité, temporel, mouvement)
  ✅ Fournit des accesseurs lisibles pour StateMachine
     et Orchestrator
  ✅ Gère le cycle de vie des enregistrements (création,
     mise à jour, archivage)
  ✅ Export CSV / JSON à la demande

  Intégration dans la boucle principale :
    cm = CustomerManager(fps=FPS)
    # à chaque frame :
    record = cm.update(
        byte_id      = tid,
        canonical_id = cid,          # depuis ReIDTracker
        bbox         = (x1,y1,x2,y2),
        conf         = confidence,
        keypoints    = kpts_global,   # depuis RTMPose (ou None)
        pose_state   = "ASSIS",       # depuis analyze_pose_state
        tables       = tables_list,   # liste de dicts {id,cx,cy}
        frame_idx    = frame_idx,
        frame        = frame,         # pour Re-ID si besoin
        reid_dist    = dist,
        is_reid_known= not is_new,
    )
    # La StateMachine lit record et décide le statut
    # L'Orchestrator lit record et décide les actions
=============================================================
"""

import time
import math
import json
import csv
import numpy as np
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Tuple, Any
from collections import deque


# ─────────────────────────────────────────────────────────────
#  CONSTANTES INTERNES (données brutes — pas de décision)
# ─────────────────────────────────────────────────────────────

# Distance max (px) pour attribuer une table à un client
TABLE_DIST_MAX_PX = 200
LINE_HYSTERESIS_PX = 15

# Longueur max de la traînée de mouvement (trail)
TRAIL_MAX_LEN = 40

# Fenêtre glissante pour le calcul de vitesse (frames)
SPEED_WINDOW = 10

# Longueur de l'historique de pose (pour détection de stabilité)
POSE_HISTORY_LEN = 15

# Durée max (s) d'absence avant marquage "perdu" (info brute, pas décision)
ABSENCE_LOST_S = 8.0


# ─────────────────────────────────────────────────────────────
#  DONNÉES BRUTES DE MOUVEMENT
# ─────────────────────────────────────────────────────────────

@dataclass
class MovementData:
    """
    Données brutes de mouvement — fenêtre glissante de positions.
    Fournit des métriques, ne décide rien.
    """
    _positions : deque = field(default_factory=lambda: deque(maxlen=TRAIL_MAX_LEN))
    _speed_win : deque = field(default_factory=lambda: deque(maxlen=SPEED_WINDOW))

    def push(self, cx: int, cy: int):
        if self._positions:
            px, py = self._positions[-1]
            dist = math.sqrt((cx - px)**2 + (cy - py)**2)
            self._speed_win.append(dist)
        self._positions.append((cx, cy))

    @property
    def trail(self) -> List[Tuple[int, int]]:
        """Liste de positions pour dessiner la traînée."""
        return list(self._positions)

    @property
    def last_position(self) -> Optional[Tuple[int, int]]:
        return self._positions[-1] if self._positions else None

    @property
    def displacement_px(self) -> float:
        """Déplacement max dans la fenêtre courante (pixels)."""
        if len(self._positions) < 2:
            return 0.0
        xs = [p[0] for p in self._positions]
        ys = [p[1] for p in self._positions]
        return float(max(max(xs) - min(xs), max(ys) - min(ys)))

    @property
    def speed_px_per_frame(self) -> float:
        """Vitesse moyenne de déplacement (px/frame)."""
        if not self._speed_win:
            return 0.0
        return float(sum(self._speed_win) / len(self._speed_win))

    def to_dict(self) -> dict:
        return {
            "displacement_px"     : round(self.displacement_px, 2),
            "speed_px_per_frame"  : round(self.speed_px_per_frame, 2),
            "trail_length"        : len(self._positions),
        }


# ─────────────────────────────────────────────────────────────
#  DONNÉES BRUTES DE POSE
# ─────────────────────────────────────────────────────────────

@dataclass
class PoseData:
    """
    Données brutes de pose RTMPose.
    Stocke les keypoints et l'historique des états détectés.
    """
    keypoints      : Optional[np.ndarray] = None   # (17, 3) : [x, y, conf]
    pose_state     : str = "?"                     # "DEBOUT" / "ASSIS" / "MAIN_LEVEE" / "?"
    _history       : deque = field(
                        default_factory=lambda: deque(maxlen=POSE_HISTORY_LEN))

    def update(self, keypoints: Optional[np.ndarray], pose_state: str):
        self.keypoints  = keypoints
        self.pose_state = pose_state
        self._history.append(pose_state)

    @property
    def pose_history(self) -> List[str]:
        return list(self._history)

    @property
    def dominant_pose(self) -> str:
        """Pose la plus fréquente dans la fenêtre historique."""
        if not self._history:
            return "?"
        from collections import Counter
        return Counter(self._history).most_common(1)[0][0]

    @property
    def pose_stability(self) -> float:
        """
        Score de stabilité de la pose [0.0 – 1.0].
        1.0 = même pose sur toute la fenêtre.
        """
        if not self._history:
            return 0.0
        from collections import Counter
        most_common_count = Counter(self._history).most_common(1)[0][1]
        return most_common_count / len(self._history)

    def has_keypoint(self, idx: int, min_conf: float = 0.3) -> bool:
        """True si le keypoint `idx` est détecté avec suffisamment de confiance."""
        if self.keypoints is None or idx >= len(self.keypoints):
            return False
        return float(self.keypoints[idx, 2]) >= min_conf

    def get_keypoint_xy(self, idx: int) -> Optional[Tuple[float, float]]:
        """Retourne (x, y) du keypoint ou None si absent/peu confiant."""
        if not self.has_keypoint(idx):
            return None
        return float(self.keypoints[idx, 0]), float(self.keypoints[idx, 1])

    def to_dict(self) -> dict:
        return {
            "pose_state"    : self.pose_state,
            "dominant_pose" : self.dominant_pose,
            "stability"     : round(self.pose_stability, 3),
            "has_keypoints" : self.keypoints is not None,
        }


# ─────────────────────────────────────────────────────────────
#  DONNÉES BRUTES DE RE-ID
# ─────────────────────────────────────────────────────────────

@dataclass
class ReIDData:
    """
    Données brutes de ré-identification.
    Stocke l'ID canonique stable et les métriques de matching.
    """
    byte_track_id   : int   = -1      # ID brut ByteTrack (change si occlusion)
    canonical_id    : int   = -1      # ID stable après Re-ID
    reid_distance   : float = 1.0    # distance cosinus du dernier match
    is_known        : bool  = False   # True si Re-ID a reconnu la personne
    match_count     : int   = 0       # nombre de fois réidentifié avec succès
    new_appearance  : bool  = True    # True à la première détection

    def update(self, byte_id: int, canonical_id: int,
               distance: float, is_known: bool):
        self.byte_track_id = byte_id
        self.canonical_id  = canonical_id
        self.reid_distance = distance
        self.is_known      = is_known
        if is_known:
            self.match_count += 1
        if self.new_appearance:
            self.new_appearance = False

    def to_dict(self) -> dict:
        return {
            "byte_track_id" : self.byte_track_id,
            "canonical_id"  : self.canonical_id,
            "reid_distance" : round(self.reid_distance, 4),
            "is_known"      : self.is_known,
            "match_count"   : self.match_count,
        }


# ─────────────────────────────────────────────────────────────
#  DONNÉES BRUTES DE TABLE
# ─────────────────────────────────────────────────────────────

@dataclass
class TableData:
    """
    Données brutes de la table associée au client.
    L'association est faite par proximité spatiale.
    """
    table_id        : Optional[int] = None
    table_dist_px   : float = 9999.0
    table_cx        : Optional[int] = None
    table_cy        : Optional[int] = None
    assignment_time : Optional[float] = None   # horloge de la première association

    def update(self, table_id: Optional[int], dist_px: float,
               cx: Optional[int] = None, cy: Optional[int] = None):
        if table_id is not None and self.assignment_time is None:
            self.assignment_time = time.time()
        self.table_id      = table_id
        self.table_dist_px = dist_px
        self.table_cx      = cx
        self.table_cy      = cy

    @property
    def has_table(self) -> bool:
        return self.table_id is not None

    @property
    def time_at_table_s(self) -> float:
        """Secondes depuis l'attribution de la table."""
        if self.assignment_time is None:
            return 0.0
        return time.time() - self.assignment_time

    def to_dict(self) -> dict:
        return {
            "table_id"       : self.table_id,
            "table_dist_px"  : round(self.table_dist_px, 1),
            "time_at_table_s": round(self.time_at_table_s, 2),
        }


# ─────────────────────────────────────────────────────────────
#  DONNÉES TEMPORELLES
# ─────────────────────────────────────────────────────────────

@dataclass
class TemporalData:
    """
    Toutes les données de temps liées à la présence d'un client.
    """
    first_seen_t    : float = field(default_factory=time.time)
    last_seen_t     : float = field(default_factory=time.time)
    first_frame     : int   = 0
    last_frame      : int   = 0
    frame_count     : int   = 0    # frames où la personne a été vue

    def on_detected(self, frame_idx: int):
        self.last_seen_t  = time.time()
        self.last_frame   = frame_idx
        self.frame_count += 1

    @property
    def presence_s(self) -> float:
        """Durée totale de présence (s) depuis la première détection."""
        return self.last_seen_t - self.first_seen_t

    @property
    def absence_s(self) -> float:
        """Durée d'absence depuis la dernière détection (s)."""
        return time.time() - self.last_seen_t

    @property
    def is_lost(self) -> bool:
        """True si absent depuis trop longtemps (donnée brute)."""
        return self.absence_s > ABSENCE_LOST_S

    def to_dict(self) -> dict:
        return {
            "presence_s"  : round(self.presence_s, 2),
            "absence_s"   : round(self.absence_s, 2),
            "frame_count" : self.frame_count,
            "first_frame" : self.first_frame,
            "last_frame"  : self.last_frame,
            "is_lost"     : self.is_lost,
        }


# ─────────────────────────────────────────────────────────────
#  CUSTOMER RECORD — enregistrement complet d'un client
# ─────────────────────────────────────────────────────────────

@dataclass
class CustomerRecord:
    """
    Toutes les données brutes d'un client, groupées par domaine.

    ❌ Pas de logique de statut ici — c'est la StateMachine qui décide.
    ❌ Pas d'actions — c'est l'Orchestrator qui décide.
    ✅ Données pures, accessibles, sérialisables.

    Accès recommandé :
        record.reid.canonical_id         → ID stable
        record.pose.dominant_pose        → pose dominante
        record.table.has_table           → table assignée ?
        record.movement.displacement_px  → mobile ?
        record.temporal.presence_s       → temps de présence
        record.visible                   → dans la frame ?
        record.bbox                      → boîte de détection
        record.confidence                → confiance YOLO
        record.statut                    → REMPLI par StateMachine
    """
    # ── Identité ─────────────────────────────────────────────
    canonical_id    : int

    # ── Données brutes par domaine ────────────────────────────
    reid            : ReIDData      = field(default_factory=ReIDData)
    movement        : MovementData  = field(default_factory=MovementData)
    pose            : PoseData      = field(default_factory=PoseData)
    table           : TableData     = field(default_factory=TableData)
    temporal        : TemporalData  = field(default_factory=TemporalData)

    # ── Détection courante ────────────────────────────────────
    bbox            : Tuple[int,int,int,int] = (0, 0, 0, 0)
    confidence      : float = 0.0
    position        : Tuple[int, int] = (0, 0)  # (cx, cy)
    visible         : bool  = True

    # ── Comptage ligne virtuelle ──────────────────────────────
    line_side       : str  = ""     # "above" | "below"
    crossed_in      : bool = False
    crossed_out     : bool = False

    # ── Champ réservé StateMachine (écrit par elle, lu par Orchestrator) ──
    statut          : str  = "NOUVEAU"
    statut_extra    : Dict = field(default_factory=dict)
    # Exemple statut_extra : {"send_robot": True, "priority": 3.5}

    # ─────────────────────────────────────────────────────────
    #  MÉTHODES DE LECTURE (accesseurs pratiques)
    # ─────────────────────────────────────────────────────────

    @property
    def track_id(self) -> int:
        """Alias : ID canonical stable."""
        return self.canonical_id

    @property
    def presence_s(self) -> float:
        return self.temporal.presence_s

    @property
    def absence_s(self) -> float:
        return self.temporal.absence_s

    @property
    def is_lost(self) -> bool:
        return self.temporal.is_lost

    @property
    def pose_state(self) -> str:
        return self.pose.pose_state

    @property
    def has_table(self) -> bool:
        return self.table.has_table

    @property
    def table_id(self) -> Optional[int]:
        return self.table.table_id

    @property
    def displacement_px(self) -> float:
        return self.movement.displacement_px

    @property
    def speed_px(self) -> float:
        return self.movement.speed_px_per_frame

    @property
    def reid_distance(self) -> float:
        return self.reid.reid_distance

    @property
    def is_reid_known(self) -> bool:
        return self.reid.is_known

    # ─────────────────────────────────────────────────────────
    #  SÉRIALISATION
    # ─────────────────────────────────────────────────────────

    def to_dict(self) -> dict:
        """Sérialisation complète pour JSON/CSV."""
        return {
            "canonical_id"  : self.canonical_id,
            "statut"        : self.statut,
            "visible"       : self.visible,
            "bbox"          : list(self.bbox),
            "position"      : list(self.position),
            "confidence"    : round(self.confidence, 3),
            "crossed_in"    : self.crossed_in,
            "crossed_out"   : self.crossed_out,
            **{f"reid_{k}":v    for k,v in self.reid.to_dict().items()},
            **{f"move_{k}":v    for k,v in self.movement.to_dict().items()},
            **{f"pose_{k}":v    for k,v in self.pose.to_dict().items()},
            **{f"table_{k}":v   for k,v in self.table.to_dict().items()},
            **{f"time_{k}":v    for k,v in self.temporal.to_dict().items()},
        }

    def to_csv_row(self, frame_idx: int) -> dict:
        """Ligne CSV — une ligne par client par snapshot."""
        d = self.to_dict()
        d["frame"] = frame_idx
        return d


# ─────────────────────────────────────────────────────────────
#  CUSTOMER MANAGER
# ─────────────────────────────────────────────────────────────

class CustomerManager:
    """
    Gestionnaire central de tous les clients détectés.

    Responsabilités :
      ✅ Créer un CustomerRecord pour chaque nouveau client
      ✅ Mettre à jour les données brutes à chaque frame
      ✅ Maintenir le registre actif (visible) et archivé (parti)
      ✅ Calculer l'attribution de table par proximité
      ✅ Mettre à jour le comptage de la ligne virtuelle
      ✅ Fournir des vues filtrées (visibles, perdus, par statut...)
      ✅ Exporter CSV / JSON à la demande

      ❌ Ne décide PAS du statut (StateMachine)
      ❌ Ne décide PAS des actions (Orchestrator)

    Usage :
        cm = CustomerManager(fps=25.0, output_dir="tracking_output")

        # Dans la boucle vidéo — pour chaque personne détectée :
        record = cm.update(
            canonical_id = cid,
            byte_id      = tid,
            bbox         = (x1,y1,x2,y2),
            conf         = confidence,
            keypoints    = kpts_global,
            pose_state   = "ASSIS",
            tables       = tables_list,
            frame_idx    = frame_idx,
            line_y       = line_y,
            reid_dist    = dist,
            is_reid_known= not is_new,
        )

        # Marquer les non-détectés comme non visibles :
        cm.mark_all_invisible()        # avant la boucle de détection
        cm.mark_visible(canonical_id)  # pour chaque détection

        # Vues pratiques :
        cm.active_records()    → clients visibles en ce moment
        cm.get(cid)            → record par ID
        cm.all_records()       → tous (actifs + archivés)
    """

    def __init__(self,
                 fps            : float = 25.0,
                 output_dir     : str   = "tracking_output",
                 table_dist_max : int   = TABLE_DIST_MAX_PX,
                 line_y_ratio   : float = 0.5):
        """
        fps            : FPS de la vidéo (pour calculs temporels)
        output_dir     : dossier de sortie CSV/JSON
        table_dist_max : distance max (px) pour attribuer une table
        line_y_ratio   : position relative de la ligne de comptage (0.5 = milieu)
        """
        self._records       : Dict[int, CustomerRecord] = {}
        self._fps           = fps
        self._output_dir    = Path(output_dir)
        self._table_dist_max= table_dist_max
        self._line_y_ratio  = line_y_ratio

        # Compteurs globaux
        self._count_in      = 0
        self._count_out     = 0
        self._frame_h       = None   # hauteur image (pour calculer line_y)

        self._output_dir.mkdir(parents=True, exist_ok=True)

    # ─────────────────────────────────────────────────────────
    #  MISE À JOUR PRINCIPALE
    # ─────────────────────────────────────────────────────────

    def mark_all_invisible(self):
        """
        À appeler EN DÉBUT de chaque frame, avant le traitement
        des détections. Marque tous les records comme non visibles.
        Les détections suivantes remettront visible=True.
        """
        for rec in self._records.values():
            rec.visible = False

    def mark_visible(self, canonical_id: int):
        """Marque explicitement un record comme visible (utile si update séparé)."""
        if canonical_id in self._records:
            self._records[canonical_id].visible = True

    def update(self,
               canonical_id  : int,
               byte_id       : int,
               bbox          : Tuple[int, int, int, int],
               conf          : float,
               frame_idx     : int,
               keypoints     : Optional[np.ndarray]  = None,
               pose_state    : str                   = "?",
               tables        : List[Dict]            = None,
               line_y        : Optional[int]         = None,
               reid_dist     : float                 = 1.0,
               is_reid_known : bool                  = False,
               frame_shape   : Optional[Tuple]       = None,
               ) -> "CustomerRecord":
        """
        Met à jour ou crée le CustomerRecord pour un client détecté.

        Paramètres :
            canonical_id  : ID stable fourni par ReIDTracker
            byte_id       : ID brut ByteTrack
            bbox          : (x1, y1, x2, y2) en pixels
            conf          : confiance de détection YOLO [0–1]
            frame_idx     : numéro de frame courant
            keypoints     : np.ndarray (17, 3) de RTMPose, ou None
            pose_state    : chaîne de caractères d'état de pose
            tables        : liste de dicts {id, cx, cy} détectés ce frame
            line_y        : position absolue (pixels) de la ligne de comptage
            reid_dist     : distance cosinus Re-ID
            is_reid_known : True si Re-ID a reconnu la personne
            frame_shape   : (H, W, C) pour calculer line_y automatiquement

        Retourne :
            Le CustomerRecord mis à jour.
        """
        x1, y1, x2, y2 = bbox
        cx = (x1 + x2) // 2
        cy = (y1 + y2) // 2

        # Stocker la hauteur image pour line_y automatique
        if frame_shape is not None:
            self._frame_h = frame_shape[0]

        # line_y automatique si non fourni
        if line_y is None and self._frame_h is not None:
            line_y = int(self._frame_h * self._line_y_ratio)

        # ── Création du record si nouveau ────────────────────
        if canonical_id not in self._records:
            rec = CustomerRecord(canonical_id=canonical_id)
            rec.temporal.first_frame  = frame_idx
            rec.temporal.first_seen_t = time.time()
            self._records[canonical_id] = rec
        else:
            rec = self._records[canonical_id]

        # ── Mise à jour détection ─────────────────────────────
        rec.bbox       = bbox
        rec.position   = (cx, cy)
        rec.confidence = conf
        rec.visible    = True

        # ── Temporel ──────────────────────────────────────────
        rec.temporal.on_detected(frame_idx)

        # ── Mouvement ─────────────────────────────────────────
        rec.movement.push(cx, cy)

        # ── Re-ID ─────────────────────────────────────────────
        rec.reid.update(byte_id, canonical_id, reid_dist, is_reid_known)

        # ── Pose RTMPose ──────────────────────────────────────
        rec.pose.update(keypoints, pose_state)

        # ── Table la plus proche ──────────────────────────────
        if tables is not None:
            t_id, t_dist, t_cx, t_cy = self._assign_table(cx, cy, tables)
            rec.table.update(t_id, t_dist, t_cx, t_cy)

        # ── Ligne de comptage ─────────────────────────────────
        if line_y is not None:
            self._update_line_count(rec, cy, line_y)

        return rec

    # ─────────────────────────────────────────────────────────
    #  ATTRIBUTION DE TABLE
    # ─────────────────────────────────────────────────────────

    def _assign_table(self,
                      person_cx : int,
                      person_cy : int,
                      tables    : List[Dict]
                      ) -> Tuple[Optional[int], float, Optional[int], Optional[int]]:
        """
        Retourne (table_id, distance_px, table_cx, table_cy).
        Retourne (None, 9999, None, None) si aucune table proche.
        """
        if not tables:
            return None, 9999.0, None, None

        best_id   = None
        best_dist = float("inf")
        best_cx   = None
        best_cy   = None

        for t in tables:
            d = math.sqrt((person_cx - t["cx"])**2 + (person_cy - t["cy"])**2)
            if d < best_dist:
                best_dist = d
                best_id   = t.get("id")
                best_cx   = t["cx"]
                best_cy   = t["cy"]

        if best_dist > self._table_dist_max:
            return None, best_dist, None, None

        return best_id, best_dist, best_cx, best_cy

    # ─────────────────────────────────────────────────────────
    #  COMPTAGE LIGNE VIRTUELLE
    # ─────────────────────────────────────────────────────────

    def _update_line_count(self, rec: CustomerRecord, cy: int, line_y: int):
        """Met à jour le comptage entrées/sorties (données brutes)."""
        if cy <= line_y - LINE_HYSTERESIS_PX:
            side_now = "above"
        elif cy >= line_y + LINE_HYSTERESIS_PX:
            side_now = "below"
        else:
            return  # zone morte : ignore les oscillations du tracker

        if rec.line_side:   # si on avait un côté précédent
            if (rec.line_side == "above" and side_now == "below"
                    and not rec.crossed_in):
                self._count_in    += 1
                rec.crossed_in     = True
            elif (rec.line_side == "below" and side_now == "above"
                  and not rec.crossed_out):
                self._count_out   += 1
                rec.crossed_out    = True

        rec.line_side = side_now

    # ─────────────────────────────────────────────────────────
    #  ACCESSEURS — VUES FILTRÉES
    # ─────────────────────────────────────────────────────────

    def get(self, canonical_id: int) -> Optional[CustomerRecord]:
        """Retourne le record d'un client par son ID canonical."""
        return self._records.get(canonical_id)

    def all_records(self) -> Dict[int, CustomerRecord]:
        """Tous les records (visibles + non visibles)."""
        return dict(self._records)

    def active_records(self) -> Dict[int, CustomerRecord]:
        """Uniquement les clients visibles dans la frame courante."""
        return {cid: r for cid, r in self._records.items() if r.visible}

    def lost_records(self) -> Dict[int, CustomerRecord]:
        """Clients absents depuis trop longtemps (donnée brute)."""
        return {cid: r for cid, r in self._records.items()
                if not r.visible and r.is_lost}

    def by_statut(self, statut: str) -> Dict[int, CustomerRecord]:
        """Records filtrés par statut (écrit par StateMachine)."""
        return {cid: r for cid, r in self._records.items()
                if r.statut == statut}

    def records_with_table(self) -> Dict[int, CustomerRecord]:
        """Records qui ont une table assignée."""
        return {cid: r for cid, r in self._records.items() if r.has_table}

    # ─────────────────────────────────────────────────────────
    #  STATISTIQUES GLOBALES (données brutes)
    # ─────────────────────────────────────────────────────────

    @property
    def count_in(self) -> int:
        """Nombre de personnes ayant traversé la ligne vers le bas."""
        return self._count_in

    @property
    def count_out(self) -> int:
        """Nombre de personnes ayant traversé la ligne vers le haut."""
        return self._count_out

    @property
    def total_seen(self) -> int:
        """Nombre total de clients uniques vus depuis le début."""
        return len(self._records)

    @property
    def visible_count(self) -> int:
        """Nombre de clients visibles en ce moment."""
        return sum(1 for r in self._records.values() if r.visible)

    def global_stats(self) -> dict:
        """Résumé des statistiques globales (pour Orchestrator ou UI)."""
        return {
            "total_seen"   : self.total_seen,
            "visible_now"  : self.visible_count,
            "count_in"     : self.count_in,
            "count_out"    : self.count_out,
            "by_statut"    : {
                s: len(self.by_statut(s))
                for s in ["NOUVEAU","CHERCHE_TABLE","ASSIS",
                           "EN_ATTENTE","SERVI","PARTI"]
            },
        }

    # ─────────────────────────────────────────────────────────
    #  EXPORT CSV / JSON
    # ─────────────────────────────────────────────────────────

    def export_csv_snapshot(self,
                            writer   : csv.DictWriter,
                            frame_idx: int):
        """
        Écrit une ligne CSV pour chaque record actif.
        Appeler toutes les N frames pour ne pas saturer le disque.

        Usage :
            with open("tracking.csv","w",newline="") as f:
                writer = cm.make_csv_writer(f)
                # dans la boucle :
                if frame_idx % 30 == 0:
                    cm.export_csv_snapshot(writer, frame_idx)
        """
        for rec in self._records.values():
            writer.writerow(rec.to_csv_row(frame_idx))

    def make_csv_writer(self, file_obj) -> csv.DictWriter:
        """Crée un DictWriter avec les bons headers et écrit l'entête."""
        # On génère un record factice pour obtenir les clés
        dummy = CustomerRecord(canonical_id=0)
        headers = list(dummy.to_csv_row(0).keys())
        writer = csv.DictWriter(file_obj, fieldnames=headers, extrasaction="ignore")
        writer.writeheader()
        return writer

    def export_json_final(self, path: Optional[str] = None) -> str:
        """
        Exporte l'état final de tous les records en JSON.
        Retourne le chemin du fichier créé.
        """
        if path is None:
            path = str(self._output_dir / "customers_final.json")

        data = {
            "global"   : self.global_stats(),
            "customers": {
                str(cid): rec.to_dict()
                for cid, rec in sorted(self._records.items())
            }
        }
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

        return path

    def print_summary(self):
        """Affiche un résumé console de tous les clients."""
        print("\n" + "=" * 65)
        print("  CUSTOMER MANAGER — RÉSUMÉ")
        print("=" * 65)
        stats = self.global_stats()
        print(f"  Total vus      : {stats['total_seen']}")
        print(f"  Visibles       : {stats['visible_now']}")
        print(f"  Entrées/Sorties: {stats['count_in']} / {stats['count_out']}")
        print(f"\n  Par statut     : {stats['by_statut']}")
        print(f"\n  Détail par client :")
        for cid, rec in sorted(self._records.items()):
            tbl = f"Table {rec.table_id}" if rec.has_table else "—"
            print(
                f"    #{cid:<4} | statut={rec.statut:<14} | "
                f"pose={rec.pose_state:<12} | "
                f"présence={rec.presence_s:.1f}s | "
                f"table={tbl} | "
                f"visible={'✓' if rec.visible else '✗'}"
            )
        print("=" * 65)


# ─────────────────────────────────────────────────────────────
#  INTÉGRATION DANS LA BOUCLE PRINCIPALE
# ─────────────────────────────────────────────────────────────
#
#  Exemple complet de boucle avec CustomerManager + StateMachine :
#
#  from customer_manager import CustomerManager, CustomerRecord
#  from state_machine import ClientStateMachine, Statut, FrameContext
#
#  cm = CustomerManager(fps=FPS, output_dir=OUTPUT_DIR)
#  state_machines: Dict[int, ClientStateMachine] = {}
#
#  while True:
#      ret, frame = cap.read()
#      frame_idx += 1
#
#      # ── 1. Marquer tout invisible avant détection ──────────
#      cm.mark_all_invisible()
#
#      # ── 2. YOLO + ByteTrack ────────────────────────────────
#      results = yolo.track(frame, ..., persist=True)
#
#      for each detected person:
#          # ── 3. Re-ID ──────────────────────────────────────
#          cid, dist, is_new = reid.process(frame, bbox, byte_id, frame_idx)
#
#          # ── 4. RTMPose ────────────────────────────────────
#          kpts = pose_model.infer(crop_person(frame, bbox))
#          pose = analyze_pose_state(kpts)
#
#          # ── 5. CustomerManager.update() ────────────────────
#          record = cm.update(
#              canonical_id  = cid,
#              byte_id       = byte_id,
#              bbox          = (x1,y1,x2,y2),
#              conf          = confidence,
#              frame_idx     = frame_idx,
#              keypoints     = kpts,
#              pose_state    = pose,
#              tables        = tables_detected,
#              line_y        = line_y,
#              reid_dist     = dist,
#              is_reid_known = not is_new,
#              frame_shape   = frame.shape,
#          )
#
#          # ── 6. StateMachine.update() ───────────────────────
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
#              robot_arrived = orchestrator.robot_at_table(record.table_id),
#              visible       = record.visible,
#          )
#          new_statut = state_machines[cid].update(ctx)
#
#          # ── 7. Écrire le statut dans le record ────────────
#          record.statut       = new_statut.value
#          record.statut_extra = state_machines[cid].to_dict()
#
#          # ── 8. Orchestrator lit record + statut → actions ──
#          orchestrator.process(record, state_machines[cid])
#
#      # ── Export CSV périodique ──────────────────────────────
#      if frame_idx % 30 == 0:
#          cm.export_csv_snapshot(csv_writer, frame_idx)
#
#  # ── Fin : export JSON ─────────────────────────────────────
#  path = cm.export_json_final()
#  cm.print_summary()


# ─────────────────────────────────────────────────────────────
#  TEST STANDALONE — simulation sans caméra
# ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Test CustomerManager standalone")
    parser.add_argument("--demo", action="store_true",
                        help="Lancer la simulation de démonstration")
    args = parser.parse_args()

    print("=" * 65)
    print("  CustomerManager — Test standalone")
    print("=" * 65)

    cm = CustomerManager(fps=25.0, output_dir="test_output")

    # ── Simulation : 3 clients sur 5 frames ──────────────────
    tables = [
        {"id": 1, "cx": 300, "cy": 400},
        {"id": 2, "cx": 500, "cy": 350},
    ]

    for frame_idx in range(1, 51):
        cm.mark_all_invisible()

        # Client 1 — entre, se déplace vers table 1
        rec1 = cm.update(
            canonical_id  = 1,
            byte_id       = 10,
            bbox          = (100 + frame_idx*2, 200, 160 + frame_idx*2, 300),
            conf          = 0.87,
            frame_idx     = frame_idx,
            pose_state    = "DEBOUT" if frame_idx < 20 else "ASSIS",
            tables        = tables,
            line_y        = 250,
            reid_dist     = 0.15,
            is_reid_known = frame_idx > 1,
            frame_shape   = (480, 640, 3),
        )

        # Client 2 — assis à table 2 dès le début
        if frame_idx > 5:
            rec2 = cm.update(
                canonical_id  = 2,
                byte_id       = 11,
                bbox          = (480, 320, 540, 420),
                conf          = 0.91,
                frame_idx     = frame_idx,
                pose_state    = "ASSIS",
                tables        = tables,
                line_y        = 250,
                reid_dist     = 0.10,
                is_reid_known = True,
                frame_shape   = (480, 640, 3),
            )
            # Simuler statut écrit par StateMachine
            rec2.statut = "ASSIS"

        # Simuler statut écrit par StateMachine pour client 1
        rec1.statut = "CHERCHE_TABLE" if frame_idx < 20 else "EN_ATTENTE"

    # ── Affichage ──────────────────────────────────────────────
    cm.print_summary()

    # ── Vues filtrées ──────────────────────────────────────────
    print("\n[TEST] active_records :", list(cm.active_records().keys()))
    print("[TEST] records_with_table :", list(cm.records_with_table().keys()))
    print("[TEST] global_stats :", cm.global_stats())

    # ── Export JSON ────────────────────────────────────────────
    path = cm.export_json_final()
    print(f"\n[TEST] JSON exporté → {path}")
    print("\n✅ CustomerManager OK — aucune décision prise.")
