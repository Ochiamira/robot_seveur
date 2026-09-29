"""
=============================================================
  NEXOR Vision — Tracking + RTMPose  (pipeline complet)
  ─────────────────────────────────────────────────────
  Étape 1 : YOLOv8s détecte person (0) + table (1)
  Étape 2 : ByteTrack assigne un ID persistant
  Étape 3 : RTMPose estime la pose de chaque personne trackée
  Étape 4 : Analyse de pose → état (debout/assis/levée de main)
  Étape 5 : Affichage + export CSV/JSON

  Backend RTMPose :
    rtmlib (pip install rtmlib onnxruntime) — classe RTMPose (pose seule,
    pas de re-détection interne : le bbox vient déjà de YOLO+ByteTrack).

  Compatibilité : Windows · Linux · Kaggle · Colab · RPi5
=============================================================
"""

import cv2
import csv
import json
import math
import time
import numpy as np
from collections import Counter, deque
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional, List, Tuple, Dict
from ultralytics import YOLO

# ─────────────────────────────────────────────────────────────
#  CONFIG — adapte ces chemins
# ─────────────────────────────────────────────────────────────

YOLO_WEIGHTS  = r"C:\pfe_project\detection\runs\resto_v7\weights\best.pt"
OUTPUT_DIR    = r"C:\pfe_project\detection\tracking_output"
VIDEO_SOURCE  = r"C:\pfe_project\detection\videos\test1.mp4"
CONF_THRESH   = 0.35
IOU_THRESH    = 0.45
TRACK_CLASSES = [0, 1]
PERSON_CLS    = 0
TABLE_CLS     = 1

COUNT_LINE_Y  = 0.5
TABLE_DIST_PX = 200

DELAY_EN_ATTENTE = 3.0
DELAY_SERVI      = 10.0
DELAY_PARTI      = 5.0

CSV_EXPORT_EVERY = 30

# ── RTMPose ──────────────────────────────────────────────────
# Modèle RTMPose à utiliser (taille vs vitesse) :
#   "RTMPose-t"  → très rapide, précision correcte (RPi5)
#   "RTMPose-s"  → rapide, bon équilibre (recommandé)
#   "RTMPose-m"  → précis, plus lent
RTMPOSE_MODEL  = "RTMPose-s"

# Marge autour du bbox (pixels) avant crop pour RTMPose
BBOX_PAD       = 20

# Seuil de confiance keypoint pour les afficher
KPT_CONF_THRESH = 0.3

# Dessiner le squelette sur la vidéo principale
DRAW_SKELETON  = True

# Analyser la pose (debout / assis / main levée)
ANALYZE_POSE   = True


# ─────────────────────────────────────────────────────────────
#  KEYPOINTS COCO-17 (index → nom)
# ─────────────────────────────────────────────────────────────

COCO_KPT_NAMES = [
    "nez", "oeil_g", "oeil_d", "oreille_g", "oreille_d",
    "epaule_g", "epaule_d", "coude_g", "coude_d",
    "poignet_g", "poignet_d", "hanche_g", "hanche_d",
    "genou_g", "genou_d", "cheville_g", "cheville_d",
]

# Connexions squelette COCO-17
SKELETON_LINKS = [
    (0,1),(0,2),(1,3),(2,4),           # tête
    (5,6),(5,7),(7,9),(6,8),(8,10),    # bras
    (5,11),(6,12),(11,12),             # torse
    (11,13),(13,15),(12,14),(14,16),   # jambes
]

# Couleurs par groupe corporel (BGR)
LIMB_COLORS = {
    "tête"  : (255, 200,  50),
    "bras"  : ( 50, 200, 255),
    "torse" : ( 50, 255, 100),
    "jambes": (200,  50, 255),
}
LINK_GROUPS = {
    (0,1):"tête",(0,2):"tête",(1,3):"tête",(2,4):"tête",
    (5,6):"torse",(5,11):"torse",(6,12):"torse",(11,12):"torse",
    (5,7):"bras",(7,9):"bras",(6,8):"bras",(8,10):"bras",
    (11,13):"jambes",(13,15):"jambes",(12,14):"jambes",(14,16):"jambes",
}


# ─────────────────────────────────────────────────────────────
#  BACKEND RTMPose — chargement automatique
# ─────────────────────────────────────────────────────────────

class RTMPoseBackend:
    """
    Wrapper unifié autour des trois backends possibles.
    Ordre de priorité : rtmlib → mmpose → onnxruntime-fallback
    """

    def __init__(self, model_name: str = "RTMPose-s"):
        self.backend  = None
        self.name     = "none"
        self._model   = None
        self._load(model_name)

    def _load(self, model_name: str):
        # ── rtmlib — seul backend (mmpose supprimé : trop instable sur Windows)
        # FIX : RTMPose (pose seule) au lieu de Body (détection+pose). Body relance
        # un détecteur de personne À L'INTÉRIEUR du crop déjà recadré par YOLO,
        # ce qui peut échouer sur des crops serrés/partiels → keypoints=None.
        try:
            from rtmlib import RTMPose
            import torch as _torch
            device = "cuda" if _torch.cuda.is_available() else "cpu"

            # FIX rtmlib 0.0.15 : RTMPose(onnx_model=) attend une URL complète ou
            # un chemin .onnx local — le nom court "RTMPose-s" n'est pas accepté.
            _RTMPOSE_URLS = {
                "RTMPose-t": "https://download.openmmlab.com/mmpose/v1/projects/rtmposev1/onnx_sdk/rtmpose-t_simcc-body7_pt-body7_420e-256x192-026a1439_20230504.zip",
                "RTMPose-s": "https://download.openmmlab.com/mmpose/v1/projects/rtmposev1/onnx_sdk/rtmpose-s_simcc-body7_pt-body7_420e-256x192-acd4a1ef_20230504.zip",
                "RTMPose-m": "https://download.openmmlab.com/mmpose/v1/projects/rtmposev1/onnx_sdk/rtmpose-m_simcc-body7_pt-body7_420e-256x192-e65b9a15_20230504.zip",
            }
            pose_url = _RTMPOSE_URLS.get(model_name, model_name)

            # CUDA rarement disponible via onnxruntime sur Windows sans ort-gpu
            # → force CPU si CUDAExecutionProvider absent
            try:
                import onnxruntime as _ort
                _providers = _ort.get_available_providers()
                if "CUDAExecutionProvider" not in _providers:
                    device = "cpu"
            except Exception:
                device = "cpu"

            self._model  = RTMPose(
                onnx_model  = pose_url,
                to_openpose = False,
                backend     = "onnxruntime",
                device      = device,
            )
            self.backend = "rtmlib"
            self.name    = f"rtmlib/{model_name} [{device}]"
            print(f"[RTMPose] ✅ rtmlib chargé — modèle={model_name}  device={device}")
            return
        except ImportError:
            print("[RTMPose] ❌ rtmlib non installé → pip install rtmlib onnxruntime")
        except Exception as e:
            print(f"[RTMPose] ❌ erreur rtmlib : {e}")

        print("[RTMPose] ⚠️  Pose désactivée — tracking seul actif.")
        self.backend = None

    def available(self) -> bool:
        return self.backend is not None

    def infer(self, img_crop: np.ndarray) -> Optional[np.ndarray]:
        """
        Retourne les keypoints COCO-17 sous forme (17, 3) : [x, y, conf]
        en coordonnées relatives au crop.
        Retourne None si le backend est indisponible.
        """
        if not self.available() or img_crop is None or img_crop.size == 0:
            return None

        try:
            if self.backend == "rtmlib":
                return self._infer_rtmlib(img_crop)
        except Exception as e:
            print(f"[RTMPose] Erreur inférence : {e}")
        return None

    def _infer_rtmlib(self, crop: np.ndarray) -> Optional[np.ndarray]:
        keypoints, scores = self._model(crop)
        # keypoints : (1, 17, 2)   scores : (1, 17)
        if keypoints is None or len(keypoints) == 0:
            return None
        kpts = np.asarray(keypoints[0], dtype=np.float32)
        scrs = np.asarray(scores[0], dtype=np.float32)
        if kpts.shape != (17, 2) or scrs.shape != (17,):
            return None
        if not np.isfinite(kpts).all() or not np.isfinite(scrs).all():
            return None
        result = np.zeros((17, 3), dtype=np.float32)
        result[:, :2] = kpts
        result[:, 2]  = scrs
        return result


# ─────────────────────────────────────────────────────────────
#  ANALYSE DE POSE
# ─────────────────────────────────────────────────────────────

class PoseState:
    DEBOUT      = "DEBOUT"
    ASSIS       = "ASSIS"
    MAIN_LEVEE  = "MAIN_LEVEE"
    INCONNU     = "?"

def angle_deg(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> float:
    """Angle en b formé par a-b-c."""
    ba = a - b
    bc = c - b
    cos_a = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc) + 1e-9)
    return float(np.degrees(np.arccos(np.clip(cos_a, -1, 1))))

def analyze_pose_state(kpts: np.ndarray, conf_thresh: float = 0.3) -> str:
    """
    Analyse simple de la posture à partir des keypoints COCO-17.

    Règles :
      MAIN_LEVEE  : poignet gauche ou droit au-dessus de l'épaule correspondante
      DEBOUT      : angle genou > 150° (jambe tendue)
      ASSIS       : angle genou < 120°
    """
    if kpts is None or np.asarray(kpts).shape != (17, 3):
        return PoseState.INCONNU

    def visible(idx):
        return kpts[idx, 2] >= conf_thresh

    def pt(idx):
        return kpts[idx, :2]

    # ── Main levée ─────────────────────────────────────────────────────────
    for poignet, epaule in [(9, 5), (10, 6)]:   # gauche, droite
        if visible(poignet) and visible(epaule):
            if pt(poignet)[1] < pt(epaule)[1]:  # y plus petit = plus haut
                return PoseState.MAIN_LEVEE

    # ── Debout / Assis via angle du genou ─────────────────────────────────
    # Les chevilles sont souvent partiellement cachées par une chaise ou une
    # table lorsque la personne est assise de profil. On exige toujours des
    # hanches/genoux fiables, mais on accepte une confiance légèrement plus
    # basse pour la cheville. Une flexion franche sur une jambe est une preuve
    # d'assise plus forte qu'une seconde jambe droite mais occultée. Le lissage
    # temporel évite qu'une flexion isolée pendant la marche suffise à changer
    # durablement la posture.
    knee_angles = []
    ankle_conf_thresh = max(0.15, conf_thresh * 0.70)
    for hanche, genou, cheville in [(11,13,15), (12,14,16)]:
        if (visible(hanche) and visible(genou)
                and kpts[cheville, 2] >= ankle_conf_thresh):
            knee_angles.append(angle_deg(pt(hanche), pt(genou), pt(cheville)))

    if knee_angles:
        if any(angle <= 125 for angle in knee_angles):
            return PoseState.ASSIS
        if all(angle >= 155 for angle in knee_angles):
            return PoseState.DEBOUT
        # Zone intermédiaire : utiliser le fallback hanche/genou ci-dessous.

    # ── Fallback via position relative tête / hanches ─────────────────────
    # Fallback géométrique lorsque chevilles/genoux sont cachés. En position
    # assise, le segment hanche→genou est généralement plus horizontal que
    # vertical ; debout, il est majoritairement vertical.
    thigh_votes = []
    for hip, knee in ((11, 13), (12, 14)):
        if visible(hip) and visible(knee):
            dx = abs(float(pt(knee)[0] - pt(hip)[0]))
            dy = abs(float(pt(knee)[1] - pt(hip)[1]))
            thigh_votes.append(PoseState.ASSIS if dx > dy * 0.85 else PoseState.DEBOUT)
    if thigh_votes:
        return Counter(thigh_votes).most_common(1)[0][0]

    return PoseState.INCONNU


class PoseSmoother:
    """Vote temporel par personne, sans masquer immédiatement une main levée."""
    def __init__(self, window: int = 7):
        self.window = max(1, int(window))
        self._history = {}

    def update(self, identity: int, pose: str) -> str:
        history = self._history.setdefault(identity, deque(maxlen=self.window))
        history.append(pose)
        if pose == PoseState.MAIN_LEVEE:
            return pose
        known = [p for p in history if p != PoseState.INCONNU]
        if not known:
            return PoseState.INCONNU
        return Counter(known).most_common(1)[0][0]

    def remove(self, identity: int) -> None:
        self._history.pop(identity, None)


# ─────────────────────────────────────────────────────────────
#  DESSIN DU SQUELETTE
# ─────────────────────────────────────────────────────────────

def draw_skeleton(frame: np.ndarray,
                  kpts_global: np.ndarray,
                  conf_thresh: float = 0.3,
                  person_color: Tuple = (0, 255, 0)):
    """
    Dessine keypoints + liens squelette sur `frame`.
    kpts_global : (17, 3) en coordonnées image complète.
    """
    # Liens
    for (i, j) in SKELETON_LINKS:
        if kpts_global[i, 2] >= conf_thresh and kpts_global[j, 2] >= conf_thresh:
            group = LINK_GROUPS.get((i,j), LINK_GROUPS.get((j,i), "torse"))
            color = LIMB_COLORS.get(group, (200, 200, 200))
            p1 = tuple(kpts_global[i, :2].astype(int))
            p2 = tuple(kpts_global[j, :2].astype(int))
            cv2.line(frame, p1, p2, color, 2, cv2.LINE_AA)

    # Keypoints
    for k in range(17):
        if kpts_global[k, 2] >= conf_thresh:
            px, py = int(kpts_global[k, 0]), int(kpts_global[k, 1])
            cv2.circle(frame, (px, py), 4, (255, 255, 255), -1)
            cv2.circle(frame, (px, py), 4, person_color, 1)


# ─────────────────────────────────────────────────────────────
#  STATUTS CLIENT + COULEURS
# ─────────────────────────────────────────────────────────────

class Statut:
    NOUVEAU        = "NOUVEAU"
    CHERCHE_TABLE  = "CHERCHE_TABLE"
    ASSIS          = "ASSIS"
    EN_ATTENTE     = "EN_ATTENTE"
    SERVI          = "SERVI"           # FIX: manquait
    PARTI          = "PARTI"           # FIX: manquait
    SERVI          = "SERVI"
    PARTI          = "PARTI"

STATUT_COLOR = {
    Statut.NOUVEAU       : (180, 255, 180),
    Statut.CHERCHE_TABLE : (255, 210, 120),   # FIX: ajouté — orange
    Statut.ASSIS         : (180, 220, 255),   # FIX: ajouté — bleu clair
    Statut.EN_ATTENTE    : (120, 160, 255),
    Statut.SERVI         : (200, 255, 255),
    Statut.PARTI         : (210, 210, 210),
}
STATUT_TXT_COLOR = {
    Statut.NOUVEAU       : (0,   130,   0),
    Statut.CHERCHE_TABLE : (160,  80,   0),   # FIX: ajouté — marron
    Statut.ASSIS         : (0,    80, 180),   # FIX: ajouté — bleu
    Statut.EN_ATTENTE    : (0,    80, 200),
    Statut.SERVI         : (0,   150, 150),
    Statut.PARTI         : (100, 100, 100),
}


# ─────────────────────────────────────────────────────────────
#  DATACLASS PersonRecord — enrichie avec la pose
# ─────────────────────────────────────────────────────────────

@dataclass
class PersonRecord:
    track_id      : int
    confidence    : float = 0.0
    bbox          : Tuple = (0,0,0,0)
    position      : Tuple = (0,0)
    track_history : List  = field(default_factory=list)
    max_trail     : int   = 40
    first_frame   : int   = 0
    last_frame    : int   = 0
    frame_count   : int   = 0
    first_seen_t  : float = field(default_factory=time.time)
    last_seen_t   : float = field(default_factory=time.time)
    statut        : str   = Statut.NOUVEAU
    statut_since  : float = field(default_factory=time.time)
    side          : str   = ""
    crossed_in    : bool  = False
    crossed_out   : bool  = False
    table_id      : Optional[int]   = None
    table_dist_px : float           = 9999.0
    visible       : bool            = True

    # ── Champs RTMPose ────────────────────────────────────────────────────────
    keypoints     : Optional[np.ndarray] = None   # (17, 3) coords image globale
    pose_state    : str = PoseState.INCONNU        # DEBOUT / ASSIS / MAIN_LEVEE

    def update_position(self, cx, cy, conf, bbox, frame_idx):
        self.position    = (cx, cy)
        self.bbox        = bbox
        self.confidence  = conf
        self.last_frame  = frame_idx
        self.last_seen_t = time.time()
        self.frame_count += 1
        self.visible     = True
        self.track_history.append((cx, cy))
        if len(self.track_history) > self.max_trail:
            self.track_history.pop(0)

    def presence_seconds(self) -> float:
        return self.last_seen_t - self.first_seen_t

    def absence_seconds(self) -> float:
        return time.time() - self.last_seen_t

    def update_statut(self):
        """SM simplifiée — pour pipeline standalone rtmpose_tracker.
        Pour la SM complète avec keypoints, utiliser state_machine.py."""
        now     = time.time()
        elapsed = now - self.statut_since

        # NOUVEAU → CHERCHE_TABLE après stabilisation
        if self.statut == Statut.NOUVEAU and elapsed > DELAY_EN_ATTENTE:
            self.statut = Statut.CHERCHE_TABLE   # FIX: était EN_ATTENTE direct
            self.statut_since = now

        # CHERCHE_TABLE → ASSIS si table assignée
        elif (self.statut == Statut.CHERCHE_TABLE
              and elapsed > DELAY_EN_ATTENTE
              and self.table_id is not None):
            self.statut = Statut.ASSIS            # FIX: ajouté
            self.statut_since = now

        # ASSIS → EN_ATTENTE après délai
        elif self.statut == Statut.ASSIS and elapsed > DELAY_SERVI:
            self.statut = Statut.EN_ATTENTE       # FIX: ajouté
            self.statut_since = now

        # EN_ATTENTE → SERVI après délai
        elif self.statut == Statut.EN_ATTENTE and elapsed > DELAY_SERVI:
            self.statut = Statut.SERVI
            self.statut_since = now

        # Tout état → PARTI si absent trop longtemps
        elif self.statut != Statut.PARTI and self.absence_seconds() > DELAY_PARTI:
            self.statut = Statut.PARTI
            self.statut_since = now
            self.visible = False

    def to_csv_row(self, frame_idx: int) -> dict:
        kpt_flat = []
        if self.keypoints is not None:
            kpt_flat = self.keypoints.flatten().tolist()
        return {
            "frame"         : frame_idx,
            "track_id"      : self.track_id,
            "statut"        : self.statut,
            "pose_state"    : self.pose_state,
            "visible"       : int(self.visible),
            "cx"            : self.position[0],
            "cy"            : self.position[1],
            "confidence"    : round(self.confidence, 3),
            "presence_s"    : round(self.presence_seconds(), 2),
            "first_frame"   : self.first_frame,
            "last_frame"    : self.last_frame,
            "frame_count"   : self.frame_count,
            "trail_pts"     : len(self.track_history),
            "table_id"      : self.table_id if self.table_id is not None else -1,
            "table_dist_px" : round(self.table_dist_px, 1),
            "crossed_in"    : int(self.crossed_in),
            "crossed_out"   : int(self.crossed_out),
            # keypoints aplatis : x0,y0,c0, x1,y1,c1, ...
            "keypoints"     : json.dumps([round(v,1) for v in kpt_flat]),
        }


# ─────────────────────────────────────────────────────────────
#  UTILITAIRES
# ─────────────────────────────────────────────────────────────

def get_color(tid: int) -> Tuple:
    np.random.seed(int(tid) * 7 + 13)
    return tuple(np.random.randint(80, 255, 3).tolist())

def dist_px(p1, p2) -> float:
    return math.sqrt((p1[0]-p2[0])**2 + (p1[1]-p2[1])**2)

def assign_table(cx, cy, tables) -> Tuple[Optional[int], float]:
    best_id, best_dist = None, 9999.0
    for t in tables:
        d = dist_px((cx,cy), (t["cx"],t["cy"]))
        if d < best_dist:
            best_dist, best_id = d, t["id"]
    if best_dist > TABLE_DIST_PX:
        return None, best_dist
    return best_id, best_dist

def crop_person(frame: np.ndarray, bbox: Tuple, pad=BBOX_PAD) -> np.ndarray:
    """Extrait le crop de la personne avec padding."""
    H, W = frame.shape[:2]
    x1, y1, x2, y2 = bbox
    if isinstance(pad, float) and 0.0 <= pad < 1.0:
        pad = int(max(x2 - x1, y2 - y1) * pad)
    pad = max(0, int(pad))
    x1 = max(0, x1 - pad); y1 = max(0, y1 - pad)
    x2 = min(W, x2 + pad); y2 = min(H, y2 + pad)
    if x2 <= x1 or y2 <= y1:
        return None
    return frame[y1:y2, x1:x2].copy()

def kpts_to_global(kpts_crop: np.ndarray, bbox: Tuple, pad=BBOX_PAD,
                   frame_shape: Tuple = None) -> np.ndarray:
    """
    Convertit les keypoints du repère crop → repère image complète.
    """
    H_f, W_f = frame_shape[:2] if frame_shape else (9999, 9999)
    x1, y1, x2, y2 = bbox
    if isinstance(pad, float) and 0.0 <= pad < 1.0:
        pad = int(max(x2 - x1, y2 - y1) * pad)
    pad = max(0, int(pad))
    x1p = max(0, x1 - pad)
    y1p = max(0, y1 - pad)
    kpts_global = kpts_crop.copy()
    kpts_global[:, 0] += x1p
    kpts_global[:, 1] += y1p
    # FIX : clip pour éviter des coordonnées hors image (x<0 ou x>W)
    kpts_global[:, 0] = np.clip(kpts_global[:, 0], 0, W_f - 1)
    kpts_global[:, 1] = np.clip(kpts_global[:, 1], 0, H_f - 1)
    return kpts_global


# ─────────────────────────────────────────────────────────────
#  TABLEAU OPENCV TEMPS RÉEL (fenêtre séparée)
# ─────────────────────────────────────────────────────────────

COLS = [
    ("ID",        50), ("Statut",    88), ("Pose",      90),
    ("Conf",      50), ("Table",     50), ("Présence",  70),
    ("Position",  88), ("Frames",    58),
]
COL_HEADERS = [c[0] for c in COLS]
COL_WIDTHS  = [c[1] for c in COLS]
ROW_H    = 28; HEADER_H = 34; TITLE_H = 30; FOOTER_H = 26; MARGIN = 8
FONT_T   = cv2.FONT_HERSHEY_SIMPLEX
FS = 0.42; FT = 1; C_BORDER = (160,160,160)

POSE_COLOR_TXT = {
    PoseState.DEBOUT     : (0,  140,  0),
    PoseState.ASSIS      : (180, 80,  0),
    PoseState.MAIN_LEVEE : (0,   0,  200),
    PoseState.INCONNU    : (120,120, 120),
}

def build_table_frame(records: Dict, frame_idx: int, fps: float,
                      count_in: int, count_out: int) -> np.ndarray:
    table_w = sum(COL_WIDTHS) + MARGIN * 2
    n_rows  = max(len(records), 1)
    table_h = TITLE_H + HEADER_H + n_rows*ROW_H + FOOTER_H + 4
    canvas  = np.ones((table_h, table_w, 3), dtype=np.uint8) * 248

    # Titre
    cv2.rectangle(canvas, (0,0), (table_w, TITLE_H), (15,55,95), -1)
    cv2.putText(canvas,
        f"NEXOR TRACKING+POSE  frame={frame_idx}  in={count_in}  out={count_out}",
        (MARGIN, TITLE_H-9), FONT_T, 0.43, (255,255,255), FT, cv2.LINE_AA)

    # En-tête
    y0 = TITLE_H
    cv2.rectangle(canvas, (0,y0), (table_w, y0+HEADER_H), (30,30,30), -1)
    x = MARGIN
    for w, hdr in zip(COL_WIDTHS, COL_HEADERS):
        cv2.putText(canvas, hdr, (x+3, y0+HEADER_H-9),
                    FONT_T, FS+0.01, (255,255,255), FT, cv2.LINE_AA)
        cv2.line(canvas, (x+w,y0), (x+w,y0+HEADER_H), (80,80,80), 1)
        x += w
    y0 += HEADER_H

    # Lignes
    for row_i, (tid, rec) in enumerate(sorted(records.items())):
        y_row = y0 + row_i * ROW_H
        bg    = STATUT_COLOR.get(rec.statut, (240,240,240))
        cv2.rectangle(canvas, (0,y_row), (table_w, y_row+ROW_H), bg, -1)

        id_color = get_color(tid)
        cv2.rectangle(canvas, (MARGIN-2, y_row+5), (MARGIN+6, y_row+ROW_H-5), id_color, -1)

        table_str = f"T{rec.table_id}" if rec.table_id is not None else "—"
        values = [
            f"#{tid}", rec.statut, rec.pose_state,
            f"{rec.confidence:.2f}", table_str,
            f"{rec.presence_seconds():.1f}s",
            f"({rec.position[0]},{rec.position[1]})",
            str(rec.frame_count),
        ]
        x = MARGIN
        for col_i, (w, val) in enumerate(zip(COL_WIDTHS, values)):
            if col_i == 1:
                tc = STATUT_TXT_COLOR.get(rec.statut, (20,20,20))
            elif col_i == 2:
                tc = POSE_COLOR_TXT.get(rec.pose_state, (20,20,20))
            else:
                tc = (20,20,20)
            cv2.putText(canvas, val, (x+4, y_row+ROW_H-8),
                        FONT_T, FS, tc, FT, cv2.LINE_AA)
            cv2.line(canvas, (x+w,y_row), (x+w,y_row+ROW_H), C_BORDER, 1)
            x += w
        cv2.line(canvas, (0,y_row+ROW_H), (table_w,y_row+ROW_H), C_BORDER, 1)

    # Footer
    fy = y0 + n_rows * ROW_H
    cv2.rectangle(canvas, (0,fy), (table_w,fy+FOOTER_H), (30,30,30), -1)
    debout  = sum(1 for r in records.values() if r.pose_state==PoseState.DEBOUT)
    assis   = sum(1 for r in records.values() if r.pose_state==PoseState.ASSIS)
    levee   = sum(1 for r in records.values() if r.pose_state==PoseState.MAIN_LEVEE)
    cv2.putText(canvas,
        f"Total={len(records)}  visibles={sum(1 for r in records.values() if r.visible)}"
        f"  debout={debout}  assis={assis}  main_levee={levee}",
        (MARGIN, fy+FOOTER_H-8), FONT_T, FS, (255,255,255), FT, cv2.LINE_AA)

    cv2.rectangle(canvas, (0,0), (table_w-1,table_h-1), C_BORDER, 1)
    return canvas


# ─────────────────────────────────────────────────────────────
#  EXPORT CSV / JSON
# ─────────────────────────────────────────────────────────────

CSV_FIELDS = [
    "frame","track_id","statut","pose_state","visible","cx","cy",
    "confidence","presence_s","first_frame","last_frame","frame_count",
    "trail_pts","table_id","table_dist_px","crossed_in","crossed_out","keypoints",
]

def init_csv(path: Path):
    """Retourne (writer, file_handle) — fermer f.close() en fin de run()."""
    f = open(path, "w", newline="", encoding="utf-8")
    w = csv.DictWriter(f, fieldnames=CSV_FIELDS)
    w.writeheader()
    return w, f   # FIX: retourne aussi le handle pour pouvoir le fermer

def export_csv_snapshot(writer, records, frame_idx):
    for rec in records.values():
        writer.writerow(rec.to_csv_row(frame_idx))

def export_json_final(records, path: Path):
    data = {}
    for tid, rec in records.items():
        kpts = rec.keypoints.tolist() if rec.keypoints is not None else None
        data[str(tid)] = {
            "track_id"      : rec.track_id,
            "statut"        : rec.statut,
            "pose_state"    : rec.pose_state,
            "presence_s"    : round(rec.presence_seconds(), 2),
            "frame_count"   : rec.frame_count,
            "last_position" : list(rec.position),
            "confidence"    : round(rec.confidence, 3),
            "table_id"      : rec.table_id,
            "table_dist_px" : round(rec.table_dist_px, 1),
            "crossed_in"    : rec.crossed_in,
            "crossed_out"   : rec.crossed_out,
            "keypoints_17x3": kpts,
        }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    print(f"[OK] JSON → {path}")


# ─────────────────────────────────────────────────────────────
#  PIPELINE PRINCIPAL
# ─────────────────────────────────────────────────────────────

def run(source=0, yolo_weights=YOLO_WEIGHTS, save_video=True):
    """
    Boucle principale :
      frame → YOLO detect+track → pour chaque personne : crop → RTMPose → analyse pose
    """
    Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)

    # ── Charger les modèles ───────────────────────────────────────────────────
    print("[INIT] Chargement YOLOv8s...")
    yolo = YOLO(yolo_weights)

    print("[INIT] Chargement RTMPose...")
    pose_model = RTMPoseBackend(RTMPOSE_MODEL)

    # ── Caméra / vidéo ────────────────────────────────────────────────────────
    cap = cv2.VideoCapture(source if isinstance(source, str) else source)
    if not cap.isOpened():
        raise RuntimeError(f"Source inaccessible : {source}")

    W   = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H   = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    FPS = cap.get(cv2.CAP_PROP_FPS) or 25

    writer_vid = None
    if save_video:
        src_name   = Path(source).stem if isinstance(source, str) else "webcam"
        out_vid    = str(Path(OUTPUT_DIR) / f"{src_name}_pose_tracked.mp4")
        writer_vid = cv2.VideoWriter(
            out_vid, cv2.VideoWriter_fourcc(*"mp4v"), FPS, (W, H))

    csv_path   = Path(OUTPUT_DIR) / "tracking_pose.csv"
    csv_writer, csv_file = init_csv(csv_path)  # FIX: récupère le handle

    records   : Dict[int, PersonRecord] = {}
    count_in  = count_out = 0
    line_y    = int(H * COUNT_LINE_Y)
    frame_idx = 0

    TABLE_WIN = "NEXOR — Tableau Tracking + Pose"
    cv2.namedWindow(TABLE_WIN, cv2.WINDOW_NORMAL)

    print(f"\n[RUN] source={source}  YOLOv8={yolo_weights}")
    print(f"[RUN] RTMPose backend={pose_model.name}")
    print(f"[RUN] 'q' pour arrêter\n")

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_idx += 1

        # ════════════════════════════════════════════
        #  ÉTAPE 1 : YOLO détection + ByteTrack
        # ════════════════════════════════════════════
        results = yolo.track(
            frame, tracker="bytetrack.yaml",
            conf=CONF_THRESH, iou=IOU_THRESH,
            classes=TRACK_CLASSES, persist=True, verbose=False,
        )

        # Extraire les tables
        tables_detected = []
        if results[0].boxes is not None:
            for i, cls in enumerate(results[0].boxes.cls.cpu().numpy().astype(int)):
                if cls == TABLE_CLS:
                    bx1,by1,bx2,by2 = map(int, results[0].boxes.xyxy[i].cpu().numpy())
                    tid_raw = results[0].boxes.id
                    t_id    = int(tid_raw[i].item()) if tid_raw is not None else -(i+1)
                    tcx, tcy = (bx1+bx2)//2, (by1+by2)//2
                    tables_detected.append({"id":t_id,"cx":tcx,"cy":tcy})
                    cv2.rectangle(frame, (bx1,by1), (bx2,by2), (0,165,255), 2)
                    cv2.putText(frame, f"Table T{t_id}", (bx1,by1-6),
                                FONT_T, 0.5, (0,165,255), 1)

        # Réinitialiser visibilité
        for rec in records.values():
            rec.visible = False

        # ════════════════════════════════════════════
        #  ÉTAPE 2 : traitement par personne
        # ════════════════════════════════════════════
        if results[0].boxes is not None and results[0].boxes.id is not None:
            box_ids = results[0].boxes.id.cpu().numpy().astype(int)

            for i, cls_val in enumerate(results[0].boxes.cls.cpu().numpy().astype(int)):
                if cls_val != PERSON_CLS:
                    continue

                tid  = int(box_ids[i])
                conf = float(results[0].boxes.conf[i].item())
                x1,y1,x2,y2 = map(int, results[0].boxes.xyxy[i].cpu().numpy())
                cx, cy = (x1+x2)//2, (y1+y2)//2
                color  = get_color(tid)

                # ── PersonRecord ──────────────────────────────────────────────
                if tid not in records:
                    records[tid] = PersonRecord(
                        track_id     = tid,
                        first_frame  = frame_idx,
                        first_seen_t = time.time(),
                        statut_since = time.time(),
                    )
                rec = records[tid]
                rec.update_position(cx, cy, conf, (x1,y1,x2,y2), frame_idx)

                # Attribution table
                t_id, t_dist = assign_table(cx, cy, tables_detected)
                rec.table_id      = t_id
                rec.table_dist_px = t_dist

                # Comptage ligne
                side_now = "above" if cy < line_y else "below"
                if rec.side:
                    if rec.side == "above" and side_now == "below":
                        count_in += 1; rec.crossed_in = True
                    elif rec.side == "below" and side_now == "above":
                        count_out += 1; rec.crossed_out = True
                rec.side = side_now
                rec.update_statut()

                # ════════════════════════════════════════════
                #  ÉTAPE 3 : RTMPose sur le crop
                # ════════════════════════════════════════════
                if pose_model.available():
                    crop = crop_person(frame, (x1,y1,x2,y2), pad=BBOX_PAD)
                    if crop is not None:
                        kpts_crop = pose_model.infer(crop)
                        if kpts_crop is not None:
                            # Convertir coordonnées crop → image globale
                            kpts_global = kpts_to_global(
                                kpts_crop, (x1,y1,x2,y2),
                                pad=BBOX_PAD, frame_shape=frame.shape)
                            rec.keypoints = kpts_global

                            # ════════════════════════════════════════════
                            #  ÉTAPE 4 : analyse de la posture
                            # ════════════════════════════════════════════
                            if ANALYZE_POSE:
                                rec.pose_state = analyze_pose_state(
                                    kpts_global, KPT_CONF_THRESH)

                            # ════════════════════════════════════════════
                            #  ÉTAPE 5 : dessin squelette
                            # ════════════════════════════════════════════
                            if DRAW_SKELETON:
                                draw_skeleton(frame, kpts_global,
                                              KPT_CONF_THRESH, color)

                # ── Boîte + label enrichi ─────────────────────────────────────
                cv2.rectangle(frame, (x1,y1), (x2,y2), color, 2)
                tbl_lbl = f" T{rec.table_id}" if rec.table_id else ""
                label   = f"#{tid} {rec.statut} {rec.pose_state}{tbl_lbl}"
                (tw,th),_ = cv2.getTextSize(label, FONT_T, 0.48, 1)
                cv2.rectangle(frame, (x1,y1-th-8), (x1+tw+6,y1), color, -1)
                cv2.putText(frame, label, (x1+3,y1-4), FONT_T, 0.48, (0,0,0), 1)

                # Trail
                pts = rec.track_history
                for k in range(1, len(pts)):
                    cv2.line(frame, pts[k-1], pts[k], color,
                             max(1, int(3*(k/len(pts)))))

        # Marquer absents
        for rec in records.values():
            if not rec.visible:
                rec.update_statut()

        # ── Ligne de comptage ─────────────────────────────────────────────────
        cv2.line(frame, (0,line_y), (W,line_y), (0,255,255), 2)
        cv2.putText(frame, "Ligne comptage", (10,line_y-8), FONT_T, 0.48, (0,255,255), 1)

        # ── Overlay stats ─────────────────────────────────────────────────────
        visible_now = sum(1 for r in records.values() if r.visible)
        ol = [f"Visibles:{visible_now}", f"Total:{len(records)}",
              f"In:{count_in}  Out:{count_out}", f"Frame:{frame_idx}",
              f"RTMPose:{pose_model.name[:18]}"]
        cv2.rectangle(frame, (0,0), (230, len(ol)*22+8), (0,0,0), -1)
        for i, l in enumerate(ol):
            cv2.putText(frame, l, (6,18+i*22), FONT_T, 0.48, (255,255,255), 1)

        # ── Affichage ─────────────────────────────────────────────────────────
        cv2.imshow("NEXOR — YOLOv8s + ByteTrack + RTMPose", frame)
        cv2.imshow(TABLE_WIN, build_table_frame(records, frame_idx, FPS, count_in, count_out))

        if writer_vid:
            writer_vid.write(frame)

        # Export CSV périodique
        if frame_idx % CSV_EXPORT_EVERY == 0:
            export_csv_snapshot(csv_writer, records, frame_idx)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    # ── Finalisation ──────────────────────────────────────────────────────────
    export_csv_snapshot(csv_writer, records, frame_idx)
    csv_file.close()           # FIX: fermeture propre du fichier CSV
    cap.release()
    if writer_vid:
        writer_vid.release()
    cv2.destroyAllWindows()

    json_path = Path(OUTPUT_DIR) / "tracking_pose_final.json"
    export_json_final(records, json_path)

    # Stats console
    print("\n" + "="*65)
    print("  STATISTIQUES FINALES")
    print("="*65)
    print(f"  Frames          : {frame_idx}")
    print(f"  Personnes vues  : {len(records)}")
    print(f"  Entrées/Sorties : {count_in} / {count_out}")
    print(f"\n  Par personne :")
    for tid, rec in sorted(records.items()):
        print(f"    #{tid:<3} | {rec.statut:<12} | {rec.pose_state:<12} | "
              f"{rec.presence_seconds():.1f}s | "
              f"table={'T'+str(rec.table_id) if rec.table_id else '—'}")
    print(f"\n  CSV  → {csv_path}")
    print(f"  JSON → {json_path}")
    print("="*65)


# ─────────────────────────────────────────────────────────────
#  MAIN
# ─────────────────────────────────────────────────────────────


# ─────────────────────────────────────────────────────────────
#  ALIAS — compatibilité avec 00-run.py
#  00-run.py importe RTMPoseEstimator, la classe s'appelle RTMPoseBackend
# ─────────────────────────────────────────────────────────────

class RTMPoseEstimator(RTMPoseBackend):
    """
    Alias de RTMPoseBackend pour compatibilité avec 00-run.py.
    Ajoute la méthode infer(crop, bbox) attendue par run_full().
    """
    def infer(self, crop: np.ndarray, bbox=None) -> Optional[np.ndarray]:
        """
        Surcharge pour accepter le paramètre bbox de run_full().
        RTMPoseBackend.infer() ne prend que crop.
        """
        return super().infer(crop)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(
        description="NEXOR — Tracking YOLOv8s + RTMPose")
    parser.add_argument("--source",  default=VIDEO_SOURCE,
                        help="0=webcam, chemin vidéo")
    parser.add_argument("--weights", default=YOLO_WEIGHTS,
                        help="Poids YOLOv8")
    parser.add_argument("--no-save", action="store_true",
                        help="Ne pas sauvegarder la vidéo")
    parser.add_argument("--no-pose", action="store_true",
                        help="Désactiver RTMPose (tracking seul)")
    args = parser.parse_args()

    if args.no_pose:
        DRAW_SKELETON = False
        ANALYZE_POSE  = False

    src = int(args.source) if args.source.isdigit() else args.source
    run(src, args.weights, save_video=not args.no_save)
