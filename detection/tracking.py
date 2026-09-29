"""
=============================================================
  TRACKING DE PERSONNES — YOLOv8s + ByteTrack  (v3)
  Détection : person (0) + table (1)
  Tracking  : personnes uniquement (classe 0)

  Nouveautés v3 :
  • Structure PersonRecord — toutes les infos par personne
  • Statuts client : NOUVEAU → EN_ATTENTE → SERVI → PARTI
  • Attribution de table la plus proche (classe 1)
  • Temps réel de présence (secondes horloge)
  • Export CSV automatique à chaque frame et à la fin
  • Tableau OpenCV enrichi (fenêtre séparée)
  • Export JSON final pour intégration ROS2 / robot
=============================================================
"""

import cv2
import csv
import json
import time
import math
import numpy as np
from pathlib import Path
from dataclasses import dataclass, field, asdict
from typing import Optional, List, Tuple, Dict
from collections import defaultdict
from ultralytics import YOLO

# ─────────────────────────────────────────────────────────────
#  CONFIG
# ─────────────────────────────────────────────────────────────

BEST_V7       = r"C:/pfe_project/detection/runs/resto_v7/weights/best.pt"
OUTPUT        = r"C:/pfe_project/detection/tracking_output"

CONF_THRESH   = 0.35
IOU_THRESH    = 0.45
TRACK_CLASSES = [0, 1]          # on détecte person ET table
PERSON_CLS    = 0
TABLE_CLS     = 1

COUNT_LINE_Y  = 0.5             # ligne virtuelle : 50% de la hauteur
TABLE_DIST_PX = 200             # distance max (px) pour attribuer une table

# Délais de transition de statut (secondes)
DELAY_EN_ATTENTE = 3.0          # NOUVEAU → EN_ATTENTE après N s sans mouvement
DELAY_SERVI      = 10.0         # EN_ATTENTE → SERVI après N s (simulation)
DELAY_PARTI      = 5.0          # invisible depuis N s → PARTI

# Export CSV (toutes les N frames pour ne pas saturer le disque)
CSV_EXPORT_EVERY = 30


# ─────────────────────────────────────────────────────────────
#  STATUTS CLIENT
# ─────────────────────────────────────────────────────────────

class Statut:
    NOUVEAU        = "NOUVEAU"
    CHERCHE_TABLE  = "CHERCHE_TABLE"   # FIX: manquait
    ASSIS          = "ASSIS"           # FIX: manquait
    EN_ATTENTE     = "EN_ATTENTE"
    SERVI          = "SERVI"
    PARTI          = "PARTI"

# Couleur BGR par statut (pour le tableau)
STATUT_COLOR = {
    Statut.NOUVEAU       : (100, 255, 100),
    Statut.CHERCHE_TABLE : (255, 200,  50),   # FIX: ajouté
    Statut.ASSIS         : (100, 200, 255),   # FIX: ajouté
    Statut.EN_ATTENTE    : ( 50, 100, 255),
    Statut.SERVI         : ( 50, 255, 200),
    Statut.PARTI         : (150, 150, 150),
}
STATUT_TXT_COLOR = {
    Statut.NOUVEAU       : (0,   130, 0),
    Statut.CHERCHE_TABLE : (180, 120, 0),    # FIX: ajouté
    Statut.ASSIS         : (0,   80,  180),  # FIX: ajouté
    Statut.EN_ATTENTE    : (0,   80,  200),
    Statut.SERVI         : (0,   150, 150),
    Statut.PARTI         : (100, 100, 100),
}


# ─────────────────────────────────────────────────────────────
#  DATACLASS — UNE ENTRÉE PAR PERSONNE
# ─────────────────────────────────────────────────────────────

@dataclass
class PersonRecord:
    """
    Toutes les informations de suivi pour une personne.
    Remplace les multiples dictionnaires séparés.
    """
    track_id      : int

    # Détection
    confidence    : float = 0.0
    bbox          : Tuple[int,int,int,int] = (0,0,0,0)   # x1,y1,x2,y2
    position      : Tuple[int,int] = (0,0)                # cx, cy

    # Historique de mouvement
    track_history : List[Tuple[int,int]] = field(default_factory=list)
    max_trail     : int = 40

    # Temporel (frames)
    first_frame   : int = 0
    last_frame    : int = 0
    frame_count   : int = 0          # frames où la personne a été vue

    # Temporel (horloge)
    first_seen_t  : float = field(default_factory=time.time)
    last_seen_t   : float = field(default_factory=time.time)

    # Statut
    statut        : str = Statut.NOUVEAU
    statut_since  : float = field(default_factory=time.time)

    # Comptage ligne
    side          : str = ""         # "above" | "below"
    crossed_in    : bool = False
    crossed_out   : bool = False

    # Table associée
    table_id      : Optional[int] = None     # ID de la table la plus proche
    table_dist_px : float = 9999.0           # distance en pixels

    # Visibilité
    visible       : bool = True              # présent dans la frame courante

    # ── Méthodes utilitaires ──────────────────────────────────────────────────

    def update_position(self, cx: int, cy: int, conf: float,
                        bbox: Tuple, frame_idx: int):
        """Met à jour la position et l'historique."""
        self.position  = (cx, cy)
        self.bbox      = bbox
        self.confidence = conf
        self.last_frame = frame_idx
        self.last_seen_t = time.time()
        self.frame_count += 1
        self.visible   = True

        self.track_history.append((cx, cy))
        if len(self.track_history) > self.max_trail:
            self.track_history.pop(0)

    def presence_seconds(self) -> float:
        """Durée de présence depuis la première détection."""
        return self.last_seen_t - self.first_seen_t

    def absence_seconds(self) -> float:
        """Durée depuis la dernière détection."""
        return time.time() - self.last_seen_t

    def update_statut(self):
        """Machine à états : transitions automatiques de statut.
        
        Note : tracking.py gère une SM simplifiée (standalone).
        Pour la SM complète avec keypoints, utiliser state_machine.py.
        """
        now     = time.time()
        elapsed = now - self.statut_since

        # NOUVEAU → CHERCHE_TABLE après stabilisation
        if self.statut == Statut.NOUVEAU and elapsed > DELAY_EN_ATTENTE:
            self.statut       = Statut.CHERCHE_TABLE  # FIX: était EN_ATTENTE direct
            self.statut_since = now

        # CHERCHE_TABLE → ASSIS si immobile et table assignée
        elif (self.statut == Statut.CHERCHE_TABLE
              and elapsed > DELAY_EN_ATTENTE
              and self.table_id is not None):
            self.statut       = Statut.ASSIS           # FIX: ajouté
            self.statut_since = now

        # ASSIS → EN_ATTENTE après délai
        elif self.statut == Statut.ASSIS and elapsed > DELAY_SERVI:
            self.statut       = Statut.EN_ATTENTE      # FIX: ajouté
            self.statut_since = now

        # EN_ATTENTE → SERVI après délai
        elif self.statut == Statut.EN_ATTENTE and elapsed > DELAY_SERVI:
            self.statut       = Statut.SERVI
            self.statut_since = now

        # Tout statut → PARTI si absent trop longtemps
        elif self.statut != Statut.PARTI and self.absence_seconds() > DELAY_PARTI:
            self.statut       = Statut.PARTI
            self.statut_since = now
            self.visible      = False

    def to_csv_row(self, frame_idx: int) -> dict:
        """Sérialisation CSV — une ligne par personne par frame exportée."""
        return {
            "frame"         : frame_idx,
            "track_id"      : self.track_id,
            "statut"        : self.statut,   # CHERCHE_TABLE/ASSIS maintenant inclus
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
        }


# ─────────────────────────────────────────────────────────────
#  COULEURS / UTILITAIRES
# ─────────────────────────────────────────────────────────────

def get_color(track_id: int) -> Tuple:
    np.random.seed(int(track_id) * 7 + 13)
    return tuple(np.random.randint(80, 255, 3).tolist())


def dist_px(p1: Tuple, p2: Tuple) -> float:
    return math.sqrt((p1[0]-p2[0])**2 + (p1[1]-p2[1])**2)


def assign_table(person_cx: int, person_cy: int,
                 tables: List[Dict]) -> Tuple[Optional[int], float]:
    """
    Retourne (table_id, distance) de la table la plus proche.
    tables : liste de dicts {"id": int, "cx": int, "cy": int}
    """
    best_id, best_dist = None, 9999.0
    for t in tables:
        d = dist_px((person_cx, person_cy), (t["cx"], t["cy"]))
        if d < best_dist:
            best_dist = d
            best_id   = t["id"]
    if best_dist > TABLE_DIST_PX:
        return None, best_dist
    return best_id, best_dist


# ─────────────────────────────────────────────────────────────
#  TABLEAU OPENCV — FENÊTRE SÉPARÉE
# ─────────────────────────────────────────────────────────────

# Colonnes du tableau
COLS = [
    ("ID",        52),
    ("Statut",    90),
    ("Conf",      52),
    ("Table",     52),
    ("Dist(px)",  68),
    ("Présence",  72),
    ("Trail",     50),
    ("Position",  90),
    ("Entrée(f)", 72),
    ("Frames",    60),
]
COL_HEADERS = [c[0] for c in COLS]
COL_WIDTHS  = [c[1] for c in COLS]

ROW_H    = 28
HEADER_H = 34
TITLE_H  = 30
FOOTER_H = 26
MARGIN   = 8
FONT     = cv2.FONT_HERSHEY_SIMPLEX
FS       = 0.42
FT       = 1
C_BORDER = (160, 160, 160)


def build_table_frame(records: Dict[int, PersonRecord],
                      frame_idx: int, fps: float,
                      count_in: int, count_out: int) -> np.ndarray:
    """Génère l'image du tableau temps réel."""
    table_w = sum(COL_WIDTHS) + MARGIN * 2
    n_rows  = max(len(records), 1)
    table_h = TITLE_H + HEADER_H + n_rows * ROW_H + FOOTER_H + 4

    canvas  = np.ones((table_h, table_w, 3), dtype=np.uint8) * 248

    # ── Titre ─────────────────────────────────────────────────────────────────
    cv2.rectangle(canvas, (0,0), (table_w, TITLE_H), (15, 55, 95), -1)
    cv2.putText(canvas,
        f"NEXOR TRACKING  |  frame {frame_idx}  |  entrees {count_in}  sorties {count_out}",
        (MARGIN, TITLE_H - 9), FONT, 0.44, (255,255,255), FT, cv2.LINE_AA)

    # ── En-tête colonnes ──────────────────────────────────────────────────────
    y0 = TITLE_H
    cv2.rectangle(canvas, (0,y0), (table_w, y0+HEADER_H), (30,30,30), -1)
    x = MARGIN
    for w, hdr in zip(COL_WIDTHS, COL_HEADERS):
        cv2.putText(canvas, hdr, (x+3, y0+HEADER_H-9),
                    FONT, FS+0.01, (255,255,255), FT, cv2.LINE_AA)
        cv2.line(canvas, (x+w, y0), (x+w, y0+HEADER_H), (80,80,80), 1)
        x += w
    y0 += HEADER_H

    # ── Lignes de données ─────────────────────────────────────────────────────
    for row_i, (tid, rec) in enumerate(sorted(records.items())):
        y_row = y0 + row_i * ROW_H
        bg    = STATUT_COLOR.get(rec.statut, (240,240,240))
        cv2.rectangle(canvas, (0, y_row), (table_w, y_row+ROW_H), bg, -1)

        # Pastille couleur ID
        id_color = get_color(tid)
        cv2.rectangle(canvas,
                      (MARGIN-2, y_row+5), (MARGIN+6, y_row+ROW_H-5),
                      id_color, -1)

        table_str = f"T{rec.table_id}" if rec.table_id is not None else "—"
        dist_str  = f"{rec.table_dist_px:.0f}" if rec.table_id is not None else "—"
        values = [
            f"#{tid}",
            rec.statut,
            f"{rec.confidence:.2f}",
            table_str,
            dist_str,
            f"{rec.presence_seconds():.1f}s",
            str(len(rec.track_history)),
            f"({rec.position[0]},{rec.position[1]})",
            str(rec.first_frame),
            str(rec.frame_count),
        ]

        x = MARGIN
        for col_i, (w, val) in enumerate(zip(COL_WIDTHS, values)):
            tc = STATUT_TXT_COLOR.get(rec.statut, (20,20,20)) \
                 if col_i == 1 else (20, 20, 20)
            cv2.putText(canvas, val, (x+5, y_row+ROW_H-8),
                        FONT, FS, tc, FT, cv2.LINE_AA)
            cv2.line(canvas, (x+w, y_row), (x+w, y_row+ROW_H), C_BORDER, 1)
            x += w

        cv2.line(canvas, (0, y_row+ROW_H), (table_w, y_row+ROW_H), C_BORDER, 1)

    # ── Footer ────────────────────────────────────────────────────────────────
    fy = y0 + n_rows * ROW_H
    cv2.rectangle(canvas, (0, fy), (table_w, fy+FOOTER_H), (30,30,30), -1)
    visible_now = sum(1 for r in records.values() if r.visible)
    cv2.putText(canvas,
        f"Personnes uniques: {len(records)}   visibles: {visible_now}   "
        f"en_attente: {sum(1 for r in records.values() if r.statut==Statut.EN_ATTENTE)}   "
        f"servis: {sum(1 for r in records.values() if r.statut==Statut.SERVI)}",
        (MARGIN, fy+FOOTER_H-8), FONT, FS, (255,255,255), FT, cv2.LINE_AA)

    cv2.rectangle(canvas, (0,0), (table_w-1, table_h-1), C_BORDER, 1)
    return canvas


# ─────────────────────────────────────────────────────────────
#  EXPORT CSV
# ─────────────────────────────────────────────────────────────

CSV_FIELDNAMES = [
    "frame","track_id","statut","visible","cx","cy","confidence",
    "presence_s","first_frame","last_frame","frame_count","trail_pts",
    "table_id","table_dist_px","crossed_in","crossed_out",
]

def init_csv(path: Path) -> csv.DictWriter:
    f = open(path, "w", newline="", encoding="utf-8")
    writer = csv.DictWriter(f, fieldnames=CSV_FIELDNAMES)
    writer.writeheader()
    return writer

def export_csv_snapshot(writer: csv.DictWriter,
                        records: Dict[int, PersonRecord],
                        frame_idx: int):
    """Écrit une ligne par personne dans le CSV."""
    for rec in records.values():
        writer.writerow(rec.to_csv_row(frame_idx))

def export_json_final(records: Dict[int, PersonRecord], path: Path):
    """Export JSON final — une entrée par personne, pour ROS2 / robot."""
    data = {}
    for tid, rec in records.items():
        data[str(tid)] = {
            "track_id"      : rec.track_id,
            "statut"        : rec.statut,
            "presence_s"    : round(rec.presence_seconds(), 2),
            "frame_count"   : rec.frame_count,
            "first_frame"   : rec.first_frame,
            "last_frame"    : rec.last_frame,
            "last_position" : list(rec.position),
            "confidence"    : round(rec.confidence, 3),
            "table_id"      : rec.table_id,
            "table_dist_px" : round(rec.table_dist_px, 1),
            "crossed_in"    : rec.crossed_in,
            "crossed_out"   : rec.crossed_out,
        }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    print(f"[OK] JSON exporté → {path}")


# ─────────────────────────────────────────────────────────────
#  TRACKING SIMPLE
# ─────────────────────────────────────────────────────────────

def track_simple(source, weights=BEST_V7):
    model = YOLO(weights)
    model.track(
        source=source, tracker="bytetrack.yaml",
        conf=CONF_THRESH, iou=IOU_THRESH,
        classes=[PERSON_CLS], show=True, save=True,
    )
    print("[DONE] Résultats sauvegardés dans runs/track/")


# ─────────────────────────────────────────────────────────────
#  TRACKING AVANCÉ
# ─────────────────────────────────────────────────────────────

def track_advanced(source, weights=BEST_V7, save_video=True):
    """
    Tracking complet avec PersonRecord, tableau temps réel,
    attribution de table, statuts client et export CSV/JSON.
    """
    Path(OUTPUT).mkdir(parents=True, exist_ok=True)

    model = YOLO(weights)
    cap   = cv2.VideoCapture(source if isinstance(source, str) else source)
    if not cap.isOpened():
        raise RuntimeError(f"Impossible d'ouvrir la source : {source}")

    W   = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H   = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    FPS = cap.get(cv2.CAP_PROP_FPS) or 25

    # Sortie vidéo
    writer_vid = None
    if save_video:
        src_name   = Path(source).stem if isinstance(source, str) else "webcam"
        out_path   = str(Path(OUTPUT) / f"{src_name}_tracked.mp4")
        writer_vid = cv2.VideoWriter(
            out_path, cv2.VideoWriter_fourcc(*"mp4v"), FPS, (W, H))

    # ── Export CSV ────────────────────────────────────────────────────────────
    csv_path   = Path(OUTPUT) / "tracking_data.csv"
    csv_writer = init_csv(csv_path)
    csv_file   = open(csv_path, "a", newline="", encoding="utf-8")  # gardé ouvert

    # ── Structures principales ────────────────────────────────────────────────
    records   : Dict[int, PersonRecord] = {}   # track_id → PersonRecord
    count_in  = 0
    count_out = 0
    line_y    = int(H * COUNT_LINE_Y)
    frame_idx = 0

    TABLE_WIN = "NEXOR — Tableau Tracking"
    cv2.namedWindow(TABLE_WIN, cv2.WINDOW_NORMAL)

    print(f"[TRACK] source={source}  poids={weights}")
    print(f"[TRACK] 'q' pour arrêter\n")

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_idx += 1

        # ── Inférence : person + table ────────────────────────────────────────
        results = model.track(
            frame,
            tracker = "bytetrack.yaml",
            conf    = CONF_THRESH,
            iou     = IOU_THRESH,
            classes = TRACK_CLASSES,
            persist = True,
            verbose = False,
        )

        # ── Extraire les tables détectées (classe 1) ──────────────────────────
        tables_detected = []
        if results[0].boxes is not None:
            all_boxes = results[0].boxes
            for i, cls in enumerate(all_boxes.cls.cpu().numpy().astype(int)):
                if cls == TABLE_CLS:
                    bx1,by1,bx2,by2 = map(int, all_boxes.xyxy[i].cpu().numpy())
                    tcx = (bx1+bx2)//2
                    tcy = (by1+by2)//2
                    tid_raw = all_boxes.id
                    tid_val = int(tid_raw[i].item()) if tid_raw is not None else -(i+1)
                    tables_detected.append({"id": tid_val, "cx": tcx, "cy": tcy,
                                            "x1": bx1, "y1": by1, "x2": bx2, "y2": by2})
                    # Dessiner les tables en orange
                    cv2.rectangle(frame, (bx1,by1), (bx2,by2), (0,165,255), 2)
                    cv2.putText(frame, f"Table T{tid_val}", (bx1, by1-6),
                                FONT, 0.5, (0,165,255), 1)

        # ── Marquer visible=False avant la frame (mis à True si réapparu) ────
        for rec in records.values():
            rec.visible = False

        current_ids = set()

        # ── Traiter les personnes (classe 0) ──────────────────────────────────
        if results[0].boxes is not None and results[0].boxes.id is not None:
            all_boxes = results[0].boxes
            box_ids   = all_boxes.id.cpu().numpy().astype(int)

            for i, (cls_val) in enumerate(all_boxes.cls.cpu().numpy().astype(int)):
                if cls_val != PERSON_CLS:
                    continue

                tid  = int(box_ids[i])
                conf = float(all_boxes.conf[i].item())
                x1,y1,x2,y2 = map(int, all_boxes.xyxy[i].cpu().numpy())
                cx = (x1+x2)//2
                cy = (y1+y2)//2

                # ── Créer ou récupérer le PersonRecord ────────────────────────
                if tid not in records:
                    records[tid] = PersonRecord(
                        track_id    = tid,
                        first_frame = frame_idx,
                        first_seen_t= time.time(),
                        statut_since= time.time(),
                    )

                rec = records[tid]
                rec.update_position(cx, cy, conf, (x1,y1,x2,y2), frame_idx)
                current_ids.add(tid)

                # ── Attribution de table ──────────────────────────────────────
                t_id, t_dist = assign_table(cx, cy, tables_detected)
                rec.table_id      = t_id
                rec.table_dist_px = t_dist

                # ── Ligne de comptage ─────────────────────────────────────────
                side_now = "above" if cy < line_y else "below"
                if rec.side:
                    if rec.side == "above" and side_now == "below":
                        count_in      += 1
                        rec.crossed_in = True
                    elif rec.side == "below" and side_now == "above":
                        count_out       += 1
                        rec.crossed_out  = True
                rec.side = side_now

                # ── Mise à jour statut ────────────────────────────────────────
                rec.update_statut()

                color = get_color(tid)

                # Trail
                pts = rec.track_history
                for k in range(1, len(pts)):
                    alpha = k / len(pts)
                    cv2.line(frame, pts[k-1], pts[k], color,
                             max(1, int(3*alpha)))

                # Boîte
                cv2.rectangle(frame, (x1,y1), (x2,y2), color, 2)

                # Label enrichi
                table_lbl = f" T{rec.table_id}" if rec.table_id else ""
                label     = f"#{tid} {rec.statut}{table_lbl} {conf:.2f}"
                (tw,th),_ = cv2.getTextSize(label, FONT, 0.5, 1)
                cv2.rectangle(frame, (x1, y1-th-8), (x1+tw+6, y1), color, -1)
                cv2.putText(frame, label, (x1+3, y1-4), FONT, 0.5, (0,0,0), 1)

        # ── Mettre à jour les personnes disparues ─────────────────────────────
        for rec in records.values():
            if not rec.visible:
                rec.update_statut()

        # ── Ligne de comptage ─────────────────────────────────────────────────
        cv2.line(frame, (0,line_y), (W,line_y), (0,255,255), 2)
        cv2.putText(frame, "Ligne comptage", (10, line_y-8),
                    FONT, 0.48, (0,255,255), 1)

        # ── Overlay stats vidéo ───────────────────────────────────────────────
        visible_now = sum(1 for r in records.values() if r.visible)
        ol = [
            f"Visibles   : {visible_now}",
            f"Total vus  : {len(records)}",
            f"Entrees    : {count_in}",
            f"Sorties    : {count_out}",
            f"Frame      : {frame_idx}",
        ]
        oh = len(ol)*22+8
        cv2.rectangle(frame, (0,0), (220, oh), (0,0,0), -1)
        for i, l in enumerate(ol):
            cv2.putText(frame, l, (6, 18+i*22), FONT, 0.5, (255,255,255), 1)

        # ── Affichage ─────────────────────────────────────────────────────────
        cv2.imshow("NEXOR Tracking — YOLOv8s + ByteTrack", frame)

        table_img = build_table_frame(records, frame_idx, FPS, count_in, count_out)
        cv2.imshow(TABLE_WIN, table_img)

        if writer_vid:
            writer_vid.write(frame)

        # ── Export CSV toutes les N frames ───────────────────────────────────
        if frame_idx % CSV_EXPORT_EVERY == 0:
            export_csv_snapshot(csv_writer, records, frame_idx)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            print("[STOP] Arrêt demandé")
            break

    # ── Dernier snapshot CSV ──────────────────────────────────────────────────
    export_csv_snapshot(csv_writer, records, frame_idx)
    csv_file.close()

    # ── Nettoyage ─────────────────────────────────────────────────────────────
    cap.release()
    if writer_vid:
        writer_vid.release()
    cv2.destroyAllWindows()

    # ── Export JSON final ─────────────────────────────────────────────────────
    json_path = Path(OUTPUT) / "tracking_final.json"
    export_json_final(records, json_path)

    # ── Stats console ─────────────────────────────────────────────────────────
    print("\n" + "="*60)
    print("  STATISTIQUES FINALES")
    print("="*60)
    print(f"  Frames traitées       : {frame_idx}")
    print(f"  Personnes uniques     : {len(records)}")
    print(f"  Entrées (↓ ligne)     : {count_in}")
    print(f"  Sorties (↑ ligne)     : {count_out}")
    print(f"\n  Détail par personne :")
    for tid, rec in sorted(records.items()):
        tbl = f"Table {rec.table_id}" if rec.table_id else "aucune table"
        print(f"    #{tid:<3} | {rec.statut:<12} | {rec.presence_seconds():.1f}s "
              f"| {rec.frame_count} frames | {tbl}")
    print(f"\n  CSV  → {csv_path}")
    print(f"  JSON → {json_path}")
    if writer_vid:
        print(f"  MP4  → {out_path}")
    print("="*60)


# ─────────────────────────────────────────────────────────────
#  TRACKING IMAGE STATIQUE
# ─────────────────────────────────────────────────────────────

def track_image(image_path, weights=BEST_V7):
    model  = YOLO(weights)
    result = model.predict(
        image_path, conf=CONF_THRESH, iou=IOU_THRESH,
        classes=[PERSON_CLS], verbose=False,
    )[0]
    img = cv2.imread(str(image_path))
    for i, box in enumerate(result.boxes):
        x1,y1,x2,y2 = map(int, box.xyxy[0])
        conf  = float(box.conf[0])
        color = get_color(i+1)
        cv2.rectangle(img, (x1,y1), (x2,y2), color, 2)
        cv2.putText(img, f"#{i+1} {conf:.2f}", (x1, y1-6), FONT, 0.6, color, 2)
    out = str(Path(OUTPUT) / f"tracked_{Path(image_path).name}")
    Path(OUTPUT).mkdir(parents=True, exist_ok=True)
    cv2.imwrite(out, img)
    print(f"[IMAGE] {len(result.boxes)} personnes → {out}")


# ─────────────────────────────────────────────────────────────
#  MAIN
# ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="NEXOR Tracking v3 — YOLOv8s + ByteTrack")
    parser.add_argument("--source",  default="0",
                        help="Source : 0=webcam, chemin vidéo, chemin image")
    parser.add_argument("--weights", default=BEST_V7, help="Poids du modèle")
    parser.add_argument("--simple",  action="store_true", help="Mode simple")
    parser.add_argument("--no-save", action="store_true", help="Ne pas sauvegarder la vidéo")
    args = parser.parse_args()

    img_exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    source   = args.source

    if Path(source).suffix.lower() in img_exts:
        track_image(source, args.weights)
    elif args.simple:
        src = int(source) if source.isdigit() else source
        track_simple(src, args.weights)
    else:
        src = int(source) if source.isdigit() else source
        track_advanced(src, args.weights, save_video=not args.no_save)