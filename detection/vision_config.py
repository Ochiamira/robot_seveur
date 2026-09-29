"""
vision_config.py
=================
Config centralisée pour run_vision_pipeline.py — remplace les chemins
Windows codés en dur qu'on trouve éparpillés dans tracking.py,
rtmpose_tracker.py et reid_module.py (ex: "C:/pfe_project/...").

Surchargeable par variables d'environnement, pour ne pas avoir à éditer
le code en passant du PC Windows de dev au Raspberry Pi 5 :

    set NEXOR_YOLO_WEIGHTS=D:\\models\\best.pt      (Windows, dev)
    export NEXOR_YOLO_WEIGHTS=/home/pi/best.pt    (Linux, Raspberry Pi 5)

Ce fichier NE MODIFIE PAS tracking.py / rtmpose_tracker.py / reid_module.py
— ils gardent leurs propres constantes pour un usage standalone/debug
indépendant. Seul run_vision_pipeline.py (le chemin de production) se base
sur CE fichier-ci.
"""

import os
from dataclasses import dataclass
from pathlib import Path


BASE_DIR = Path(__file__).resolve().parent


def _env_str(name: str, default: str) -> str:
    return os.environ.get(name, default)


def _env_float(name: str, default: float) -> float:
    return float(os.environ.get(name, default))


def _env_int(name: str, default: int) -> int:
    return int(os.environ.get(name, default))


@dataclass
class VisionConfig:
    yolo_weights:         str
    output_dir:           str
    video_source_default: str

    pose_model:  str = "RTMPose-s"
    reid_model:  str = "osnet_x0_25"

    conf_thresh: float = 0.35
    iou_thresh:  float = 0.45
    person_cls:  int   = 0
    table_cls:   int   = 1

    count_line_y:  float = 0.5    # position (0-1) de la ligne de comptage
    table_dist_px: float = 200.0  # seuil d'attribution de table — voir note ⚠️ plus bas
    bbox_pad:      int   = 10
    pose_bbox_pad_ratio: float = 0.15
    pose_conf_thresh: float = 0.30
    pose_min_height_px: int = 96
    pose_smoothing_frames: int = 7

    fps:               float = 0.0  # 0 = lire le FPS réel de la source
    cooldown_robot_s:  float = 30.0   # délai mini entre 2 envois du robot vers la même table
    alert_waiting_s:   float = 120.0  # alerte staff si client EN_ATTENTE depuis plus longtemps
    reid_purge_every:  int   = 300    # nettoyage galerie Re-ID toutes les N frames

    @classmethod
    def from_env(cls) -> "VisionConfig":
        return cls(
            yolo_weights=_env_str(
                "NEXOR_YOLO_WEIGHTS",
                str(BASE_DIR / "weights" / "best.pt"),
            ),
            output_dir=_env_str(
                "NEXOR_VISION_OUTPUT",
                str(BASE_DIR / "tracking_output"),
            ),
            video_source_default=_env_str(
                "NEXOR_VIDEO_SOURCE",
                str(BASE_DIR / "videos" / "test1.mp4"),
            ),
            conf_thresh=_env_float("NEXOR_CONF_THRESH", 0.35),
            iou_thresh=_env_float("NEXOR_IOU_THRESH", 0.45),
            table_dist_px=_env_float("NEXOR_TABLE_DIST_PX", 200.0),
            fps=_env_float("NEXOR_FPS", 0.0),
            pose_conf_thresh=_env_float("NEXOR_POSE_CONF", 0.30),
            pose_min_height_px=_env_int("NEXOR_POSE_MIN_HEIGHT", 96),
            pose_smoothing_frames=_env_int("NEXOR_POSE_SMOOTHING_FRAMES", 7),
            cooldown_robot_s=_env_float("NEXOR_COOLDOWN_ROBOT_S", 30.0),
            alert_waiting_s=_env_float("NEXOR_ALERT_WAITING_S", 120.0),
        )

# ⚠️ table_dist_px reste un seuil en PIXELS, pas en unités réelles (mètres).
# Il dépend de la position/résolution de ta caméra — à recalibrer si tu
# changes de caméra ou d'angle. Une vraie calibration (homographie
# pixel→sol) est une amélioration future, hors scope de cette passe.
