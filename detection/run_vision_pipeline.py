"""
run_vision_pipeline.py
=======================
LE script qui manquait : assemble enfin les 6 modules vision NEXOR en une
seule chaîne qui tourne réellement, de la caméra jusqu'au dashboard staff.

    YOLOv8 (person+table) + ByteTrack
        │
        ▼
    Re-ID (reid_module.ReIDTracker)          → identité stable, même si
        │                                       la personne sort du champ
        ▼
    RTMPose (rtmpose_tracker.RTMPoseBackend) → posture (debout/assis/main levée)
        │
        ▼
    CustomerManager (customer_manager.py)    → une fiche par client
        │
        ▼
    ClientStateMachine (state_machine.py)    → LE SEUL FSM qui fait foi
        │                                       (pas de doublon, cf. audit)
        ▼
    Orchestrator (orchestrator.py)           → décide : envoyer le robot,
        │                                       alerter le staff...
        ▼
    vision_bridge.on_action                  → pousse vers le dashboard
                                                 en temps réel (main.py)

Ne modifie AUCUN des 6 modules existants — se contente de les enchaîner
correctement, en réutilisant leurs briques déjà écrites et testées
(RTMPoseBackend, analyze_pose_state, crop_person, assign_table, ReIDTracker,
CustomerManager, ClientStateMachine, Orchestrator).

tracking.py et rtmpose_tracker.py restent utilisables tels quels comme
outils de DEBUG/visualisation autonomes (leur tableau OpenCV, leurs
exports CSV) — mais ce sont des chemins alternatifs légers, PAS le chemin
de production : leur logique de statut interne (PersonRecord.update_statut
dans tracking.py, la boucle run() de rtmpose_tracker.py) n'est jamais
appelée ici, exprès, pour ne garder qu'UN SEUL FSM faisant foi.

Usage :
    python run_vision_pipeline.py --source 0                  # webcam
    python run_vision_pipeline.py --source chemin/video.mp4
    python run_vision_pipeline.py --source video.mp4 --no-display   # RPi5 headless
"""

import argparse
import logging
import queue
import threading
import time

import cv2
from ultralytics import YOLO

from vision_config import VisionConfig
from staff_detector import StaffUniformConfig
from staff_badge_detector import combined_is_staff
from rtmpose_tracker import (RTMPoseBackend, PoseState, PoseSmoother,
                             analyze_pose_state, crop_person, kpts_to_global)
from reid_module import ReIDTracker
from identity_fusion import IdentityFusion
from customer_manager import CustomerManager
from state_machine import ClientStateMachine, FrameContext
from orchestrator import Orchestrator, OrchestratorConfig
import vision_bridge
from vision_bridge import on_action as dashboard_on_action

logger = logging.getLogger(__name__)
_DEFAULT_STAFF_CONFIG = object()


def _is_live_source(source) -> bool:
    """Return True for webcams and network streams that must not be buffered."""
    if isinstance(source, int):
        return True
    if not isinstance(source, str):
        return False
    return source.lower().startswith(("http://", "https://", "rtsp://", "udp://"))


class LatestFrameCapture:
    """Read a live stream continuously and expose only its newest frame.

    Detection and pose inference can be slower than the camera. A regular
    ``VideoCapture.read()`` loop then consumes buffered, increasingly old
    frames. This reader drains the stream in a background thread and keeps a
    queue of size one, so inference always works on the most recent image.
    """

    def __init__(self, source) -> None:
        self._source = source
        self._cap = self._open_capture()
        self._frames = queue.Queue(maxsize=1)
        self._stop = threading.Event()
        self._thread = None
        self._opened_initially = self._cap.isOpened()

        if self._opened_initially:
            self._thread = threading.Thread(
                target=self._reader_loop,
                name="nexor-latest-frame",
                daemon=True,
            )
            self._thread.start()

    def _open_capture(self):
        cap = cv2.VideoCapture(self._source)
        if cap.isOpened():
            # Best effort: some OpenCV backends ignore this property, while
            # the size-one queue below guarantees the desired behaviour.
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        return cap

    def _replace_latest(self, item) -> None:
        try:
            self._frames.get_nowait()
        except queue.Empty:
            pass
        try:
            self._frames.put_nowait(item)
        except queue.Full:
            pass

    def _reader_loop(self) -> None:
        reconnecting = False
        while not self._stop.is_set():
            try:
                ret, frame = self._cap.read()
            except cv2.error as exc:
                logger.warning("Lecture MJPEG interrompue par OpenCV: %s", exc)
                ret, frame = False, None
            except Exception:
                logger.exception("Erreur inattendue pendant la lecture MJPEG")
                ret, frame = False, None

            if not ret or frame is None:
                if not reconnecting:
                    logger.warning("Flux video interrompu; reconnexion automatique...")
                    reconnecting = True
                try:
                    self._cap.release()
                except Exception:
                    pass
                if self._stop.wait(0.5):
                    return
                self._cap = self._open_capture()
                if not self._cap.isOpened():
                    continue
                continue

            if reconnecting:
                logger.info("Flux video reconnecte")
                reconnecting = False
            self._replace_latest((True, frame))

    def isOpened(self) -> bool:
        return self._opened_initially

    def get(self, prop_id: int) -> float:
        try:
            return self._cap.get(prop_id)
        except cv2.error:
            return 0.0

    def read(self):
        # A Gazebo load spike can temporarily interrupt the MJPEG connection.
        # Give the background reader enough time to reconnect instead of
        # terminating the complete vision pipeline on the first missed frame.
        deadline = time.monotonic() + 30.0
        while not self._stop.is_set():
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                logger.error("Aucune image recue depuis 30 secondes")
                return False, None
            try:
                return self._frames.get(timeout=min(1.0, remaining))
            except queue.Empty:
                continue
        return False, None

    def release(self) -> None:
        self._stop.set()
        try:
            self._cap.release()
        except Exception:
            pass
        if self._thread is not None:
            self._thread.join(timeout=1.0)


def run(cfg: VisionConfig, source, display: bool = True,
        staff_cfg=_DEFAULT_STAFF_CONFIG, max_frames: int = None,
        sim_table_id: int = None) -> None:
    print("=" * 65)
    print("  NEXOR VISION — pipeline complet (production)")
    print("=" * 65)

    if staff_cfg is _DEFAULT_STAFF_CONFIG:
        staff_cfg = StaffUniformConfig()
    staff_mode = ("QR uniquement" if staff_cfg is None
                  else "QR prioritaire + couleur de tenue en secours")
    print(f"[VISION] Détection personnel : {staff_mode}.")

    yolo = YOLO(cfg.yolo_weights)
    pose = RTMPoseBackend(cfg.pose_model)
    reid = ReIDTracker(cfg.reid_model)
    cm = None
    live_source = _is_live_source(source)
    cap = (LatestFrameCapture(source) if live_source
           else cv2.VideoCapture(source if isinstance(source, str) else source))
    if not cap.isOpened():
        raise RuntimeError(f"Impossible d'ouvrir la source : {source}")

    source_fps = float(cap.get(cv2.CAP_PROP_FPS))
    effective_fps = cfg.fps if cfg.fps > 0 else source_fps
    if not (effective_fps > 0 and effective_fps < 240):
        effective_fps = 25.0
    cm = CustomerManager(fps=effective_fps, output_dir=cfg.output_dir,
                         table_dist_max=cfg.table_dist_px)
    vision_bridge.set_customer_manager(cm)
    fusion = IdentityFusion(reid, cm)
    smoother = PoseSmoother(cfg.pose_smoothing_frames)
    orch = Orchestrator(
        config=OrchestratorConfig(cooldown_robot_s=cfg.cooldown_robot_s,
                                  alert_waiting_s=cfg.alert_waiting_s),
        output_dir=cfg.output_dir, on_action=dashboard_on_action)
    vision_bridge.set_orchestrator(orch)
    state_machines: dict[int, ClientStateMachine] = {}
    frame_idx = 0
    is_recorded_source = not live_source
    print(f"[VISION] source={source}  poids={cfg.yolo_weights}")
    print("[VISION] 'q' pour arrêter\n" if display else "[VISION] mode headless (RPi5)\n")

    try:
      while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_idx += 1
        if max_frames is not None and frame_idx > max_frames:
            break
        timestamp_s = (frame_idx / effective_fps if is_recorded_source
                       else time.monotonic())
        cm.mark_all_invisible()
        processed_ids = set()

        results = yolo.track(
            frame, tracker="bytetrack.yaml",
            conf=cfg.conf_thresh, iou=cfg.iou_thresh,
            classes=[cfg.person_cls, cfg.table_cls],
            persist=True, verbose=False,
        )

        # ── Tables détectées cette frame ────────────────────────────────────
        tables_detected = []
        if results[0].boxes is not None:
            boxes = results[0].boxes
            for i, cls in enumerate(boxes.cls.cpu().numpy().astype(int)):
                if cls == cfg.table_cls:
                    bx1, by1, bx2, by2 = map(int, boxes.xyxy[i].cpu().numpy())
                    tid = int(boxes.id[i].item()) if boxes.id is not None else -(i + 1)
                    tables_detected.append({
                        "id": tid, "cx": (bx1 + bx2) // 2, "cy": (by1 + by2) // 2,
                    })

        # ── Personnes détectées cette frame ─────────────────────────────────
        if results[0].boxes is not None and results[0].boxes.id is not None:
            boxes   = results[0].boxes
            box_ids = boxes.id.cpu().numpy().astype(int)

            for i, cls_val in enumerate(boxes.cls.cpu().numpy().astype(int)):
                if cls_val != cfg.person_cls:
                    continue

                byte_id = int(box_ids[i])
                conf    = float(boxes.conf[i].item())
                x1, y1, x2, y2 = map(int, boxes.xyxy[i].cpu().numpy())
                bbox = (x1, y1, x2, y2)
                cx, cy = (x1 + x2) // 2, (y1 + y2) // 2

                # 0. Filtre personnel — QR prioritaire (certain), couleur en
                #    secours si le badge n'est pas lisible cette frame (angle,
                #    flou de mouvement). AVANT toute création d'identité/fiche
                #    client, pour qu'un serveur ne pollue ni la galerie Re-ID
                #    ni CustomerManager ni le FSM.
                crop = crop_person(frame, bbox, pad=cfg.pose_bbox_pad_ratio)
                staff_detected, staff_reason = combined_is_staff(crop, staff_cfg)
                if staff_detected:
                    if display:
                        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 165, 255), 2)
                        cv2.putText(frame, f"STAFF ({staff_reason})", (x1, y1 - 8),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 165, 255), 1)
                    continue

                # 1. Identité : Re-ID + fusion spatio-temporelle en zone grise
                #    -> canonical_id STABLE (même personne = même ID, toujours)
                fusion_result = fusion.process(
                    frame, bbox=bbox, byte_id=byte_id,
                    frame_idx=frame_idx, position=(cx, cy))
                canonical_id = fusion_result.canonical_id
                reid_dist    = fusion_result.reid_distance
                is_new       = fusion_result.is_new

                # 2. Posture, via RTMPose sur le crop déjà extrait à l'étape 0
                if crop is not None and crop.shape[0] >= cfg.pose_min_height_px:
                    kpts_crop = pose.infer(crop)
                else:
                    kpts_crop = None
                pose_raw = analyze_pose_state(kpts_crop, cfg.pose_conf_thresh)
                pose_state = smoother.update(canonical_id, pose_raw)
                kpts = (kpts_to_global(kpts_crop, bbox, pad=cfg.pose_bbox_pad_ratio,
                                       frame_shape=frame.shape)
                        if kpts_crop is not None else None)

                # 4. Fiche client à jour
                record = cm.update(
                    canonical_id=canonical_id, byte_id=byte_id,
                    bbox=bbox, conf=conf, frame_idx=frame_idx,
                    keypoints=kpts, pose_state=pose_state, tables=tables_detected,
                    line_y=int(frame.shape[0] * cfg.count_line_y),
                    reid_dist=reid_dist, is_reid_known=not is_new,
                    frame_shape=frame.shape,
                )
                if record is None:
                    continue
                # Controlled Gazebo test: bind the simulated customer to a
                # fixed restaurant table ID. ByteTrack IDs are temporary and
                # must never be interpreted as map coordinates or table IDs.
                if sim_table_id is not None:
                    record.table.update(
                        sim_table_id,
                        record.table.table_dist_px,
                        record.table.table_cx,
                        record.table.table_cy,
                    )
                processed_ids.add(canonical_id)
                table_id = record.table_id
                table_dist = record.table.table_dist_px

                # 5. LE SEUL FSM qui fait foi (pas de doublon de statut ici,
                #    contrairement à tracking.py/rtmpose_tracker.py standalone)
                if canonical_id not in state_machines:
                    state_machines[canonical_id] = ClientStateMachine(fps=effective_fps)

                ctx = FrameContext(
                    frame_idx=frame_idx, fps=effective_fps, timestamp_s=timestamp_s,
                    pose_state=pose_state, keypoints=kpts, position=(cx, cy),
                    table_id=table_id, table_dist_px=table_dist,
                    reid_distance=reid_dist, is_reid_known=not is_new,
                    robot_arrived=orch.is_robot_at_table(table_id),
                    visible=True,
                )
                new_statut = state_machines[canonical_id].update(ctx)
                record.statut = new_statut.value

                # 6. Décisions (envoyer le robot, alerter...) + push dashboard
                orch.process(record, state_machines[canonical_id])

                if display:
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 200, 0), 2)
                    label = f"#{canonical_id} {record.statut} {pose_state}"
                    cv2.putText(frame, label, (x1, y1 - 8),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 200, 0), 1)

        # Faire progresser aussi les FSM absentes : indispensable pour PARTI.
        for canonical_id, sm in list(state_machines.items()):
            if canonical_id in processed_ids:
                continue
            record = cm.get(canonical_id)
            if record is None:
                continue
            old_status = record.statut
            status = sm.update(FrameContext(
                frame_idx=frame_idx, fps=effective_fps, timestamp_s=timestamp_s,
                pose_state=PoseState.INCONNU, position=record.position,
                table_id=record.table_id, table_dist_px=record.table.table_dist_px,
                reid_distance=record.reid_distance,
                is_reid_known=record.is_reid_known, visible=False))
            record.statut = status.value
            if record.statut != old_status:
                orch.process(record, sm)
                if record.statut == "PARTI":
                    smoother.remove(canonical_id)

        # Actions périodiques (nettoyage missions, priorités) — 1x/frame
        orch.tick()
        if frame_idx % cfg.reid_purge_every == 0:
            reid.purge(frame_idx)

        if display:
            cv2.imshow("NEXOR Vision — pipeline complet", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                print("[STOP] Arrêt demandé")
                break
    finally:
        cap.release()
        if display:
            cv2.destroyAllWindows()
        orch.print_summary()
        cm.print_summary()
        print(f"\n[VISION] Log Orchestrator → {orch.export_log()}")
        print(f"[VISION] Stats Re-ID       → {reid.summary()}")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    parser = argparse.ArgumentParser(description="NEXOR Vision — pipeline complet (production)")
    parser.add_argument("--source", default=None, help="0=webcam, ou chemin vidéo")
    parser.add_argument("--weights", default=None, help="Poids YOLO (sinon vision_config.py / variable d'env)")
    parser.add_argument("--no-display", action="store_true", help="Mode headless (Raspberry Pi 5 sans écran)")
    parser.add_argument("--staff-hsv", default=None,
                         help="Bornes HSV calibrées 'Hl,Sl,Vl,Hu,Su,Vu' (voir staff_detector.py --calibrate). "
                              "Sans cette option, bornes par défaut NON calibrées — à éviter en usage réel.")
    parser.add_argument("--staff-qr-only", action="store_true",
                         help="Désactive le repli couleur : personnel reconnu UNIQUEMENT par badge QR "
                              "(zéro faux positif possible, mais raté si badge mal orienté/caché).")
    parser.add_argument("--max-frames", type=int, default=None,
                        help="Limite de frames, utile pour un test rapide")
    parser.add_argument(
        "--sim-table-id", type=int, choices=(1, 2, 3, 4), default=None,
        help="Test Gazebo controle: associe le client a l'ID fixe de table",
    )
    args = parser.parse_args()

    cfg = VisionConfig.from_env()
    if args.weights:
        cfg.yolo_weights = args.weights

    staff_cfg = None if args.staff_qr_only else StaffUniformConfig()
    if args.staff_hsv and not args.staff_qr_only:
        hl, sl, vl, hu, su, vu = map(int, args.staff_hsv.split(","))
        staff_cfg = StaffUniformConfig(hsv_lower=(hl, sl, vl), hsv_upper=(hu, su, vu))

    source_arg = args.source if args.source is not None else cfg.video_source_default
    src = int(source_arg) if source_arg.isdigit() else source_arg
    run(cfg, src, display=not args.no_display, staff_cfg=staff_cfg,
        max_frames=args.max_frames, sim_table_id=args.sim_table_id)
