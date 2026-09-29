"""
NEXOR Vision — Module Re-ID
============================
Résout le problème : une personne sort du champ caméra et revient
→ ByteTrack lui donne un NOUVEL ID → le système la croit nouveau client.

Solution : pour chaque nouveau track_id, on extrait un vecteur
d'apparence (embedding 512-d via OSNet/torchreid). Si la distance
cosinus entre ce vecteur et un ID connu est < SEUIL, c'est la même
personne → on réassigne l'ancien ID.

Structure :
  ReIDExtractor   — charge OSNet, extrait le vecteur d'une image crop
  ReIDGallery     — stocke les embeddings connus, fait le matching
  ReIDTracker     — point d'entrée : wrap autour du tracking ByteTrack

Compatibilité : Windows · Linux · Kaggle · Colab
Dépendances    : pip install torchreid opencv-python numpy
"""

import cv2
import time
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field
from collections import defaultdict

import torch
import torch.nn.functional as F

# ─────────────────────────────────────────────────────────────
#  CONFIG
# ─────────────────────────────────────────────────────────────

REID_THRESHOLD_SAME  = 0.30   # distance cosinus < seuil → même personne
REID_THRESHOLD_DIFF  = 0.55   # distance cosinus > seuil → personne différente
# zone grise entre 0.30 et 0.55 → incertain (on ne réassigne pas)

EMBEDDING_DIM  = 512           # dimension du vecteur OSNet
CROP_H, CROP_W = 256, 128      # taille d'entrée OSNet (standard ReID)
GALLERY_MAX_AGE = 300          # frames max pour garder un embedding sans le voir
GALLERY_UPDATE_ALPHA = 0.3     # EMA : nouveau_emb = α*nouveau + (1-α)*ancien

# Padding autour du bbox avant crop (pixels)
BBOX_PAD = 10


# ─────────────────────────────────────────────────────────────
#  EXTRACTEUR D'EMBEDDING (OSNet via torchreid)
# ─────────────────────────────────────────────────────────────

class ReIDExtractor:
    """
    Charge OSNet (ou un fallback MobileNet) et extrait
    un vecteur L2-normalisé de dimension 512 pour un crop BGR.
    """

    def __init__(self, model_name: str = "osnet_x0_25", device: str = "auto"):
        """
        model_name : "osnet_x0_25"  → très rapide (RPi5 compatible)
                     "osnet_x0_5"   → équilibre vitesse/précision
                     "osnet_x1_0"   → précision max
        device     : "auto" détecte CUDA sinon CPU
        """
        self.device = self._get_device(device)
        self.model  = None
        self.name   = model_name
        self._load(model_name)

    def _get_device(self, device: str) -> torch.device:
        if device == "auto":
            return torch.device("cuda" if torch.cuda.is_available() else "cpu")
        return torch.device(device)

    def _load(self, model_name: str):
        """Charge OSNet via torchreid (téléchargement auto des poids)."""
        try:
            import torchreid
            self.model = torchreid.models.build_model(
                name       = model_name,
                num_classes= 1000,        # poids pré-entraînés Market-1501
                pretrained = True,
            )
            self.model.eval()
            self.model.to(self.device)
            # Supprime le classifieur final — on veut les features, pas les classes
            self.model.classifier = torch.nn.Identity()
            print(f"[ReID] OSNet chargé : {model_name} sur {self.device}")

        except ImportError:
            print("[ReID] torchreid non installé → fallback MobileNetV3")
            print("       Pour OSNet : pip install torchreid")
            self._load_mobilenet_fallback()

        except Exception as e:
            print(f"[ReID] Erreur chargement OSNet : {e} → fallback MobileNetV3")
            self._load_mobilenet_fallback()

    def _load_mobilenet_fallback(self):
        """
        Fallback : MobileNetV3-Small pré-entraîné ImageNet.
        Moins précis qu'OSNet mais fonctionne sans torchreid.
        L'embedding est réduit à 512-d via average pooling.
        """
        from torchvision import models
        base = models.mobilenet_v3_small(weights="DEFAULT")
        # On retire le classifieur et on garde les features (576-d)
        self.model = torch.nn.Sequential(
            base.features,
            base.avgpool,
            torch.nn.Flatten(),
            torch.nn.Linear(576, EMBEDDING_DIM),   # projection → 512
        )
        self.model.eval()
        self.model.to(self.device)
        self.name = "mobilenet_v3_small_fallback"
        print(f"[ReID] Fallback MobileNetV3 chargé sur {self.device}")

    def preprocess(self, crop_bgr: np.ndarray) -> torch.Tensor:
        """
        BGR crop (H, W, 3)  →  tensor (1, 3, 256, 128) normalisé ImageNet.
        """
        img = cv2.resize(crop_bgr, (CROP_W, CROP_H))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0

        mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        std  = np.array([0.229, 0.224, 0.225], dtype=np.float32)
        img  = (img - mean) / std

        tensor = torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0)
        return tensor.to(self.device)

    @torch.no_grad()
    def extract(self, crop_bgr: np.ndarray) -> Optional[np.ndarray]:
        """
        Extrait et retourne le vecteur L2-normalisé (512,) en float32.
        Retourne None si le crop est invalide.
        """
        if crop_bgr is None or crop_bgr.size == 0:
            return None
        if crop_bgr.shape[0] < 20 or crop_bgr.shape[1] < 10:
            return None   # trop petit pour être fiable

        try:
            tensor  = self.preprocess(crop_bgr)
            feat    = self.model(tensor)            # (1, D)
            feat    = F.normalize(feat, p=2, dim=1) # L2-normalisation
            return feat.squeeze(0).cpu().numpy().astype(np.float32)
        except Exception as e:
            print(f"[ReID] Erreur extraction : {e}")
            return None


# ─────────────────────────────────────────────────────────────
#  DISTANCE COSINUS
# ─────────────────────────────────────────────────────────────

def cosine_distance(v1: np.ndarray, v2: np.ndarray) -> float:
    """
    Distance cosinus entre deux vecteurs L2-normalisés.
    Si les vecteurs sont déjà normalisés : dist = 1 - dot(v1, v2)
    Plage : 0.0 (identiques) → 2.0 (opposés), typiquement 0.0–1.0
    """
    dot = float(np.dot(v1, v2))
    dot = max(-1.0, min(1.0, dot))   # clip numérique
    return 1.0 - dot


def cosine_distance_matrix(query: np.ndarray,
                            gallery: np.ndarray) -> np.ndarray:
    """
    Distance cosinus entre un vecteur query (512,) et une galerie (N, 512).
    Retourne un tableau (N,) de distances.
    Opération vectorisée : rapide même pour N > 1000.
    """
    dots = gallery @ query          # (N,)
    dots = np.clip(dots, -1.0, 1.0)
    return 1.0 - dots


# ─────────────────────────────────────────────────────────────
#  GALERIE D'EMBEDDINGS
# ─────────────────────────────────────────────────────────────

@dataclass
class GalleryEntry:
    """Un enregistrement dans la galerie : embedding + métadonnées."""
    original_id   : int                          # track_id ByteTrack original
    canonical_id  : int                          # ID final (après réassignation)
    embedding     : np.ndarray                   # vecteur (512,)
    last_seen_frame: int = 0
    hit_count     : int  = 1                     # combien de fois vu
    embeddings_ema: np.ndarray = field(default=None)  # moyenne exponentielle


class ReIDGallery:
    """
    Stocke les embeddings de toutes les personnes vues.
    Associe chaque nouveau track_id à un canonical_id (stable).

    Algorithme :
      1. Nouveau track_id détecté par ByteTrack
      2. On extrait son embedding
      3. On calcule la distance cosinus avec tous les embeddings connus
      4. Si min_distance < SEUIL_SAME → réassigne l'ancien canonical_id
      5. Sinon → nouveau canonical_id
      6. Mise à jour EMA de l'embedding pour robustesse
    """

    def __init__(self):
        self._entries       : Dict[int, GalleryEntry] = {}  # canonical_id → entry
        self._id_map        : Dict[int, int] = {}            # byte_id → canonical_id
        self._next_canonical: int = 1

    @property
    def size(self) -> int:
        return len(self._entries)

    def canonical_id(self, byte_id: int) -> Optional[int]:
        """Retourne le canonical_id pour un track ByteTrack (None si inconnu)."""
        return self._id_map.get(byte_id)

    def match_and_register(self,
                           byte_id  : int,
                           embedding: np.ndarray,
                           frame_idx: int) -> Tuple[int, float, bool]:
        """
        Point d'entrée principal.

        Paramètres
        ----------
        byte_id   : track_id assigné par ByteTrack cette frame
        embedding : vecteur (512,) L2-normalisé du crop
        frame_idx : numéro de frame courant

        Retour
        ------
        (canonical_id, best_distance, is_new_person)
          canonical_id  : ID stable final
          best_distance : distance cosinus du meilleur match (1.0 si galerie vide)
          is_new_person : True si c'est une nouvelle personne jamais vue
        """
        # ── Cas 1 : on a déjà vu ce byte_id dans cette session ───────────────
        if byte_id in self._id_map:
            cid = self._id_map[byte_id]
            if cid not in self._entries:
                self._entries[cid] = GalleryEntry(
                    original_id=byte_id, canonical_id=cid,
                    embedding=embedding.copy(), last_seen_frame=frame_idx,
                    embeddings_ema=embedding.copy())
            else:
                self._update_embedding(cid, embedding, frame_idx)
            return cid, 0.0, False

        # ── Cas 2 : nouveau byte_id → cherche dans la galerie ────────────────
        if self._entries:
            canon_ids  = list(self._entries.keys())
            embeddings = np.stack([self._entries[c].embedding for c in canon_ids])

            distances = cosine_distance_matrix(embedding, embeddings)
            best_idx  = int(np.argmin(distances))
            best_dist = float(distances[best_idx])
            best_cid  = canon_ids[best_idx]

            if best_dist < REID_THRESHOLD_SAME:
                # ── Réassignation : c'est la même personne qui revient ────────
                self._id_map[byte_id] = best_cid
                self._update_embedding(best_cid, embedding, frame_idx)
                print(f"[ReID] byte_id={byte_id} → canonical #{best_cid} "
                      f"(dist={best_dist:.3f}, même personne)")
                return best_cid, best_dist, False

            elif best_dist > REID_THRESHOLD_DIFF:
                # ── Clairement une nouvelle personne ──────────────────────────
                cid = self._register_new(byte_id, embedding, frame_idx)
                print(f"[ReID] byte_id={byte_id} → nouveau canonical #{cid} "
                      f"(dist={best_dist:.3f})")
                return cid, best_dist, True

            else:
                # ── Zone d'incertitude : on crée un nouvel ID prudemment ──────
                cid = self._register_new(byte_id, embedding, frame_idx)
                print(f"[ReID] byte_id={byte_id} → #{cid} (incertain dist={best_dist:.3f})")
                return cid, best_dist, True

        else:
            # ── Galerie vide : premier individu ───────────────────────────────
            cid = self._register_new(byte_id, embedding, frame_idx)
            return cid, 1.0, True

    def _register_new(self, byte_id: int, embedding: np.ndarray,
                      frame_idx: int) -> int:
        """Crée un nouvel entrée dans la galerie."""
        cid = self._next_canonical
        self._next_canonical += 1
        self._entries[cid] = GalleryEntry(
            original_id    = byte_id,
            canonical_id   = cid,
            embedding      = embedding.copy(),
            last_seen_frame= frame_idx,
            embeddings_ema = embedding.copy(),
        )
        self._id_map[byte_id] = cid
        return cid

    def register_without_embedding(self, byte_id: int, frame_idx: int) -> int:
        """Réserve un ID pour un crop inexploitable sans polluer la galerie."""
        cid = self._next_canonical
        self._next_canonical += 1
        self._id_map[byte_id] = cid
        return cid

    def _update_embedding(self, canonical_id: int,
                          new_embedding: np.ndarray,
                          frame_idx: int):
        """
        Met à jour l'embedding via EMA pour robustesse aux changements
        d'éclairage, de pose, de point de vue.
        embedding = α * nouveau + (1-α) * ancien  (puis re-normalisation L2)
        """
        entry = self._entries[canonical_id]
        entry.last_seen_frame = frame_idx
        entry.hit_count      += 1

        # EMA
        updated = (GALLERY_UPDATE_ALPHA * new_embedding
                   + (1 - GALLERY_UPDATE_ALPHA) * entry.embeddings_ema)
        # Re-normalisation L2
        norm = np.linalg.norm(updated)
        if norm > 1e-9:
            updated /= norm
        entry.embeddings_ema = updated
        entry.embedding      = updated  # on utilise l'EMA comme référence

    def purge_old(self, current_frame: int):
        """Supprime les embeddings trop anciens (personnages partis depuis longtemps)."""
        to_remove = [
            cid for cid, e in self._entries.items()
            if (current_frame - e.last_seen_frame) > GALLERY_MAX_AGE
        ]
        for cid in to_remove:
            del self._entries[cid]
            # Nettoyer aussi le mapping byte_id → canonical
            stale_bytes = [b for b, c in self._id_map.items() if c == cid]
            for b in stale_bytes:
                del self._id_map[b]


# ─────────────────────────────────────────────────────────────
#  UTILITAIRE — CROP AVEC PADDING
# ─────────────────────────────────────────────────────────────

def extract_crop(frame: np.ndarray,
                 bbox: Tuple[int, int, int, int],
                 pad: int = BBOX_PAD) -> Optional[np.ndarray]:
    """
    Extrait le crop BGR d'une personne avec padding.
    bbox = (x1, y1, x2, y2) en pixels image.
    """
    H, W = frame.shape[:2]
    x1, y1, x2, y2 = bbox
    x1 = max(0, x1 - pad)
    y1 = max(0, y1 - pad)
    x2 = min(W, x2 + pad)
    y2 = min(H, y2 + pad)
    if x2 <= x1 or y2 <= y1:
        return None
    return frame[y1:y2, x1:x2].copy()


# ─────────────────────────────────────────────────────────────
#  TRACKER AVEC RE-ID INTÉGRÉ
# ─────────────────────────────────────────────────────────────

class ReIDTracker:
    """
    Wrapper à utiliser dans la boucle principale du tracking.
    À chaque frame, pour chaque détection YOLOv8 trackée :
      1. Extraire le crop
      2. Calculer l'embedding OSNet
      3. Demander à la galerie le canonical_id stable
      4. Utiliser canonical_id à la place du track_id ByteTrack

    Usage minimal :
        tracker = ReIDTracker()
        # dans la boucle vidéo :
        canonical_id, dist, is_new = tracker.process(
            frame, bbox=(x1,y1,x2,y2), byte_id=tid
        )
        # utiliser canonical_id pour PersonRecord
    """

    def __init__(self, model_name: str = "osnet_x0_25"):
        print("[ReID] Initialisation...")
        self.extractor = ReIDExtractor(model_name)
        self.gallery   = ReIDGallery()
        self._stats    = defaultdict(int)

    def process(self,
                frame    : np.ndarray,
                bbox     : Tuple[int, int, int, int],
                byte_id  : int,
                frame_idx: int = 0) -> Tuple[int, float, bool]:
        """
        Traite une détection et retourne le canonical_id stable.

        Retour : (canonical_id, cosine_distance, is_new_person)
        """
        crop      = extract_crop(frame, bbox)
        embedding = self.extractor.extract(crop)

        if embedding is None:
            # Pas d'embedding : réserver un ID sans ajouter de vecteur nul.
            fallback_cid = self.gallery.canonical_id(byte_id)
            is_new = fallback_cid is None
            if fallback_cid is None:
                fallback_cid = self.gallery.register_without_embedding(byte_id, frame_idx)
            self._stats["processed"] += 1
            self._stats["new_persons" if is_new else "reassigned"] += 1
            return fallback_cid, 1.0, is_new

        cid, dist, is_new = self.gallery.match_and_register(
            byte_id, embedding, frame_idx)

        self._stats["processed"] += 1
        if is_new:
            self._stats["new_persons"] += 1
        else:
            self._stats["reassigned"] += 1

        return cid, dist, is_new

    def purge(self, frame_idx: int):
        """À appeler toutes les N frames pour nettoyer la galerie."""
        self.gallery.purge_old(frame_idx)

    def summary(self) -> Dict:
        return {
            "gallery_size"  : self.gallery.size,
            "total_seen"    : self.gallery._next_canonical - 1,
            "processed"     : self._stats["processed"],
            "new_persons"   : self._stats["new_persons"],
            "reassigned"    : self._stats["reassigned"],
        }


# ─────────────────────────────────────────────────────────────
#  INTÉGRATION DANS rtmpose_tracker.py
# ─────────────────────────────────────────────────────────────
#
#  Dans run() de rtmpose_tracker.py, remplacer :
#
#    records : Dict[int, PersonRecord] = {}
#
#  par :
#
#    from reid_module import ReIDTracker
#    reid = ReIDTracker(model_name="osnet_x0_25")
#    records : Dict[int, PersonRecord] = {}
#
#  Puis dans la boucle, après avoir récupéré tid de ByteTrack :
#
#    canonical_id, reid_dist, is_new = reid.process(
#        frame, bbox=(x1,y1,x2,y2), byte_id=tid, frame_idx=frame_idx)
#
#    if canonical_id not in records:
#        records[canonical_id] = PersonRecord(
#            track_id     = canonical_id,
#            first_frame  = frame_idx,
#            first_seen_t = time.time(),
#            statut_since = time.time(),
#        )
#    rec = records[canonical_id]
#    rec.reid_distance = reid_dist     # ajouter ce champ à PersonRecord
#    rec.is_new_person = is_new
#
#  Nettoyer la galerie toutes les 100 frames :
#    if frame_idx % 100 == 0:
#        reid.purge(frame_idx)
#
# ─────────────────────────────────────────────────────────────


# ─────────────────────────────────────────────────────────────
#  TEST STANDALONE
# ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Test Re-ID sur une vidéo")
    parser.add_argument("--source",  default="0",
                        help="0=webcam, chemin vidéo")
    parser.add_argument("--weights", default="C:/pfe_project/detection/runs/resto_v7/weights/best.pt",
                        help="Poids YOLOv8")
    parser.add_argument("--model",   default="osnet_x0_25",
                        help="Modèle Re-ID (osnet_x0_25 / osnet_x0_5 / osnet_x1_0)")
    args = parser.parse_args()

    from ultralytics import YOLO
    yolo  = YOLO(args.weights)
    reid  = ReIDTracker(model_name=args.model)
    src   = int(args.source) if args.source.isdigit() else args.source
    cap   = cv2.VideoCapture(src)

    FONT = cv2.FONT_HERSHEY_SIMPLEX
    frame_idx = 0
    np.random.seed(42)

    print("[TEST] Démarrage — 'q' pour quitter")

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_idx += 1

        results = yolo.track(
            frame, tracker="bytetrack.yaml",
            conf=0.35, iou=0.45, classes=[0],
            persist=True, verbose=False,
        )

        if results[0].boxes is not None and results[0].boxes.id is not None:
            for i, box in enumerate(results[0].boxes):
                byte_id = int(box.id[0].item())
                x1,y1,x2,y2 = map(int, box.xyxy[0].cpu().numpy())

                # ── Re-ID ──────────────────────────────────────────────────────
                canonical_id, dist, is_new = reid.process(
                    frame, (x1,y1,x2,y2), byte_id, frame_idx)

                # Couleur stable par canonical_id
                np.random.seed(canonical_id * 7 + 13)
                color = tuple(np.random.randint(80, 255, 3).tolist())

                # Boîte
                thickness = 3 if is_new else 2
                cv2.rectangle(frame, (x1,y1), (x2,y2), color, thickness)

                # Label
                label = f"#{canonical_id}  d={dist:.2f}"
                if is_new:
                    label += "  NOUVEAU"
                (tw,th),_ = cv2.getTextSize(label, FONT, 0.5, 1)
                cv2.rectangle(frame, (x1,y1-th-8), (x1+tw+6,y1), color, -1)
                cv2.putText(frame, label, (x1+3,y1-4), FONT, 0.5, (0,0,0), 1)

        # Stats overlay
        s = reid.summary()
        overlay = [
            f"Galerie : {s['gallery_size']} personnes",
            f"Nouvelles: {s['new_persons']}",
            f"Reidentifiées: {s['reassigned']}",
            f"Frame : {frame_idx}",
            f"Modele : {reid.extractor.name[:20]}",
        ]
        cv2.rectangle(frame, (0,0), (240, len(overlay)*22+8), (0,0,0), -1)
        for i, l in enumerate(overlay):
            cv2.putText(frame, l, (6,18+i*22), FONT, 0.48, (255,255,255), 1)

        cv2.imshow("NEXOR — Re-ID test", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()

    print("\n" + "="*50)
    print("  RÉSULTAT Re-ID")
    print("="*50)
    for k, v in reid.summary().items():
        print(f"  {k:<20} : {v}")
    print("="*50)
