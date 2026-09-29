"""
identity_fusion.py
===================
Fusionne PLUSIEURS signaux avant de trancher "nouvelle personne" ou
"personne déjà connue qui revient" — au lieu de se fier au seul score
Re-ID visuel.

Le problème concret : dans reid_module.ReIDGallery.match_and_register(),
la "zone grise" (0.30 < distance cosinus < 0.55) tranche AUJOURD'HUI
systématiquement pour "nouvelle personne", par construction — même si la
personne vient de disparaître 1 seconde derrière un serveur et réapparaît
exactement à la même position.

Signaux fusionnés dans cette zone grise, et SEULEMENT dans cette zone :
  1. Distance Re-ID (embedding visuel OSNet)       — poids dominant (0.70)
  2. Continuité spatiale (position vs dernier point connu) — 0.20
  3. Proximité temporelle (frames depuis disparition)       — 0.10

Principe directeur — le même que le fuzzy matching du menu côté NLP
(seuil strict + marge, jamais de rapprochement hasardeux) :
  - Le Re-ID garde TOUJOURS le dernier mot sur les cas nets (hors zone
    grise) : ce module n'intervient JAMAIS en dehors de cette zone, pour
    ne jamais diverger du comportement déjà validé de reid_module.py.
  - Un signal spatial fort ne peut RATTRAPER un embedding visuel que dans
    sa zone d'incertitude EXISTANTE — jamais au-delà.
  - En cas de signal manquant (pas de position connue, etc.), on retombe
    proprement sur les signaux disponibles plutôt que de deviner.

Ne modifie ni reid_module.py ni customer_manager.py — ce module les
compose. Il réutilise volontairement leurs méthodes internes de mutation
(_update_embedding, _register_new) plutôt que de dupliquer cette logique
ailleurs : la galerie Re-ID reste l'unique source de vérité pour le
stockage des embeddings, on ne fait que choisir QUAND réassigner un ID
existant plutôt que d'en créer un nouveau.
"""

import math
from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import numpy as np

from reid_module import (
    ReIDTracker, ReIDGallery, cosine_distance_matrix,
    REID_THRESHOLD_SAME, REID_THRESHOLD_DIFF, EMBEDDING_DIM,
    extract_crop,
)
from customer_manager import CustomerManager


# ── Config ──────────────────────────────────────────────────────────────

# Distance spatiale (pixels) en-dessous de laquelle une réapparition "au
# même endroit" est plausible. Comme TABLE_DIST_PX (vision_config.py),
# c'est un seuil en PIXELS, pas calibré en unités réelles — dépend de la
# résolution/position caméra. À recalibrer avec la vraie caméra si besoin.
SPATIAL_MAX_PX = 150.0

# Écart de frames max depuis la disparition pour que la continuité
# spatiale compte encore. Au-delà, trop de temps s'est écoulé pour que
# "même position" veuille dire grand-chose.
MAX_FRAME_GAP = 75          # ≈ 3s à 25 fps

# Poids de fusion — le Re-ID visuel reste TOUJOURS dominant, la continuité
# spatio-temporelle ne fait qu'arbitrer les cas déjà incertains.
WEIGHT_REID     = 0.70
WEIGHT_SPATIAL  = 0.20
WEIGHT_TEMPORAL = 0.10

# Score de confiance combiné minimum pour réassigner l'ID dans la zone
# grise du Re-ID. En-dessous, on reste prudent : nouvel ID (même défaut
# que reid_module.py dans le doute).
FUSION_REASSIGN_THRESHOLD = 0.55


@dataclass
class FusionResult:
    canonical_id:   int
    is_new:         bool
    reid_distance:  float
    spatial_score:  Optional[float]   # None si aucun signal spatial disponible
    temporal_score: Optional[float]   # None si aucun signal temporel disponible
    fused_score:    Optional[float]   # None si décision tranchée par le Re-ID seul (hors zone grise)
    reason:         str               # traçabilité — utile pour debug/audit/logs


class IdentityFusion:
    """
    Enveloppe un ReIDTracker + un CustomerManager déjà existants
    (composition, aucune modification des deux). Point d'entrée à utiliser
    dans la boucle vidéo À LA PLACE de reid_tracker.process() directement.

    Usage (dans run_vision_pipeline.py) :

        fusion = IdentityFusion(reid_tracker, customer_manager)
        result = fusion.process(frame, bbox=bbox, byte_id=byte_id,
                                 frame_idx=frame_idx, position=(cx, cy))
        canonical_id = result.canonical_id
    """

    def __init__(self, reid_tracker: ReIDTracker, customer_manager: CustomerManager):
        self.reid_tracker = reid_tracker
        self.gallery: ReIDGallery = reid_tracker.gallery
        self.cm = customer_manager
        self._claim_frame = -1
        self._claimed = {}  # canonical_id -> byte_id, pour une frame donnée

    def _finish(self, result: FusionResult, byte_id: int) -> FusionResult:
        self._claimed[result.canonical_id] = byte_id
        return result

    # ── Point d'entrée haut niveau (drop-in replacement de ReIDTracker.process) ──

    def process(self, frame: np.ndarray, bbox: Tuple[int, int, int, int],
                byte_id: int, frame_idx: int, position: Tuple[int, int]) -> FusionResult:
        """Extrait le crop + l'embedding, puis résout l'identité fusionnée."""
        crop = extract_crop(frame, bbox)
        embedding = self.reid_tracker.extractor.extract(crop)
        result = self.resolve(byte_id, embedding, position, frame_idx)
        self.reid_tracker._stats["processed"] += 1
        self.reid_tracker._stats["new_persons" if result.is_new else "reassigned"] += 1
        return result

    # ── Résolution d'identité (embedding déjà extrait) ─────────────────────

    def resolve(self, byte_id: int, embedding: Optional[np.ndarray],
                position: Tuple[int, int], frame_idx: int) -> FusionResult:
        """
        Résout l'identité d'une détection. Retourne un FusionResult avec
        le canonical_id final ET la traçabilité complète de la décision.
        """
        if frame_idx != self._claim_frame:
            self._claim_frame = frame_idx
            self._claimed.clear()

        # Un ID canonique ne peut représenter deux détections simultanées.
        mapped = self.gallery.canonical_id(byte_id)
        if mapped is not None and self._claimed.get(mapped, byte_id) != byte_id:
            self.gallery._id_map.pop(byte_id, None)

        # Cas 0 — pas d'embedding exploitable (crop trop petit) : rien à
        # fusionner, on retombe sur le comportement Re-ID de base, déjà
        # robuste à ce cas précis (voir ReIDTracker.process).
        if embedding is None:
            fallback_cid = self.gallery.canonical_id(byte_id)
            is_new = fallback_cid is None
            if fallback_cid is None:
                # Ne jamais injecter un vecteur nul dans la galerie : il ne
                # représente aucune apparence et fausserait les futurs matchs.
                fallback_cid = self.gallery.register_without_embedding(byte_id, frame_idx)
            return self._finish(FusionResult(
                fallback_cid, is_new, 1.0, None, None, None,
                reason="pas d'embedding — fallback Re-ID brut"), byte_id)

        # Cas 1 — byte_id déjà suivi cette session : rien à trancher.
        existing_cid = self.gallery.canonical_id(byte_id)
        if existing_cid is not None:
            cid, dist, is_new = self.gallery.match_and_register(byte_id, embedding, frame_idx)
            return self._finish(FusionResult(cid, is_new, dist, None, None, None,
                                 reason="byte_id déjà suivi cette session"), byte_id)

        # Cas 2 — galerie vide : premier client, rien à fusionner.
        if self.gallery.size == 0:
            cid, dist, is_new = self.gallery.match_and_register(byte_id, embedding, frame_idx)
            return self._finish(FusionResult(cid, is_new, dist, None, None, None,
                                 reason="galerie vide — premier client"), byte_id)

        # Cas 3 — distance Re-ID au meilleur candidat.
        canon_ids = [cid for cid in self.gallery._entries
                     if cid not in self._claimed]
        if not canon_ids:
            cid = self.gallery._register_new(byte_id, embedding, frame_idx)
            return self._finish(FusionResult(
                cid, True, 1.0, None, None, None,
                reason="tous les candidats Re-ID sont déjà présents dans la frame"), byte_id)
        embeddings = np.stack([self.gallery._entries[c].embedding for c in canon_ids])
        distances  = cosine_distance_matrix(embedding, embeddings)
        best_idx   = int(np.argmin(distances))
        best_dist  = float(distances[best_idx])
        best_cid   = canon_ids[best_idx]

        # Hors zone grise -> le Re-ID seul tranche déjà, AUCUNE intervention.
        # On garde EXACTEMENT le même comportement que reid_module.py pour
        # ne jamais diverger en dehors du cas précis qu'on cible.
        if best_dist < REID_THRESHOLD_SAME or best_dist > REID_THRESHOLD_DIFF:
            if best_dist < REID_THRESHOLD_SAME:
                self.gallery._id_map[byte_id] = best_cid
                self.gallery._update_embedding(best_cid, embedding, frame_idx)
                cid, dist, is_new = best_cid, best_dist, False
            else:
                cid = self.gallery._register_new(byte_id, embedding, frame_idx)
                dist, is_new = best_dist, True
            return self._finish(FusionResult(
                cid, is_new, dist, None, None, None,
                reason="hors zone grise — décision Re-ID seule (comportement inchangé)"), byte_id)

        # Cas 4 — ZONE GRISE : c'est ici, et SEULEMENT ici, que la fusion agit.
        spatial_score  = self._spatial_score(best_cid, position)
        temporal_score = self._temporal_score(best_cid, frame_idx)

        # Confiance Re-ID normalisée 0-1 SUR LA ZONE GRISE UNIQUEMENT
        # (0.30 -> 1.0, 0.55 -> 0.0), jamais extrapolée au-delà.
        reid_confidence = 1.0 - (best_dist - REID_THRESHOLD_SAME) / (REID_THRESHOLD_DIFF - REID_THRESHOLD_SAME)
        reid_confidence = max(0.0, min(1.0, reid_confidence))

        parts, weights = [reid_confidence], [WEIGHT_REID]
        if spatial_score is not None:
            parts.append(spatial_score); weights.append(WEIGHT_SPATIAL)
        if temporal_score is not None:
            parts.append(temporal_score); weights.append(WEIGHT_TEMPORAL)

        # Renormalise si un signal manque (pas de position connue, etc.) —
        # jamais de score gonflé artificiellement par un poids "gratuit".
        total_w = sum(weights)
        fused_score = sum(p * w for p, w in zip(parts, weights)) / total_w

        if fused_score >= FUSION_REASSIGN_THRESHOLD:
            # Réassignation PRUDENTE : uniquement dans la zone grise du
            # Re-ID, jamais au-delà — le Re-ID garde le dernier mot sur
            # les cas nets (cf. Cas 3 ci-dessus).
            self.gallery._id_map[byte_id] = best_cid
            self.gallery._update_embedding(best_cid, embedding, frame_idx)
            return self._finish(FusionResult(
                best_cid, False, best_dist, spatial_score, temporal_score,
                fused_score, reason=f"zone grise + fusion favorable ({fused_score:.2f})"), byte_id)

        # Fusion pas assez favorable -> on reste prudent : nouvel ID
        # (même défaut que reid_module.py dans le doute).
        cid = self.gallery._register_new(byte_id, embedding, frame_idx)
        return self._finish(FusionResult(
            cid, True, best_dist, spatial_score, temporal_score,
            fused_score, reason=f"zone grise + fusion insuffisante ({fused_score:.2f})"), byte_id)

    # ── Signaux individuels ──────────────────────────────────────────────

    def _spatial_score(self, canonical_id: int, position: Tuple[int, int]) -> Optional[float]:
        """
        Score 0-1 : 1.0 = exactement à la dernière position connue de ce
        candidat, 0.0 = à SPATIAL_MAX_PX ou plus. None si ce candidat n'a
        aucune fiche CustomerManager encore (jamais vu par le manager).
        """
        record = self.cm.get(canonical_id)
        if record is None:
            return None
        last_x, last_y = record.position
        dist = math.hypot(position[0] - last_x, position[1] - last_y)
        return max(0.0, 1.0 - dist / SPATIAL_MAX_PX)

    def _temporal_score(self, canonical_id: int, frame_idx: int) -> Optional[float]:
        """
        Score 0-1 : 1.0 = disparu il y a 0 frame, 0.0 = disparu depuis
        MAX_FRAME_GAP frames ou plus. None si le candidat est encore
        visible cette même frame (pas vraiment "disparu").
        """
        record = self.cm.get(canonical_id)
        if record is None:
            return None
        gap = frame_idx - record.temporal.last_frame
        if gap <= 0:
            return None
        return max(0.0, 1.0 - gap / MAX_FRAME_GAP)

    # ── Diagnostic ─────────────────────────────────────────────────────────

    def summary(self) -> Dict:
        return {
            "gallery_size":      self.gallery.size,
            "spatial_max_px":    SPATIAL_MAX_PX,
            "max_frame_gap":     MAX_FRAME_GAP,
            "fusion_threshold":  FUSION_REASSIGN_THRESHOLD,
            "weights":           {"reid": WEIGHT_REID, "spatial": WEIGHT_SPATIAL, "temporal": WEIGHT_TEMPORAL},
        }


if __name__ == "__main__":
    # Test standalone SANS caméra ni vrais poids OSNet : on simule des
    # embeddings synthétiques pour vérifier la logique de fusion elle-même.
    import numpy as _np

    class _FakeExtractor:
        def extract(self, crop):
            return None  # pas utilisé ici, on appelle resolve() directement

    class _FakeReIDTracker(ReIDTracker):
        def __init__(self):
            self.extractor = _FakeExtractor()
            self.gallery = ReIDGallery()
            from collections import defaultdict
            self._stats = defaultdict(int)

    print("=" * 65)
    print("  IdentityFusion — test standalone (embeddings synthétiques)")
    print("=" * 65)

    tracker = _FakeReIDTracker()
    cm = CustomerManager(fps=25.0, output_dir="/tmp/test_fusion")
    fusion = IdentityFusion(tracker, cm)

    rng = _np.random.default_rng(42)
    base_emb = rng.normal(size=512).astype(_np.float32)
    base_emb /= _np.linalg.norm(base_emb)

    # 1. Premier client (galerie vide) — doit devenir canonical_id=1
    r1 = fusion.resolve(byte_id=10, embedding=base_emb, position=(300, 400), frame_idx=1)
    print(f"[1] {r1.reason} -> canonical_id={r1.canonical_id} is_new={r1.is_new}")
    cm.update(canonical_id=r1.canonical_id, byte_id=10, bbox=(270,350,330,450), conf=0.9,
              frame_idx=1, pose_state="ASSIS", tables=[], line_y=250,
              reid_dist=r1.reid_distance, is_reid_known=not r1.is_new, frame_shape=(480,640,3))

    # 2. Même personne disparaît puis revient, embedding LÉGÈREMENT différent
    #    (zone grise ~0.40) mais À LA MÊME POSITION, peu de frames plus tard
    #    -> la fusion doit la RATTACHER au même canonical_id, pas en créer un nouveau
    noisy_emb = base_emb + rng.normal(scale=0.05, size=512).astype(_np.float32)
    noisy_emb /= _np.linalg.norm(noisy_emb)
    r2 = fusion.resolve(byte_id=11, embedding=noisy_emb, position=(305, 405), frame_idx=10)
    print(f"[2] {r2.reason} -> canonical_id={r2.canonical_id} is_new={r2.is_new} "
          f"(dist={r2.reid_distance:.3f}, spatial={r2.spatial_score}, fused={r2.fused_score})")

    # 3. Nouvelle personne, embedding complètement différent
    #    -> doit être un NOUVEAU canonical_id, quelle que soit la position
    different_emb = rng.normal(size=512).astype(_np.float32)
    different_emb /= _np.linalg.norm(different_emb)
    r3 = fusion.resolve(byte_id=12, embedding=different_emb, position=(305, 405), frame_idx=11)
    print(f"[3] {r3.reason} -> canonical_id={r3.canonical_id} is_new={r3.is_new}")

    print(f"\nRésumé : {fusion.summary()}")
    assert r2.canonical_id == r1.canonical_id, "❌ La fusion aurait dû rattacher la même personne"
    assert r3.canonical_id != r1.canonical_id, "❌ Une personne différente ne doit jamais être rattachée"
    print("\n✅ Comportement attendu validé (rattachement zone grise + rejet net clairement différent)")
