"""
staff_detector.py
==================
Distingue le personnel des clients par la couleur de la tenue (tablier),
SANS réentraîner YOLO — analyse le crop déjà extrait par crop_person()
pour chaque détection "personne", avant de créer un client.

Pourquoi la couleur plutôt qu'un badge/QR : ça ne dépend d'aucune action
du personnel (pas de badge à porter, oublier ou perdre) — juste la tenue
de travail déjà obligatoire pendant le service.

⚠️ LIMITE ASSUMÉE, à connaître avant d'activer en production : sensible à
l'éclairage (une salle très éclairée le midi vs tamisée le soir peut
décaler les couleurs perçues) et aux vêtements clients qui se
rapprocheraient par coïncidence de la couleur de la tenue. Les bornes HSV
par défaut ci-dessous sont des EXEMPLES (tenue sombre générique) — PAS
des valeurs prêtes à l'emploi. Calibre-les avec calibrate_from_samples()
sur de vraies photos de LA tenue réelle avant toute utilisation en salle.
"""

import argparse
from dataclasses import dataclass
from typing import List, Optional, Tuple

import cv2
import numpy as np


@dataclass
class StaffUniformConfig:
    # Bornes HSV (convention OpenCV : H 0-179, S/V 0-255) de la couleur
    # de la tenue. Valeurs d'exemple = tenue sombre générique (tablier
    # noir/gris foncé) — À RECALIBRER, voir calibrate_from_samples().
    hsv_lower: Tuple[int, int, int] = (0, 0, 0)
    hsv_upper: Tuple[int, int, int] = (180, 60, 60)

    # Zone du crop analysée : le TORSE uniquement (fraction de la hauteur
    # du crop, du haut vers le bas) — pas la tête (visage, cheveux) ni les
    # jambes (souvent un jean, couleur peu discriminante entre client et
    # personnel).
    torso_top_ratio:    float = 0.25
    torso_bottom_ratio: float = 0.70

    # Proportion minimale de pixels "couleur tenue" dans la zone torse
    # pour classer la détection comme personnel plutôt que client.
    min_ratio: float = 0.35


def _torso_region(crop: np.ndarray, cfg: StaffUniformConfig) -> Optional[np.ndarray]:
    if crop is None or crop.size == 0:
        return None
    h = crop.shape[0]
    y1 = int(h * cfg.torso_top_ratio)
    y2 = int(h * cfg.torso_bottom_ratio)
    if y2 <= y1:
        return None
    return crop[y1:y2, :]


def is_staff(crop: np.ndarray, cfg: StaffUniformConfig = StaffUniformConfig()) -> Tuple[bool, float]:
    """
    Retourne (est_personnel, ratio_pixels_tenue).

    N'analyse QUE la zone torse du crop, pour limiter les faux positifs
    venant de l'arrière-plan ou des jambes (jean souvent similaire chez
    client ET personnel, peu discriminant).
    """
    region = _torso_region(crop, cfg)
    if region is None:
        return False, 0.0

    hsv  = cv2.cvtColor(region, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, np.array(cfg.hsv_lower), np.array(cfg.hsv_upper))
    ratio = float(np.count_nonzero(mask)) / mask.size

    return ratio >= cfg.min_ratio, ratio


def calibrate_from_samples(image_paths: List[str],
                            torso_top_ratio: float = 0.25,
                            torso_bottom_ratio: float = 0.70) -> StaffUniformConfig:
    """
    Aide à choisir les bornes HSV à partir de quelques photos RECADRÉES
    sur une personne en tenue de travail (crops déjà centrés sur la
    personne, torse visible — pas besoin d'un cadrage parfait).

    Usage :
        cfg = calibrate_from_samples(["staff1.jpg", "staff2.jpg", "staff3.jpg"])
        # copie hsv_lower/hsv_upper affichés dans ta config de production

    Idéalement 5-10 échantillons, sous différents éclairages (matin,
    midi, soir) pour une plage robuste. Utilise les percentiles 5-95
    plutôt que min/max pour ignorer les pixels aberrants (reflets,
    ombres, léger mauvais cadrage).
    """
    all_pixels = []
    tmp_cfg = StaffUniformConfig(torso_top_ratio=torso_top_ratio, torso_bottom_ratio=torso_bottom_ratio)

    for path in image_paths:
        img = cv2.imread(path)
        if img is None:
            print(f"⚠️  Impossible de lire {path}, ignoré")
            continue
        region = _torso_region(img, tmp_cfg)
        if region is None:
            continue
        hsv = cv2.cvtColor(region, cv2.COLOR_BGR2HSV)
        all_pixels.append(hsv.reshape(-1, 3))

    if not all_pixels:
        raise ValueError("Aucun échantillon exploitable — vérifie les chemins d'image")

    pixels = np.concatenate(all_pixels, axis=0)
    lower = np.percentile(pixels, 5, axis=0).astype(int)
    upper = np.percentile(pixels, 95, axis=0).astype(int)

    cfg = StaffUniformConfig(
        hsv_lower=tuple(int(v) for v in lower),
        hsv_upper=tuple(int(v) for v in upper),
        torso_top_ratio=torso_top_ratio,
        torso_bottom_ratio=torso_bottom_ratio,
    )

    print(f"[CALIBRATION] {len(image_paths)} échantillons, {pixels.shape[0]} pixels analysés")
    print(f"[CALIBRATION] hsv_lower = {cfg.hsv_lower}")
    print(f"[CALIBRATION] hsv_upper = {cfg.hsv_upper}")
    print("[CALIBRATION] ⚠️  Teste ensuite is_staff() sur des crops QUI NE SONT PAS dans "
          "l'échantillon (personnel ET clients) avant de valider en production.")
    return cfg


def evaluate(cfg: StaffUniformConfig, staff_images: List[str], client_images: List[str]) -> None:
    """
    Valide une config calibrée sur des exemples frais : combien de vrais
    positifs (personnel bien détecté) et de faux positifs (client
    confondu avec personnel) — À FAIRE avant toute mise en production,
    jamais se fier aux seuls chiffres de calibrate_from_samples().
    """
    def _rate(paths, expected):
        correct = 0
        for p in paths:
            img = cv2.imread(p)
            if img is None:
                continue
            detected, ratio = is_staff(img, cfg)
            ok = detected == expected
            correct += int(ok)
            print(f"  {p}: {'PERSONNEL' if detected else 'CLIENT'} (ratio={ratio:.2f}) "
                  f"{'✅' if ok else '❌'}")
        return correct, len(paths)

    print("\n[ÉVALUATION] Échantillons personnel (attendu: PERSONNEL) :")
    c1, n1 = _rate(staff_images, True)
    print("\n[ÉVALUATION] Échantillons clients (attendu: CLIENT) :")
    c2, n2 = _rate(client_images, False)

    print(f"\n[ÉVALUATION] Personnel bien reconnu : {c1}/{n1}")
    print(f"[ÉVALUATION] Clients bien exclus     : {c2}/{n2}")
    if n1 and c1 / n1 < 0.8:
        print("⚠️  Trop de personnel non reconnu — élargis hsv_lower/hsv_upper ou baisse min_ratio")
    if n2 and c2 / n2 < 0.8:
        print("⚠️  Trop de clients confondus avec le personnel — resserre les bornes HSV "
              "ou augmente min_ratio. Envisage la solution badge/QR si la couleur ne suffit pas.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Calibration détection personnel par couleur de tenue")
    parser.add_argument("--calibrate", nargs="+", metavar="IMG",
                         help="Photos de personnel (crops) pour calculer les bornes HSV")
    parser.add_argument("--eval-staff", nargs="+", metavar="IMG", default=[],
                         help="Photos de personnel FRAÎCHES pour valider la config calibrée")
    parser.add_argument("--eval-clients", nargs="+", metavar="IMG", default=[],
                         help="Photos de clients pour vérifier qu'ils ne sont PAS confondus")
    args = parser.parse_args()

    if not args.calibrate:
        parser.error("Utilise --calibrate avec au moins 3-5 photos de personnel en tenue")

    cfg = calibrate_from_samples(args.calibrate)

    if args.eval_staff or args.eval_clients:
        evaluate(cfg, args.eval_staff, args.eval_clients)
