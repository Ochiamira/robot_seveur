"""
staff_badge_detector.py
========================
Distingue le personnel des clients par un badge QR porté visiblement
(ex: épinglé sur le tablier) — élimine les faux positifs liés à
l'éclairage ou aux vêtements clients de couleur proche que la détection
couleur seule (staff_detector.py) pouvait produire.

Convention : le QR code doit contenir un texte commençant par le préfixe
STAFF_QR_PREFIX ("NEXOR-STAFF:") suivi d'un identifiant, ex :
"NEXOR-STAFF:marie". Ce préfixe évite qu'un QR quelconque (menu affiché
sur le téléphone d'un client, code promo sur une table) ne soit pris pour
un badge personnel.

Utilisation recommandée : EN COMBINAISON avec staff_detector.py (couleur),
jamais l'un à la place de l'autre :
  - QR lu dans le crop           -> personnel, certain (source de vérité)
  - QR NON lu cette frame        -> ne veut PAS dire "client" à coup sûr
                                     (angle, badge caché par le bras, flou
                                     de mouvement) -> la couleur sert de
                                     filet de sécurité, voir combined_is_staff().
"""

import argparse
from dataclasses import dataclass
from typing import Optional, Tuple

import cv2
import numpy as np

STAFF_QR_PREFIX = "NEXOR-STAFF:"

_detector = cv2.QRCodeDetector()


def detect_staff_qr(crop: np.ndarray) -> Tuple[bool, Optional[str]]:
    """
    Cherche un badge QR dans le crop. Retourne (est_personnel, identifiant_lu).
    Le préfixe STAFF_QR_PREFIX est vérifié pour ignorer tout QR non lié au
    badge (menu, code promo, téléphone d'un client...).
    """
    if crop is None or crop.size == 0:
        return False, None
    try:
        data, points, _ = _detector.detectAndDecode(crop)
    except cv2.error:
        return False, None
    if not data or not data.startswith(STAFF_QR_PREFIX):
        return False, None
    return True, data[len(STAFF_QR_PREFIX):]


def combined_is_staff(crop: np.ndarray, staff_uniform_cfg=None) -> Tuple[bool, str]:
    """
    Décision finale "est-ce du personnel ?", en combinant QR (fiable,
    prioritaire) et couleur (secours si le QR n'est pas lisible cette
    frame précise). Retourne (est_personnel, raison) pour traçabilité/debug.

    staff_uniform_cfg : une staff_detector.StaffUniformConfig calibrée,
    ou None pour désactiver complètement le filet de sécurité couleur
    (QR uniquement — plus strict, plus de faux négatifs si badge mal
    orienté, mais zéro faux positif possible).
    """
    is_qr_staff, staff_id = detect_staff_qr(crop)
    if is_qr_staff:
        return True, f"badge QR détecté ({staff_id})"

    if staff_uniform_cfg is not None:
        from staff_detector import is_staff as is_staff_color
        color_detected, ratio = is_staff_color(crop, staff_uniform_cfg)
        if color_detected:
            return True, f"couleur tenue (ratio={ratio:.2f}) — QR non lu cette frame"

    return False, "aucun signal personnel détecté"


# ── Génération des badges (outil de préparation, hors boucle vidéo) ────────

def generate_badge(staff_id: str, output_path: str) -> None:
    """
    Génère une image de badge QR à imprimer pour un membre du personnel.
    Nécessite le paquet 'qrcode' :
        pip install qrcode[pil] --break-system-packages
    Outil de PRÉPARATION uniquement — jamais appelé dans run_vision_pipeline.py.
    """
    try:
        import qrcode
    except ImportError:
        raise ImportError(
            "Le paquet 'qrcode' est nécessaire pour générer les badges : "
            "pip install qrcode[pil] --break-system-packages"
        )
    content = f"{STAFF_QR_PREFIX}{staff_id}"
    img = qrcode.make(content)
    img.save(output_path)
    print(f"[BADGE] Badge généré pour '{staff_id}' → {output_path}")
    print(f"[BADGE] Contenu encodé : {content}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Génère un badge QR personnel NEXOR")
    parser.add_argument("staff_id", help="Identifiant du membre du personnel (ex: 'marie', 'ahmed')")
    parser.add_argument("--output", default=None, help="Chemin de sortie (défaut: badge_<id>.png)")
    args = parser.parse_args()
    output = args.output or f"badge_{args.staff_id}.png"
    generate_badge(args.staff_id, output)
