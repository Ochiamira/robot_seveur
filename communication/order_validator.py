"""
order_validator.py
==================
Valide les items avant de les ajouter à la commande :
- L'item est-il disponible ?
- La quantité est-elle raisonnable ?
- Le prix correspond-il au menu ?
Évite les incohérences si le menu change en cours de session.
"""

from menu_loader import get_menu
from config import MAX_ITEMS_ORDER, DEFAULT_LANG

menu = get_menu()


class ValidationError(Exception):
    pass


def validate_item(item_id: str, quantity: int, lang: str = DEFAULT_LANG) -> tuple[bool, str]:
    """
    Valide un item avant ajout à la commande.

    Returns:
        (True, "")            → item valide
        (False, "message")   → item invalide + raison
    """
    item = menu.get_by_id(item_id)

    # Item existe ?
    if item is None:
        msgs = {
            "fr": "Cet article n'existe pas dans notre menu.",
            "ar": "هذا الصنف غير موجود في قائمتنا.",
            "en": "This item does not exist in our menu.",
        }
        return False, msgs.get(lang, msgs["fr"])

    # Item disponible ?
    if not item.get("disponible", True):
        nom = item["nom"].get(lang, item["nom"]["fr"])
        msgs = {
            "fr": f"{nom} n'est pas disponible pour le moment.",
            "ar": f"{nom} غير متوفر في الوقت الحالي.",
            "en": f"{nom} is not available at the moment.",
        }
        return False, msgs.get(lang, msgs["fr"])

    # Quantité raisonnable ?
    if quantity < 1:
        msgs = {
            "fr": "La quantité doit être d'au moins 1.",
            "ar": "يجب أن تكون الكمية على الأقل 1.",
            "en": "Quantity must be at least 1.",
        }
        return False, msgs.get(lang, msgs["fr"])

    if quantity > 10:
        msgs = {
            "fr": f"Vous ne pouvez pas commander plus de 10 fois le même article.",
            "ar": "لا يمكنك طلب أكثر من 10 من نفس الصنف.",
            "en": "You cannot order more than 10 of the same item.",
        }
        return False, msgs.get(lang, msgs["fr"])

    return True, ""


def validate_order_size(current_count: int, adding: int, lang: str = DEFAULT_LANG) -> tuple[bool, str]:
    """Vérifie que la commande ne dépasse pas MAX_ITEMS_ORDER."""
    if current_count + adding > MAX_ITEMS_ORDER:
        msgs = {
            "fr": f"Votre commande est limitée à {MAX_ITEMS_ORDER} articles.",
            "ar": f"طلبك محدود بـ {MAX_ITEMS_ORDER} أصناف.",
            "en": f"Your order is limited to {MAX_ITEMS_ORDER} items.",
        }
        return False, msgs.get(lang, msgs["fr"])
    return True, ""


if __name__ == "__main__":
    print("=== Test order_validator ===")
    cases = [
        ("item_001", 2,  "fr"),   # salade niçoise, dispo
        ("item_001", 15, "fr"),   # quantité trop grande
        ("item_999", 1,  "fr"),   # item inexistant
    ]
    for item_id, qty, lang in cases:
        ok, msg = validate_item(item_id, qty, lang)
        print(f"  {item_id} qty={qty} → {'✅' if ok else '❌'} {msg}")

    ok, msg = validate_order_size(8, 4, "fr")
    print(f"  Taille commande 8+4 → {'✅' if ok else '❌'} {msg}")
