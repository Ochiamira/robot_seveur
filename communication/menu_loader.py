"""
menu_loader.py
==============
Charge et indexe le menu JSON.
Fournit des helpers pour rechercher des items par nom, catégorie, etc.
"""

import json
from pathlib import Path
from typing import Optional
from config import MENU_PATH, SUPPORTED_LANGS


class MenuLoader:
    def __init__(self, menu_path: Path = MENU_PATH):
        with open(menu_path, encoding="utf-8") as f:
            self._data = json.load(f)

        self.restaurant = self._data["restaurant"]
        self.categories = {c["id"]: c for c in self._data["categories"]}
        self.items       = self._data["items"]
        self.options     = self._data["options_globales"]

        # Index : id → item
        self._by_id = {item["id"]: item for item in self.items}

        # Index : nom (toutes langues, minuscule) → item
        self._by_name: dict[str, dict] = {}
        for item in self.items:
            for lang in SUPPORTED_LANGS:
                nom = item["nom"].get(lang, "").lower().strip()
                if nom:
                    self._by_name[nom] = item

    # ── Recherche ─────────────────────────────────────────────────────────────
    def get_by_id(self, item_id: str) -> Optional[dict]:
        return self._by_id.get(item_id)

    def get_by_name(self, name: str) -> Optional[dict]:
        """Recherche exacte puis partielle sur le nom (toutes langues)."""
        key = name.lower().strip()
        if key in self._by_name:
            return self._by_name[key]
        # Recherche partielle
        for nom, item in self._by_name.items():
            if key in nom or nom in key:
                return item
        return None

    def get_by_category(self, category_id: str) -> list[dict]:
        return [i for i in self.items if i["categorie"] == category_id]

    def get_available(self) -> list[dict]:
        return [i for i in self.items if i["disponible"]]

    def get_vegetarian(self) -> list[dict]:
        return [i for i in self.items if i["vegetarien"] and i["disponible"]]

    def get_vegan(self) -> list[dict]:
        return [i for i in self.items if i["vegan"] and i["disponible"]]

    def get_gluten_free(self) -> list[dict]:
        return [i for i in self.items if i["sans_gluten"] and i["disponible"]]

    def search(self, query: str) -> list[dict]:
        """Recherche textuelle sur nom + description (toutes langues)."""
        q = query.lower().strip()
        results = []
        for item in self.items:
            if not item["disponible"]:
                continue
            for lang in SUPPORTED_LANGS:
                nom  = item["nom"].get(lang, "").lower()
                desc = item["description"].get(lang, "").lower()
                if q in nom or q in desc:
                    if item not in results:
                        results.append(item)
                    break
        return results

    # ── Formatage pour le prompt LLM ──────────────────────────────────────────
    def format_for_prompt(self, lang: str = "fr") -> str:
        """Génère un texte compact du menu pour l'injecter dans le prompt."""
        lines = [f"=== Menu {self.restaurant['nom']} ===\n"]
        for cat_id, cat in self.categories.items():
            cat_nom  = cat["nom"].get(lang, cat_id)
            cat_items = self.get_by_category(cat_id)
            avail     = [i for i in cat_items if i["disponible"]]
            if not avail:
                continue
            lines.append(f"\n[{cat_nom.upper()}]")
            for item in avail:
                nom   = item["nom"].get(lang, item["nom"]["fr"])
                prix  = item["prix"]
                devise = self.restaurant["devise"]
                lines.append(f"  - {nom} : {prix:.3f} {devise}")
        return "\n".join(lines)

    def format_item(self, item: dict, lang: str = "fr") -> str:
        """Formate un item pour affichage."""
        nom    = item["nom"].get(lang, item["nom"]["fr"])
        prix   = item["prix"]
        devise = self.restaurant["devise"]
        return f"{nom} ({prix:.3f} {devise})"


# Singleton global
_menu_instance: Optional[MenuLoader] = None

def get_menu() -> MenuLoader:
    global _menu_instance
    if _menu_instance is None:
        _menu_instance = MenuLoader()
    return _menu_instance


if __name__ == "__main__":
    menu = get_menu()
    print("✅ Menu chargé :")
    print(f"   {len(menu.items)} items  |  {len(menu.categories)} catégories")
    print("\n--- Format prompt (fr) ---")
    print(menu.format_for_prompt("fr"))
    print("\n--- Recherche 'couscous' ---")
    results = menu.search("couscous")
    for r in results:
        print(f"  → {menu.format_item(r)}")
