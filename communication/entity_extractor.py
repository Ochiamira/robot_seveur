"""
entity_extractor.py
===================
Extrait les entités nommées du texte :
  - items du menu (plats, boissons, desserts)
  - quantités (chiffres + mots en ar/fr/en)
  - modificateurs (sans, avec, bien cuit, etc.)
  - taille (petit, moyen, grand)

Améliorations v2 :
  - Match exact prioritaire
  - Match partiel par mots significatifs (> 3 lettres)
  - Match par mot principal (le plus long du nom)
  - Match phonétique simple (coffee → café espresso)
  - break correct après match → pas de doublons
  - remaining_text mis à jour après chaque match
  - Alias multilingues pour les items courants
"""

import re
import difflib
import unicodedata
from menu_loader import get_menu
from config import SUPPORTED_LANGS

menu = get_menu()

# Similarité minimale pour proposer une suggestion ("vous vouliez peut-être ?").
# Volontairement STRICT (0.78) : l'arabe non-vocalisé a un alphabet réduit et
# des préfixes/suffixes très fréquents (ال، ة), donc difflib peut donner un
# ratio trompeusement élevé entre deux mots SANS AUCUN rapport (ex: mesuré en
# test réel, "الكفتة" (kofta) vs "الشوكولاتة" (chocolat) → ratio 0.62 à cause
# du "ال..ة" partagé). Un seuil bas transformerait cette fonction en la même
# source d'hallucination qu'on cherche à éviter. On exige aussi un écart net
# avec le 2e meilleur candidat, pour ne suggérer que quand c'est net.
_SUGGESTION_MIN_RATIO = 0.78
_SUGGESTION_MIN_MARGIN = 0.08  # écart minimum avec le 2e meilleur score


def _suggest_closest_item(word: str, lang: str) -> dict | None:
    """
    Cherche, PAR SIMILARITÉ TEXTUELLE UNIQUEMENT (difflib, déterministe),
    le nom de plat du menu le plus proche d'un mot non reconnu.
    Ne renvoie qu'une SUGGESTION affichée au client — jamais ajoutée
    automatiquement à la commande. Objectif : distinguer "le client a
    mal prononcé / le STT a mal transcrit" de "ce plat n'existe vraiment
    pas", sans jamais faire ce choix à sa place (contrairement à un LLM
    qui pourrait improviser une correspondance). Seuil volontairement
    strict + marge avec le 2e candidat pour ne jamais halluciner un
    rapprochement hasardeux (voir note ci-dessus, cas testé sur "kofta").
    """
    if not word or len(word) < 4:
        return None

    best_item, best_ratio, second_ratio = None, 0.0, 0.0
    for item in menu.get_available():
        nom = item["nom"].get(lang, item["nom"].get("fr", ""))
        if not nom:
            continue
        item_best = max(
            difflib.SequenceMatcher(None, word.lower(), candidate.lower()).ratio()
            for candidate in [nom] + nom.split()
        )
        if item_best > best_ratio:
            second_ratio = best_ratio
            best_item, best_ratio = item, item_best
        elif item_best > second_ratio:
            second_ratio = item_best

    if (best_item and best_ratio >= _SUGGESTION_MIN_RATIO
            and (best_ratio - second_ratio) >= _SUGGESTION_MIN_MARGIN):
        return best_item
    return None

# ── Alias supplémentaires par langue ──────────────────────────────────────────
# Quand le client dit "café" sans préciser → mappe vers l'item le plus proche
_ALIASES = {
    # (alias_en_minuscules, lang) → item_id
    ("café", "fr"):          "item_009",   # café espresso
    ("cafe", "fr"):          "item_009",
    ("un café", "fr"):       "item_009",
    ("coffee", "en"):        "item_009",
    ("a coffee", "en"):      "item_009",
    ("قهوة", "ar"):          "item_009",
    ("couscous", "fr"):      "item_003",
    ("كسكسي", "ar"):         "item_003",
    ("couscous", "en"):      "item_003",
    ("jus orange", "fr"):    "item_012",
    ("orange juice", "en"):  "item_012",
    ("eau", "fr"):           "item_014",
    ("water", "en"):         "item_014",
    ("ماء", "ar"):           "item_014",
    ("salade", "fr"):        "item_001",
    ("salad", "en"):         "item_001",
    ("سلطة", "ar"):          "item_001",
    ("thé", "fr"):           "item_011",
    ("tea", "en"):           "item_011",
    ("شاي", "ar"):           "item_011",
    ("chocolat", "fr"):      "item_015",
    ("chocolate", "en"):     "item_015",
    ("شوكولاتة", "ar"):      "item_015",
    # ("pizza", "fr"): retiré — la pizza existe désormais au menu sous plusieurs
    # noms (Margherita, 4 fromages, thon & poivrons) ; un alias générique "pizza"
    # -> None forcerait à tort "n'existe pas" même quand une pizza précise a été
    # correctement reconnue par ailleurs. Le match par nom complet gère déjà
    # "pizza margherita" etc. ; sans alias, "une pizza" (générique, sans nom
    # précis) tombera proprement sur la suggestion floue / "inconnu" au lieu
    # d'une fausse certitude d'absence.
    ("burger", "fr"):        None,
    ("hamburger", "en"):     None,
}

# ── Modificateurs par langue ───────────────────────────────────────────────────
_MODIFIERS = {
    "fr": {
        "sans sucre":  "sans_sucre",
        "sans gluten": "sans_gluten",
        "sans lait":   "sans_lait",
        "bien cuit":   "bien_cuit",
        "à point":     "a_point",
        "saignant":    "saignant",
        "pimenté":     "pimente",
        "pas pimenté": "sans_piment",
        "allégé":      "allege",
        "chaud":       "chaud",
        "froid":       "froid",
        "sans":        "exclusion",
        "avec":        "inclusion",
    },
    "ar": {
        "بدون سكر":  "sans_sucre",
        "حار":       "pimente",
        "بارد":      "froid",
        "ساخن":      "chaud",
        "بدون":      "exclusion",
        "مع":        "inclusion",
    },
    "en": {
        "no sugar":  "sans_sucre",
        "sugar free":"sans_sucre",
        "spicy":     "pimente",
        "not spicy": "sans_piment",
        "hot":       "chaud",
        "cold":      "froid",
        "without":   "exclusion",
        "with":      "inclusion",
        "well done": "bien_cuit",
        "medium":    "a_point",
        "rare":      "saignant",
        "no":        "exclusion",
    },
}

# ── Tailles ────────────────────────────────────────────────────────────────────
_SIZES = {
    "fr": {
        "petit":  "small",
        "petite": "small",
        "moyen":  "medium",
        "moyenne":"medium",
        "grand":  "large",
        "grande": "large",
    },
    "ar": {
        "صغير": "small",
        "وسط":  "medium",
        "كبير": "large",
    },
    "en": {
        "small":  "small",
        "medium": "medium",
        "large":  "large",
        "big":    "large",
        "tall":   "small",
        "grande": "large",
        "venti":  "large",
    },
}

# ── Quantités ──────────────────────────────────────────────────────────────────
_NUMBERS = {
    "fr": {
        "un": 1, "une": 1, "deux": 2, "trois": 3,
        "quatre": 4, "cinq": 5, "six": 6, "sept": 7,
        "huit": 8, "neuf": 9, "dix": 10,
    },
    "ar": {
        "واحد": 1, "واحدة": 1, "اثنين": 2, "اثنان": 2,
        "ثلاثة": 3, "أربعة": 4, "خمسة": 5, "ستة": 6,
        "سبعة": 7, "ثمانية": 8, "تسعة": 9, "عشرة": 10,
    },
    "en": {
        "one": 1, "two": 2, "three": 3,
        "four": 4, "five": 5, "six": 6, "seven": 7,
        "eight": 8, "nine": 9, "ten": 10,
    },
}


def _extract_quantity(text: str, lang: str) -> int:
    """Extrait la quantité du texte (chiffre ou mot)."""
    # Chiffre numérique
    m = re.search(r"\b(\d+)\b", text)
    if m:
        return int(m.group(1))
    # Mot
    for word, val in _NUMBERS.get(lang, {}).items():
        if re.search(rf"\b{re.escape(word)}\b", text, re.UNICODE):
            return val
    return 1


def _extract_size(text: str, lang: str) -> str | None:
    """Extrait la taille du texte."""
    for word, size_id in _SIZES.get(lang, {}).items():
        if re.search(rf"\b{re.escape(word)}\b", text, re.UNICODE):
            return size_id
    return None


def _extract_modifiers(text: str, lang: str) -> list:
    """Extrait les modificateurs du texte."""
    found = []
    # Trie par longueur décroissante → "sans sucre" avant "sans"
    sorted_mods = sorted(
        _MODIFIERS.get(lang, {}).items(),
        key=lambda x: len(x[0]), reverse=True
    )
    matched_phrases = []
    for phrase, mod_id in sorted_mods:
        if phrase in text and mod_id not in found:
            if mod_id in {"exclusion", "inclusion"}:
                continue
            found.append(mod_id)
            matched_phrases.append(phrase)

    def canonical(value: str) -> str:
        value = unicodedata.normalize("NFKD", value)
        value = "".join(c for c in value if not unicodedata.combining(c))
        return re.sub(r"[^\w]+", "_", value.lower(), flags=re.UNICODE).strip("_")

    generic_patterns = {
        "fr": [("sans", r"\bsans\s+(?:(?:le|la|les|du|de la|des)\s+)?([\w'-]+)"),
               ("avec", r"\bavec\s+(?:(?:le|la|les|du|de la|des|un|une)\s+)?([\w'-]+)")],
        "en": [("sans", r"\b(?:without|no)\s+(?:the\s+)?([\w'-]+)"),
               ("avec", r"\bwith\s+(?:(?:the|a|an)\s+)?([\w'-]+)")],
        "ar": [("sans", r"\bبدون\s+([\w'-]+)"), ("avec", r"\bمع\s+([\w'-]+)")],
    }
    for prefix, pattern in generic_patterns.get(lang, []):
        match = re.search(pattern, text, flags=re.IGNORECASE | re.UNICODE)
        if not match:
            continue
        target = canonical(match.group(1))
        modifier = f"{prefix}_{target}" if target else prefix
        if not any(phrase in match.group(0) for phrase in matched_phrases) and modifier not in found:
            found.append(modifier)
    return found


def _find_item_by_alias(text: str, lang: str) -> dict | None:
    """Cherche un item via les alias multilingues."""
    # Trie par longueur décroissante → "a coffee" avant "coffee"
    sorted_aliases = sorted(
        [(k, v) for k, v in _ALIASES.items() if k[1] == lang],
        key=lambda x: len(x[0][0]), reverse=True
    )
    for (alias, alias_lang), item_id in sorted_aliases:
        if alias in text:
            if item_id is None:
                return {"NOT_IN_MENU": alias}
            item = menu.get_by_id(item_id)
            if item:
                return item
    return None


def _find_item_in_text(text: str, remaining: str, item: dict) -> tuple[bool, str]:
    """
    Essaie de matcher un item dans le texte restant — UNIQUEMENT via les
    stratégies FORTES (sans ambiguïté possible) :
    1. Match exact du nom complet
    2. Tous les mots significatifs (> 3 lettres) présents

    La stratégie faible ("mot principal seul") est gérée séparément par
    _find_item_by_principal_word, en deux passes (voir _extract_from_clause) :
    sans ça, un menu avec plusieurs variantes d'un même plat de base
    ("Couscous agneau" / "Couscous poulet", "Café espresso" / "Café crème" /
    "Cappuccino"...) peut faire matcher la MAUVAISE variante — le mot
    générique partagé ("couscous", "café") est consommé par le premier item
    testé (ordre de tri par longueur de nom, arbitraire), avant que le bon
    item n'ait sa chance de faire un match exact complet (bug observé en
    test réel : "couscous agneau" → "Couscous poulet").
    """
    for l in SUPPORTED_LANGS:
        nom = item["nom"].get(l, "").lower().strip()
        if not nom:
            continue

        # 1. Match exact
        if nom in remaining:
            return True, remaining.replace(nom, " ", 1)

        # 2. Tous les mots significatifs présents
        mots = [m for m in nom.split() if len(m) > 3]
        if mots and all(m in remaining for m in mots):
            # Retire le premier mot trouvé pour éviter les réutilisations
            new_remaining = remaining
            for m in mots:
                new_remaining = new_remaining.replace(m, " ", 1)
            return True, new_remaining

    return False, remaining


_generic_words_cache: set | None = None


def _generic_words() -> set:
    """
    Mots à NE JAMAIS utiliser comme "mot principal" pour le match faible :
    tout mot présent dans le nom de PLUSIEURS items du menu (toutes langues
    confondues) est par définition trop générique pour désigner un plat
    précis. Calculé dynamiquement depuis le menu (pas de liste en dur
    "pizza"/"couscous"/... — reste correct si le menu change).

    Exemple concret qui a motivé ce fix : sans lui, "une pizza" (sans
    préciser laquelle) matchait "Pizza Reine" par accident (le mot "pizza"
    gagnait le tie-break de longueur face à "reine" dans _principal_word),
    ajoutant silencieusement la mauvaise pizza à la commande. Avec ce
    filtre, "pizza" est exclu (partagé par 7+ items) -> plus aucun match
    faible sur "une pizza" seule (on tombe sur la clarification), tandis
    que "reine", "margherita", etc. restent utilisables (uniques).
    """
    global _generic_words_cache
    if _generic_words_cache is not None:
        return _generic_words_cache

    from collections import Counter
    counts = Counter()
    for item in menu.get_available():
        for l in SUPPORTED_LANGS:
            nom = item["nom"].get(l, "").lower().strip()
            for word in nom.split():
                if len(word) > 4:
                    counts[word] += 1
    _generic_words_cache = {w for w, c in counts.items() if c > 1}
    return _generic_words_cache


def _principal_word(item: dict) -> str | None:
    """Le mot principal (le plus long, > 4 lettres, NON générique) du nom
    d'un item, toutes langues confondues — utilisé UNIQUEMENT pour la
    stratégie de match faible. Un mot partagé par plusieurs items du menu
    (ex: "pizza", "couscous", "café") est écarté : voir _generic_words()."""
    generic = _generic_words()
    best = None
    for l in SUPPORTED_LANGS:
        nom = item["nom"].get(l, "").lower().strip()
        longs = [m for m in nom.split() if len(m) > 4 and m not in generic]
        if longs:
            candidate = max(longs, key=len)
            if best is None or len(candidate) > len(best):
                best = candidate
    return best


def _find_items_by_principal_word(remaining: str, candidate_items: list) -> tuple[dict | None, str]:
    """
    Stratégie FAIBLE, deuxième passe : cherche un item par son seul mot
    principal (ex: "couscous", "café", "pizza"). N'accepte le match QUE
    s'il est NON AMBIGU, c'est-à-dire qu'un seul item candidat (parmi ceux
    pas encore trouvés) a ce mot principal présent dans le texte restant.
    Si plusieurs items différents partagent ce mot (plusieurs variantes de
    couscous/café/pizza au menu), on préfère ne rien deviner plutôt que de
    risquer la mauvaise variante — le client devra préciser (ou tombera sur
    la suggestion floue / "article non trouvé" gérée en amont).
    """
    matches = []
    for item in candidate_items:
        word = _principal_word(item)
        if word and word in remaining:
            matches.append((item, word))

    if len(matches) != 1:
        return None, remaining

    item, word = matches[0]
    return item, remaining.replace(word, " ", 1)


# ── Séparateurs de clauses (pour isoler quantité/item par segment) ────────────
# Ex: "deux couscous et un café" -> ["deux couscous", "un café"]
# Découper évite qu'une quantité destinée à un item "fuite" vers un autre item
# de la même phrase (bug historique : quantité globale appliquée à tous les items).
_CLAUSE_SPLIT = {
    "fr": r"(?:\b(?:et aussi|et|ainsi que|avec ça)\b|,)",
    "ar": r"(?:\b(?:و|كذلك)\b|,|،)",
    "en": r"(?:\b(?:and also|and|as well as)\b|,)",
}


def _split_clauses(text: str, lang: str) -> list:
    pattern = _CLAUSE_SPLIT.get(lang, _CLAUSE_SPLIT["fr"])
    parts = re.split(pattern, text, flags=re.UNICODE)
    return [p.strip() for p in parts if p and p.strip()]


def _extract_from_clause(clause: str, lang: str) -> dict:
    """Extrait quantité/taille/modificateurs/item pour UN SEUL segment de phrase."""
    quantity  = _extract_quantity(clause, lang)
    modifiers = _extract_modifiers(clause, lang)
    size      = _extract_size(clause, lang)

    remaining = clause
    found_items = []
    found_ids   = set()
    unknown = []
    suggestions = {}

    # ── Nom complet du catalogue D'ABORD (le plus spécifique gagne) ──────────
    # Priorité au nom complet ("café au lait") sur un alias générique ("café")
    # pour éviter qu'un alias substring court ne "vole" le match avant le nom
    # complet plus précis (ex: "un café" matchait dans "un café au lait").
    #
    # PASSE 1 — matches FORTS uniquement (nom exact ou tous les mots
    # significatifs) : jamais ambigu, donc l'ordre de test entre items
    # n'a pas d'importance ici.
    all_items = menu.get_available()
    items_by_length = sorted(
        all_items,
        key=lambda i: max(len(i["nom"].get(l, "")) for l in SUPPORTED_LANGS),
        reverse=True,
    )
    for item in items_by_length:
        if item["id"] in found_ids:
            continue
        matched, new_remaining = _find_item_in_text(clause, remaining, item)
        if matched:
            found_items.append(item)
            found_ids.add(item["id"])
            remaining = new_remaining

    # PASSE 2 — match FAIBLE (mot principal seul), seulement pour ce qui
    # n'a pas matché en passe 1, et seulement si non ambigu (voir
    # _find_items_by_principal_word). Fait APRÈS la passe 1 complète pour
    # que toutes les variantes exactes aient eu leur chance avant qu'un mot
    # générique partagé ne soit consommé par la mauvaise variante.
    if not found_items:
        remaining_candidates = [i for i in items_by_length if i["id"] not in found_ids]
        weak_item, new_remaining = _find_items_by_principal_word(remaining, remaining_candidates)
        if weak_item:
            found_items.append(weak_item)
            found_ids.add(weak_item["id"])
            remaining = new_remaining

    # ── Alias en secours (uniquement si rien trouvé via le nom complet) ─────
    if not found_items:
        alias_item = _find_item_by_alias(remaining, lang)
        if alias_item:
            if "NOT_IN_MENU" in alias_item:
                unknown.append(alias_item["NOT_IN_MENU"])
            elif alias_item["id"] not in found_ids:
                found_items.append(alias_item)
                found_ids.add(alias_item["id"])
                for l in SUPPORTED_LANGS:
                    nom = alias_item["nom"].get(l, "").lower()
                    if nom and nom in remaining:
                        remaining = remaining.replace(nom, " ", 1)
                        break

    # ── Suggestion floue en dernier recours ───────────────────────────────
    # Si rien n'a matché du tout (ni nom complet, ni alias), on essaie de
    # repérer LE mot le plus probable désignant un plat (le plus long mot
    # significatif restant) et on cherche une suggestion par similarité,
    # uniquement pour l'affichage ("vous vouliez peut-être ... ?").
    # On ne touche jamais found_items ici : aucune commande n'est passée
    # sur la base d'une suggestion, seul un humain (le client) confirme.
    if not found_items and not unknown:
        _stopwords = {
            "fr": {"un", "une", "le", "la", "les", "de", "du", "des", "sur",
                   "avec", "sans", "je", "veux", "voudrais", "s'il", "vous",
                   "plait", "plaît", "svp", "commander", "et", "côté"},
            "ar": {"هل", "يمكنني", "الحصول", "على", "مع", "من", "فضلك",
                   "الجانب", "و"},
            "en": {"i", "want", "would", "like", "can", "get", "have",
                   "please", "the", "a", "an", "on", "side", "with"},
        }.get(lang, set())
        candidates = [
            w for w in re.findall(r"\w+", remaining, re.UNICODE)
            if len(w) > 2 and w not in _stopwords
        ]
        for word in sorted(candidates, key=len, reverse=True):
            suggestion = _suggest_closest_item(word, lang)
            if suggestion:
                suggestions[word] = suggestion
            else:
                unknown.append(word)
            break  # un seul mot-candidat par clause suffit (le plus long)

    def text_position(item: dict) -> int:
        positions = []
        for language in SUPPORTED_LANGS:
            name = item["nom"].get(language, "").lower().strip()
            if not name:
                continue
            exact = clause.find(name)
            if exact >= 0:
                positions.append(exact)
            for word in name.split():
                if len(word) > 3:
                    pos = clause.find(word)
                    if pos >= 0:
                        positions.append(pos)
        return min(positions) if positions else len(clause)

    found_items.sort(key=text_position)
    return {
        "items":       found_items,
        "quantity":    quantity,
        "modifiers":   modifiers,
        "size":        size,
        "unknown":     unknown,
        "suggestions": suggestions,
    }


def extract(text: str, lang: str = "fr") -> dict:
    """
    Extrait toutes les entités du texte STT normalisé.

    Le texte est découpé en clauses (sur "et"/"and"/"و"/virgule) afin que
    chaque item reçoive SA PROPRE quantité/taille/modificateurs, au lieu
    d'appliquer une unique quantité globale à tous les items de la phrase.

    Returns:
        {
          "items"     : [{"item": dict, "quantity": int, "modifiers": list, "size": str}],
          "quantity"  : int,   # quantité du 1er segment (rétro-compatibilité)
          "modifiers" : list,  # modificateurs du 1er segment (rétro-compatibilité)
          "size"      : str | None,
          "raw_text"  : str,
          "unknown"   : list,   # items demandés mais absents du menu
        }
    """
    text_lower = text.lower().strip()
    clauses = _split_clauses(text_lower, lang) or [text_lower]

    found_items = []
    unknown     = []
    suggestions = {}

    for clause in clauses:
        parsed = _extract_from_clause(clause, lang)
        for item in parsed["items"]:
            extracted = {
                "item":      item,
                "quantity":  parsed["quantity"],
                "modifiers": parsed["modifiers"],
                "size":      parsed["size"],
            }
            existing = next((e for e in found_items if (
                e["item"]["id"] == item["id"]
                and e["size"] == extracted["size"]
                and e["modifiers"] == extracted["modifiers"]
            )), None)
            if existing:
                existing["quantity"] += extracted["quantity"]
            else:
                found_items.append(extracted)
        unknown.extend(parsed["unknown"])
        suggestions.update(parsed.get("suggestions", {}))

    result = {
        "items":       found_items,
        "quantity":    found_items[0]["quantity"]  if found_items else _extract_quantity(text_lower, lang),
        "modifiers":   found_items[0]["modifiers"] if found_items else _extract_modifiers(text_lower, lang),
        "size":        found_items[0]["size"]      if found_items else _extract_size(text_lower, lang),
        "raw_text":    text,
        "unknown":     unknown,
        "suggestions": suggestions,   # {mot_non_reconnu: item_menu_le_plus_proche}
    }
    return result


if __name__ == "__main__":
    tests = [
        ("je voudrais deux couscous et un café au lait", "fr"),
        ("un magret de canard bien cuit s'il vous plaît", "fr"),
        ("i want a large orange juice please", "en"),
        ("i want a coffee please", "en"),
        ("نحب كسكسي واحد و قهوة من فضلك", "ar"),
        ("نحب نطلب كسكسي من فضلك", "ar"),
        ("ajoute une salade niçoise", "fr"),
        ("je veux un thé à la menthe", "fr"),
        ("can i have a sandwich please", "en"),
    ]
    print("=== Test entity_extractor v2 ===\n")
    for text, lang in tests:
        entities = extract(text, lang)
        items_str = [menu.format_item(e["item"]) for e in entities["items"]]
        print(f"[{lang}] '{text}'")
        print(f"  qty={entities['quantity']}  size={entities['size']}  mods={entities['modifiers']}")
        print(f"  items   : {items_str}")
        if entities["unknown"]:
            print(f"  inconnus: {entities['unknown']}")
        print()
