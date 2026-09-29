"""
intent_classifier.py
====================
Classifie l'intention du client à partir du texte normalisé.

v2 — hybride :
  1. Patterns regex (rapide, déterministe) — INCHANGÉ, priorité absolue.
  2. Si regex ne trouve rien (intent="autre", score=0.0) → DistilBERT en secours,
     pour capter les formulations que les patterns n'ont pas prévues.

Le LLM n'intervient JAMAIS ici — comme avant, il ne sert qu'à la génération
de réponse dans dialog_manager.py / llm_engine.py.
"""

import re
from pathlib import Path

from config import INTENTS, BASE_DIR

# ── Patterns regex (INCHANGÉS — copie exacte de ton fichier original) ─────────
_PATTERNS = {
    "commander": {
        "fr": [
            r"\bje (veux|voudrais|voudrai|aimerais|prends|prend)\b",
            r"\bcommander?\b",
            r"\bapportez[-\s]moi\b",
            r"\bdonner?[-\s]moi\b",
            r"\bun(e)?\s+\w+\s+(s'il|sil)\b",
            r"\bje (vais|prendrai|prendrait) (prendre|commander)\b",
        ],
        "ar": [
            r"\bأريد\b", r"\bنحب\b", r"\bنبغي\b", r"\bبغيت\b",
            r"\bنطلب\b", r"\bاطلب\b", r"\bعطيني\b", r"\bجيب\b",
            r"\bهل يمكنني الحصول على\b", r"\bممكن ناخذ\b", r"\bممكن نحصل على\b",
        ],
        "en": [
            r"\bi('d| would) like\b",
            r"\bi('ll| will) have\b",
            r"\bi (want|would like|ll have|will have|need)\b",
            r"\bcan i (get|have|order)\b",
            r"\bmay i (have|get)\b",
            r"\bbrings? me\b",
            r"\bgive me\b",
        ],
    },
    "ajouter": {
        "fr": [
            r"\bajoute[rz]?\b", r"\brajoute[rz]?\b",
            r"\bégalement\b", r"\baussi\b.*\b(un|une)\b",
            r"\bet (aussi|en plus|avec ça)\b",
            r"\ben plus\b",
        ],
        "ar": [r"\bزيد\b", r"\bأضف\b", r"\bوأيضا\b", r"\bوكذلك\b"],
        "en": [
            r"\badd\b", r"\balso\b.*\ba\b", r"\band (also|as well)\b",
            r"\bin addition\b",
        ],
    },
    "supprimer": {
        "fr": [
            r"\benlève[rz]?\b", r"\bsupprime[rz]?\b", r"\bannule[rz]?\b",
            r"\bretire[rz]?\b",
        ],
        "ar": [r"\bاحذف\b", r"\bشيل\b"],
        "en": [
            r"\bremove\b", r"\btake off\b", r"\bcancel\b.*\bitem\b",
        ],
    },
    "modifier": {
        "fr": [
            r"\bchange[rz]?\b", r"\bmodifie[rz]?\b", r"\bremplace[rz]?\b",
            r"\bau lieu de\b", r"\bplutôt\b", r"\bà la place de\b",
        ],
        "ar": [r"\bغير\b", r"\bبدل\b", r"\bعوض\b"],
        "en": [
            r"\bchange\b", r"\bswap\b", r"\breplace\b",
            r"\binstead of\b", r"\brather\b",
        ],
    },
    "confirmer_commande": {
        "fr": [
            r"\bc(\'|')est tout\b", r"\bvoilà\b", r"\bça sera tout\b",
            r"\bok merci\b", r"\bconfirme[rz]?\b", r"\bc(\'|')est bon\b",
            r"\bpasse[rz]? la commande\b", r"\bmerci c(\'|')est tout\b",
        ],
        "ar": [
            r"\bهذا كل شيء\b", r"\bبس\b", r"\bكفى\b",
            r"\bخلاص\b", r"\bتمام\b", r"\bشكرا\b.*\bبس\b",
            r"\bاكتفينا\b",
        ],
        "en": [
            r"\bthat('?s| is) (all|it)\b", r"\bconfirm\b",
            r"\bthat('?ll| will) do\b", r"\bplace.*(the )?order\b",
            r"\byes please\b",
        ],
    },
    "annuler_commande": {
        "fr": [
            r"\bannule[rz]? (?:tout|toute|tous|toutes|la commande|ma commande)\b",
            r"\bannule[rz]? toute ma commande\b",
            r"\bremettre à zéro\b", r"\btout annuler\b",
        ],
        "ar": [r"\bألغِ\b", r"\bإلغاء\b", r"\bألغ كل شيء\b"],
        "en": [
            r"\bcancel (everything|all|the order|my order)\b",
            r"\bstart over\b", r"\bclear.*(my )?order\b",
        ],
    },
    "demander_total": {
        "fr": [
            r"\bcombien\b", r"\bc(\'|')est combien\b", r"\bquel est le (total|prix|montant)\b",
            r"\bprix total\b", r"\bça fait combien\b", r"\baddition\b",
        ],
        "ar": [r"\bكم\b", r"\bقداش\b", r"\bالحساب\b", r"\bالمجموع\b"],
        "en": [
            r"\bhow much\b", r"\bwhat(\'?s| is) (the )?total\b",
            r"\bthe bill\b", r"\bcheck please\b",
        ],
    },
    "demander_menu": {
        "fr": [
            r"\bqu(\'|')est[- ]ce que vous avez\b",
            r"\bvotre menu\b", r"\bla carte\b",
            r"\bavez[- ]vous\b", r"\bqu(\'|')avez[- ]vous\b",
            r"\bque proposez[- ]vous\b", r"\bvous avez quoi\b",
        ],
        "ar": [r"\bماذا عندكم\b", r"\bالقائمة\b", r"\bالمنيو\b", r"\bعندكم\b"],
        "en": [
            r"\bwhat do you (have|offer|serve)\b",
            r"\bthe menu\b", r"\byour menu\b",
            r"\bwhat(\'?s| is) available\b",
        ],
    },
    "salutation": {
        "fr": [r"\bbonjour\b", r"\bsalut\b", r"\bbonsoir\b", r"\bhello\b"],
        "ar": [r"\bمرحبا\b", r"\bالسلام عليكم\b", r"\bأهلا\b", r"\bعسلامة\b"],
        "en": [r"\bhello\b", r"\bhi\b", r"\bgood (morning|afternoon|evening)\b"],
    },
    "au_revoir": {
        "fr": [r"\bau revoir\b", r"\bmerci\b.*\bau revoir\b", r"\bbonne journée\b"],
        "ar": [r"\bمع السلامة\b", r"\bإلى اللقاء\b", r"\bشكرا\b.*\bمع السلامة\b"],
        "en": [r"\bgoodbye\b", r"\bbye\b", r"\bthank you\b.*\bbye\b"],
    },
}

# ── DistilBERT en secours (chargé une seule fois, lazy) ───────────────────────
_INTENT_MODEL_DIR = BASE_DIR / "intent_model" / "final"
_distilbert_classifier = None  # singleton, chargé au premier besoin
_DISTILBERT_MIN_CONFIDENCE = 0.55  # sous ce seuil, on garde "autre" plutôt que de risquer un faux positif


def _get_distilbert_classifier():
    """Charge le modèle DistilBERT une seule fois (coûteux à instancier)."""
    global _distilbert_classifier
    if _distilbert_classifier is None:
        if not _INTENT_MODEL_DIR.exists():
            return None  # modèle pas encore entraîné/téléchargé — dégrade gracieusement
        from intent_classifier_distilbert import IntentClassifier  # import tardif (torch est lourd)
        _distilbert_classifier = IntentClassifier(str(_INTENT_MODEL_DIR))
    return _distilbert_classifier


def warmup_intent_classifier() -> bool:
    """Précharge le fallback DistilBERT avant le premier tour de dialogue."""
    return _get_distilbert_classifier() is not None


def _classify_regex(text: str, lang: str) -> tuple[str, float]:
    """Classification par patterns — code original inchangé."""
    if not text:
        return "autre", 0.0

    text_lower = text.lower()
    scores = {}

    for intent, lang_patterns in _PATTERNS.items():
        patterns = lang_patterns.get(lang, []) + lang_patterns.get("fr", [])
        count = 0
        for pattern in patterns:
            if re.search(pattern, text_lower, flags=re.IGNORECASE | re.UNICODE):
                count += 1
        if count > 0:
            scores[intent] = count

    if not scores:
        return "autre", 0.0

    # En cas d'égalité, les actions explicites priment sur les formulations
    # génériques comme « je veux », sinon « je veux modifier... » devient une
    # nouvelle commande au lieu de muter le panier existant.
    priority = {
        "annuler_commande": 100,
        "confirmer_commande": 90,
        "modifier": 80,
        "supprimer": 70,
        "ajouter": 60,
        "demander_total": 50,
        "demander_menu": 40,
        "au_revoir": 30,
        "salutation": 20,
        "commander": 10,
    }
    best_intent = max(scores, key=lambda name: (scores[name], priority.get(name, 0)))
    best_score = min(scores[best_intent] / 3.0, 1.0)
    return best_intent, round(best_score, 2)


# ── Paiement ──────────────────────────────────────────────────────────────
# Le robot NEXOR n'est PAS un terminal de paiement (pas d'encaissement réel),
# mais il peut désormais ENREGISTRER le mode de paiement choisi par le client
# (cash / carte) pour le transmettre au staff — cf. dialog_manager._handle_payment.
# Le pourboire reste hors scope dans tous les cas : le robot ne doit jamais
# donner l'impression qu'il peut encaisser un montant additionnel lui-même.
_TIP_KEYWORDS_RE = re.compile(
    r"\b(tip|pourboire|gratuity|إكرامية)\b",
    flags=re.IGNORECASE | re.UNICODE,
)
_PAYMENT_KEYWORDS_RE = re.compile(
    r"\b(cash|espèces|especes|liquide|carte( bancaire)?|card|credit card|"
    r"pay(ing)?|payer|paiement|نقدا|بطاقة|أدفع|الدفع)\b",
    flags=re.IGNORECASE | re.UNICODE,
)


def classify(text: str, lang: str = "fr") -> tuple[str, float]:
    """
    Classifie l'intent du texte. Signature INCHANGÉE — dialog_manager.py et
    confidence_checker.py n'ont rien à modifier.

    Ordre de priorité:
        0. Filtre paiement/pourboire — toujours "paiement" (dialog_manager distingue ensuite).
        1. Regex (rapide, déterministe) — utilisé si un pattern matche.
        2. DistilBERT (secours) — utilisé UNIQUEMENT si le regex ne trouve rien,
           pour capter les formulations imprévues par les patterns.
    """
    if text and (_PAYMENT_KEYWORDS_RE.search(text) or _TIP_KEYWORDS_RE.search(text)):
        # Sans ce filtre, une phrase comme "add a tip" matche le pattern
        # générique \badd\b -> intent="ajouter" à tort (observé en test
        # réel : "Can you add one hundred as a tip please?" classé comme
        # ajout d'article). On route tout mot lié au paiement/pourboire
        # vers l'intent dédié AVANT la classification normale, quelle que
        # soit la langue. dialog_manager distingue ensuite pourboire
        # (toujours hors scope) et mode de paiement (accepté).
        return "paiement", 1.0

    intent, score = _classify_regex(text, lang)

    if intent != "autre" or not text:
        return intent, score

    # Regex n'a rien trouvé -> on tente DistilBERT en secours
    classifier = _get_distilbert_classifier()
    if classifier is None:
        return "autre", 0.0

    db_intent, db_confidence = classifier.predict(text)

    if db_confidence >= _DISTILBERT_MIN_CONFIDENCE:
        return db_intent, round(db_confidence, 2)

    return "autre", 0.0


if __name__ == "__main__":
    tests = [
        ("je voudrais un couscous s'il vous plaît", "fr"),
        ("ajoute aussi un café", "fr"),
        ("c'est tout merci", "fr"),
        ("combien ça fait ?", "fr"),
        ("i want a coffee please", "en"),
        ("نحب نطلب كسكسي", "ar"),
        ("bonjour", "fr"),
        # Formulation que le regex ne couvre probablement pas -> teste le fallback DistilBERT
        ("finalement mets-en moi un de plus", "fr"),
    ]
    print("=== Test intent_classifier (hybride regex + DistilBERT) ===")
    for text, lang in tests:
        intent, score = classify(text, lang)
        print(f"  [{lang}] '{text[:40]}' → {intent}  (score={score})")
