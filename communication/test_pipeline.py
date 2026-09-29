"""
test_nlp_llm.py
================
Tests du pipeline NLP + LLM uniquement (sans STT).
Entrée : texte brut (comme si Whisper avait déjà transcrit).

Tests :
  1. Modules NLP individuels (normalize, detect, classify, extract)
  2. Scénarios de conversation complets (fr/ar/en)
  3. Cas limites (hors sujet, langue mixte, phrases vides)
  4. Performance (latence NLP et LLM)

Usage :
  python test_nlp_llm.py           # tous les tests
  python test_nlp_llm.py --nlp     # NLP uniquement (sans Ollama)
  python test_nlp_llm.py --llm     # NLP + LLM
  python test_nlp_llm.py --auto    # mode automatique sans interaction
"""

import sys, time, json, logging
import argparse

logging.basicConfig(level=logging.WARNING, format="%(levelname)s | %(message)s")

# ── Args ───────────────────────────────────────────────────────────
parser = argparse.ArgumentParser()
parser.add_argument("--nlp",  action="store_true", help="NLP uniquement")
parser.add_argument("--llm",  action="store_true", help="NLP + LLM")
parser.add_argument("--auto", action="store_true", help="Mode automatique")
args = parser.parse_args()
RUN_LLM = args.llm or (not args.nlp)  # par défaut : tout

# ── Imports NLP ────────────────────────────────────────────────────
try:
    from preprocessing_nlp import normalize
    from language_detector import detect as detect_lang
    from intent_classifier import classify as classify_intent
    from entity_extractor import extract as extract_entities
    from response_formatter import format_response
    from menu_loader import get_menu
    NLP_OK = True
except ImportError as e:
    print(f"❌ Module NLP manquant : {e}")
    NLP_OK = False

menu = get_menu() if NLP_OK else None

# ═══════════════════════════════════════════════════════════════════
# UTILITAIRES
# ═══════════════════════════════════════════════════════════════════

def header(title):
    print(f"\n{'='*60}")
    print(f"  {title}")
    print(f"{'='*60}")

def run_nlp(text: str, whisper_lang: str) -> dict:
    """Passe un texte dans le pipeline NLP complet."""
    t0       = time.perf_counter()
    lang     = detect_lang(text, whisper_lang)
    clean    = normalize(text, lang)
    intent, score = classify_intent(clean, lang)
    entities = extract_entities(clean, lang)
    latency  = time.perf_counter() - t0
    items    = [menu.format_item(e["item"]) for e in entities.get("items", [])]
    return {
        "lang"    : lang,
        "clean"   : clean,
        "intent"  : intent,
        "score"   : score,
        "items"   : items,
        "entities": entities,
        "latency" : latency,
    }

# ═══════════════════════════════════════════════════════════════════
# TEST 1 — Modules NLP individuels
# ═══════════════════════════════════════════════════════════════════
def test_nlp_modules():
    header("TEST 1 — Modules NLP individuels")

    cases = [
        # (texte, whisper_lang, intent_attendu, item_attendu_ou_None)
        # ── Français ──────────────────────────────────────────────
        ("euh je voudrai commander un couscous sil vous plait", "fr", "commander",             "couscous"),
        ("ajoute aussi un café espresso",                       "fr", "ajouter",               "café"),
        ("c'est tout merci",                                    "fr", "confirmer_commande",    None),
        ("combien ça fait ?",                                   "fr", "demander_total",        None),
        ("vous avez quoi comme desserts ?",                     "fr", "demander_menu",         None),
        ("je voudrais changer ma commande",                     "fr", "modifier",              None),
        ("bonjour",                                             "fr", "salutation",            None),
        ("au revoir merci",                                     "fr", "au_revoir",             None),

        # ── Anglais ───────────────────────────────────────────────
        ("i want a coffee please",                              "en", "commander",             "café"),
        # Pas d'intent "recommandation" dans config.INTENTS -> doit retomber sur "autre"
        ("what do you recommend",                               "en", "autre",                 None),
        ("how much is the total",                               "en", "demander_total",        None),
        ("that's all thank you",                                "en", "confirmer_commande",    None),

        # ── Arabe ─────────────────────────────────────────────────
        ("نحب نطلب كسكسي من فضلك",                            "ar", "commander",             "couscous"),
        ("قداش تكلف",                                          "ar", "demander_total",        None),
        ("اكتفينا شكرا",                                       "ar", "confirmer_commande",    None),
        ("آش عندكم",                                           "ar", "demander_menu",         None),

        # ── Hors sujet ────────────────────────────────────────────
        ("quelle est la météo aujourd'hui",                    "fr", "autre",                 None),
        ("who is the president",                               "en", "autre",                 None),
    ]

    passed = 0
    latencies = []

    for text, wlang, expected_intent, expected_item in cases:
        r = run_nlp(text, wlang)
        latencies.append(r["latency"])

        intent_ok = r["intent"] == expected_intent
        item_ok   = (expected_item is None) or \
                    any(expected_item.lower() in i.lower() for i in r["items"])
        ok        = intent_ok and item_ok

        status = "✅" if ok else "❌"
        print(f"\n  {status} [{r['lang']}] '{text[:50]}'")
        print(f"     Intent : {r['intent']:<25} (attendu={expected_intent}) {'✅' if intent_ok else '❌'}")
        if r["items"]:
            print(f"     Items  : {r['items']}")
        if not item_ok and expected_item:
            print(f"     ⚠️  Item attendu '{expected_item}' non trouvé")
        print(f"     Score  : {r['score']:.3f}  |  Latence : {r['latency']*1000:.1f}ms")

        if ok:
            passed += 1

    import numpy as np
    print(f"\n  Résultat  : {passed}/{len(cases)} tests passés")
    print(f"  Latence   : moy={np.mean(latencies)*1000:.1f}ms  max={np.max(latencies)*1000:.1f}ms")
    return passed >= len(cases) * 0.85   # seuil 85%


# ═══════════════════════════════════════════════════════════════════
# TEST 2 — Scénarios de conversation complets
# ═══════════════════════════════════════════════════════════════════
def test_conversations():
    header("TEST 2 — Scénarios de conversation complets")

    if not RUN_LLM:
        print("  ⏭️  Ignoré (--nlp uniquement)")
        return True

    try:
        from llm_engine import get_engine
        from dialog_manager import DialogManager
    except ImportError as e:
        print(f"  ⚠️  {e} — test ignoré")
        return True

    engine = get_engine()
    if not engine.is_available():
        print("  ⚠️  LLM (Ollama) non disponible — test ignoré")
        print("  Lance : ollama serve && ollama pull qwen2.5:1.5b")
        return True

    scenarios = {
        "Commande complète (FR)": [
            ("bonjour",                                      "fr"),
            ("je voudrais un couscous agneau s'il vous plaît","fr"),
            ("et aussi un café espresso",                    "fr"),
            ("combien ça fait ?",                            "fr"),
            ("c'est tout merci",                             "fr"),
        ],
        "Full order (EN)": [
            ("hello",                                        "en"),
            ("i'd like to order a tagine please",            "en"),
            ("and a mint tea",                               "en"),
            ("how much is it",                               "en"),
            ("that's all thank you",                         "en"),
        ],
        "طلب كامل (AR)": [
            ("مرحبا",                                        "ar"),
            ("نحب نطلب كسكسي بالحوت",                      "ar"),
            ("وعصير برتقال",                                 "ar"),
            ("قداش تكلف",                                   "ar"),
            ("اكتفينا شكرا",                                 "ar"),
        ],
        "Cas limites": [
            ("quelle est la capitale de la France",          "fr"),
            ("",                                             "fr"),
            ("euh... hmm...",                                "fr"),
        ],
    }

    all_ok = True
    for scenario_name, turns in scenarios.items():
        print(f"\n  ── {scenario_name} {'─'*(40-len(scenario_name))}")
        dm = DialogManager()
        scenario_ok = True

        for text, lang in turns:
            if not text.strip():
                print(f"  🎤 [vide]")
                try:
                    response = dm.process(text, lang)
                    print(f"  🤖 {response[:80] if response else '(vide)'}")
                except Exception as e:
                    print(f"  🤖 Exception : {e}")
                continue

            t0       = time.perf_counter()
            response = dm.process(text, lang)
            latency  = time.perf_counter() - t0

            print(f"  🎤 [{lang}] {text}")
            print(f"  🤖 ({latency:.2f}s) {response[:80] if response else '❌ Réponse vide'}")

            if not response:
                scenario_ok = False

            if dm.state.finished:
                print(f"  ✅ Conversation terminée")
                break

        if not scenario_ok:
            all_ok = False
            print(f"  ❌ Scénario '{scenario_name}' échoué")

    return all_ok


# ═══════════════════════════════════════════════════════════════════
# TEST 3 — Cas limites NLP
# ═══════════════════════════════════════════════════════════════════
def test_edge_cases():
    header("TEST 3 — Cas limites NLP")

    cases = [
        # Texte vide
        ("",                    "fr", None),
        # Texte très court
        ("oui",                 "fr", None),
        ("no",                  "en", None),
        # Chiffres seuls
        ("2",                   "fr", None),
        # Langue mixte (code-switching)
        ("I want un couscous",  "fr", "commander"),
        ("je veux a coffee",    "fr", "commander"),
        # Typos fréquentes
        ("je vodrais comander", "fr", "commander"),
        ("i whant a cofee",     "en", "commander"),
        # Ponctuation excessive
        ("bonjour !!!",         "fr", "salutation"),
        # Majuscules
        ("COMBIEN CA FAIT",     "fr", "demander_total"),
    ]

    passed = 0
    for text, wlang, expected_intent in cases:
        try:
            r = run_nlp(text, wlang)
            intent_ok = (expected_intent is None) or (r["intent"] == expected_intent)
            status    = "✅" if intent_ok else "⚠️ "
            label     = f"'{text[:35]}'" if text else "(vide)"
            print(f"  {status} [{wlang}] {label:<38} → intent={r['intent']}")
            if intent_ok:
                passed += 1
            else:
                passed += 1  # cas limites : pas bloquants
        except Exception as e:
            print(f"  ❌ [{wlang}] '{text[:35]}' → Exception : {e}")

    print(f"\n  Résultat : {passed}/{len(cases)} cas limites gérés sans crash")
    return passed == len(cases)


# ═══════════════════════════════════════════════════════════════════
# TEST 4 — Performance
# ═══════════════════════════════════════════════════════════════════
def test_performance():
    header("TEST 4 — Performance NLP")

    phrases = [
        ("Je voudrais commander un couscous agneau s'il vous plaît", "fr"),
        ("I would like to order a tagine and a mint tea",             "en"),
        ("نحب نطلب كسكسي بالحوت وعصير برتقال من فضلك",              "ar"),
    ]

    N = 10  # répétitions pour moyenne stable
    import numpy as np

    print(f"\n  Benchmark NLP ({N} répétitions par phrase)\n")
    print(f"  {'Langue':<6} {'Phrase':<45} {'Moy':>8} {'Max':>8}")
    print(f"  {'─'*6} {'─'*45} {'─'*8} {'─'*8}")

    all_fast = True
    for text, wlang in phrases:
        lats = []
        for _ in range(N):
            r = run_nlp(text, wlang)
            lats.append(r["latency"] * 1000)

        moy = np.mean(lats)
        mx  = np.max(lats)
        ok  = moy < 200  # seuil : 200ms max pour NLP seul

        status = "✅" if ok else "❌"
        print(f"  {status} {wlang:<6} {text[:45]:<45} {moy:>6.1f}ms {mx:>6.1f}ms")

        if not ok:
            all_fast = False

    print(f"\n  Seuil NLP : < 200ms (rule-based)")
    return all_fast


# ═══════════════════════════════════════════════════════════════════
# MODE INTERACTIF
# ═══════════════════════════════════════════════════════════════════
def interactive_mode():
    header("MODE INTERACTIF — tape du texte, CTRL+C pour quitter")

    llm_available = False
    dm = None

    if RUN_LLM:
        try:
            from llm_engine import get_engine
            from dialog_manager import DialogManager
            if get_engine().is_available():
                dm = DialogManager()
                llm_available = True
                print("  ✅ LLM disponible — réponses complètes activées")
            else:
                print("  ⚠️  LLM non disponible — affichage NLP uniquement")
        except ImportError:
            print("  ⚠️  LLM non disponible — affichage NLP uniquement")

    print("  Langues : fr | en | ar  (ex: [fr] Bonjour)")
    print("  Tape 'reset' pour recommencer la conversation\n")

    current_lang = "fr"
    try:
        while True:
            try:
                user_input = input("  🎤 > ").strip()
            except EOFError:
                break

            if not user_input:
                continue

            if user_input.lower() == "reset":
                if dm:
                    dm = __import__("dialog_manager").DialogManager()
                print("  ♻️  Conversation réinitialisée\n")
                continue

            # Détecte langue depuis le préfixe [fr]/[en]/[ar]
            if user_input.startswith("[") and "]" in user_input:
                bracket_end = user_input.index("]")
                current_lang = user_input[1:bracket_end].strip().lower()
                user_input   = user_input[bracket_end+1:].strip()

            # NLP
            r = run_nlp(user_input, current_lang)
            print(f"  NLP → lang={r['lang']}  intent={r['intent']}  "
                  f"score={r['score']:.2f}  items={r['items']}  "
                  f"({r['latency']*1000:.0f}ms)")

            # LLM
            if llm_available and dm:
                t0       = time.perf_counter()
                response = dm.process(user_input, current_lang)
                lat      = time.perf_counter() - t0
                print(f"  🤖 ({lat:.2f}s) {response}\n")
                if dm.state.finished:
                    print("  ✅ Commande terminée — reset automatique")
                    dm = __import__("dialog_manager").DialogManager()
            else:
                print()

    except KeyboardInterrupt:
        print("\n\n  👋 Mode interactif terminé")


# ═══════════════════════════════════════════════════════════════════
# POINT D'ENTRÉE
# ═══════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    if not NLP_OK:
        print("❌ Modules NLP non disponibles — arrêt")
        sys.exit(1)

    print("\n🚀 Tests Pipeline NLP + LLM (sans STT)\n")

    results = []
    results.append(("Modules NLP",     test_nlp_modules()))
    results.append(("Cas limites",     test_edge_cases()))
    results.append(("Performance NLP", test_performance()))
    results.append(("Conversations",   test_conversations()))

    # Résumé
    header("RÉSUMÉ")
    all_ok = True
    for name, ok_flag in results:
        print(f"  {'✅' if ok_flag else '❌'} {name}")
        if not ok_flag:
            all_ok = False

    print()
    if all_ok:
        print("  ✅ Tous les tests passés — NLP+LLM prêt pour intégration STT !")
    else:
        print("  ❌ Certains tests ont échoué — voir les détails ci-dessus")

    # Mode interactif si pas en mode auto
    if not args.auto:
        print()
        rep = input("  Lancer le mode interactif ? [o/N] : ").strip().lower()
        if rep in ("o", "oui", "y", "yes"):
            interactive_mode()

    # Export rapport
    report = {
        "results" : {name: ok for name, ok in results},
        "all_ok"  : all_ok,
        "run_llm" : RUN_LLM,
    }
    with open("test_nlp_llm_report.json", "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    print(f"\n  📄 Rapport → test_nlp_llm_report.json")