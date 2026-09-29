"""
dialog_manager.py
=================
Gère l'état du dialogue et orchestre tout le pipeline NLP.

Pipeline complet pour chaque tour :
  texte STT
    → garde-fou texte vide (fallback_handler)
    → preprocessing (normalisation)
    → language_detector (langue)
    → intent_classifier (intention)
    → entity_extractor (items, quantités)
    → confidence_checker (filtre de confiance -> fallback_handler si ambigu)
    → order_validator (validation avant ajout/màj commande)
    → prompt_builder (construction prompt)
    → llm_engine (génération réponse)
    → response_formatter (nettoyage pour TTS)
    → session_logger (journalisation du tour)
    → réponse vocale
"""

import logging
import re
import time
import uuid
from dataclasses import dataclass, field

from menu_loader import get_menu, MenuLoader
from preprocessing_nlp import normalize
from language_detector import detect as detect_lang
from intent_classifier import classify as classify_intent
from entity_extractor import extract as extract_entities
from confidence_checker import check as check_confidence
from order_validator import validate_item, validate_order_size
from fallback_handler import FallbackHandler
from staff_notifier import (
    notify_order as log_order_audit,
    notify_payment as log_payment_audit,
)
from staff_app_client import (
    notify_order,
    notify_payment as push_payment_live,
    notify_dialog_state as push_dialog_state,
    notify_draft_order as push_draft_order,
)
from session_logger import SessionLogger
from prompt_builder import build_messages
from llm_engine import get_engine
from response_formatter import format_response
from config import DEFAULT_LANG, LOG_SENSITIVE_DATA, MAX_ITEMS_ORDER, MAX_TURNS

logger = logging.getLogger(__name__)
menu   = get_menu()

# Score minimum attribué quand on corrige "autre" -> "commander" parce que
# des entités ont été détectées : l'extraction d'entités est un signal fort,
# on ne veut pas que confidence_checker rejette ce tour à cause du score
# regex initial (souvent 0.0 puisque aucun pattern d'intent n'a matché).
_ENTITY_OVERRIDE_SCORE = 0.6

# ── Détection du mode de paiement ──────────────────────────────────────────
# Le robot n'encaisse jamais réellement : il enregistre le choix du client
# et informe le personnel. Regex séparées de intent_classifier.py car ce
# n'est plus une question de classification d'intent (déjà tranché en
# "paiement") mais d'extraction du détail (cash / carte / pourboire seul).
_CASH_RE = re.compile(r"\b(cash|espèces|especes|liquide|نقدا)\b", re.IGNORECASE | re.UNICODE)
_CARD_RE = re.compile(r"\b(carte( bancaire)?|card|credit card|بطاقة)\b", re.IGNORECASE | re.UNICODE)
_TIP_RE  = re.compile(r"\b(tip|pourboire|gratuity|إكرامية)\b", re.IGNORECASE | re.UNICODE)
_NEGATED_ITEM_RE = re.compile(
    r"\b(?:pas de|ne veux pas|sans vouloir|do not want|don't want|no quiero)\b",
    re.IGNORECASE | re.UNICODE,
)


@dataclass
class OrderItem:
    item_id:   str
    nom:       str
    prix:      float
    quantity:  int    = 1
    modifiers: list   = field(default_factory=list)
    size:      str    = None


@dataclass
class DialogState:
    lang:        str        = DEFAULT_LANG
    turn:        int        = 0
    order:       list       = field(default_factory=list)  # [OrderItem]
    history:     list       = field(default_factory=list)  # messages LLM
    confirmed:   bool       = False
    finished:    bool       = False
    payment_method: str     = None   # "cash" | "card" | None
    table_id:    str        = None   # ID fixe de la table (carte de navigation robot)
    order_id:    str        = None   # ID stable pour le dashboard staff (assigné au 1er item ajouté)


class DialogManager:
    def __init__(self, table_id: str = None):
        self.engine   = get_engine()
        self.state    = DialogState(table_id=table_id)
        # Instances PAR SESSION (pas de singleton global) : le compteur de
        # tours flous et le journal de session ne doivent pas être partagés
        # entre deux clients différents.
        self.fallback = FallbackHandler()
        self.session  = SessionLogger()

    def reset(self, table_id: str = None):
        """Réinitialise le dialogue (nouveau client). Conserve table_id
        si non précisé, pour permettre un reset sans perdre la table."""
        self.state    = DialogState(table_id=table_id if table_id is not None else self.state.table_id)
        self.fallback = FallbackHandler()
        self.session  = SessionLogger()
        logger.info(f"🔄 Dialogue réinitialisé (table={self.state.table_id})")

    def process(self, stt_text: str, whisper_lang: str = None) -> str:
        """
        Traite un tour de dialogue complet, ET pousse l'état vers l'écran
        client (orbe + panier live) via /api/events/dialog et
        /api/events/draft_order. La logique métier elle-même est dans
        _process_inner() — inchangée, pas touchée par ce wrapper.

        ⚠️ Limite honnête : dialog_manager.py ne sait PAS quand le micro
        est réellement en train d'enregistrer ("listening") ni quand le
        TTS a fini de parler (retour à "idle") — ces deux états doivent
        être poussés par le script qui pilote le micro/haut-parleur
        (hors de ce fichier), pas ici. Ce wrapper pousse uniquement
        "processing" (dès que le texte transcrit arrive) puis "speaking"
        (dès que la réponse est prête).
        """
        push_dialog_state("processing", text=stt_text or "", lang=self.state.lang)

        response = self._process_inner(stt_text, whisper_lang)

        push_dialog_state("speaking", text=response, lang=self.state.lang)
        self._push_draft_order()

        return response

    def _push_draft_order(self) -> None:
        """Pousse le panier EN COURS (avant confirmation) vers l'écran
        client — purement pour affichage live, jamais persisté côté
        backend (voir DraftOrderEvent dans main.py)."""
        items = [self._format_item_label(o) for o in self.state.order]
        total = sum(o.prix * o.quantity for o in self.state.order) if self.state.order else None
        push_draft_order(items, total)


    def _process_inner(self, stt_text: str, whisper_lang: str = None) -> str:
        """
        Traite un tour de dialogue complet.

        Args:
            stt_text     : texte brut issu de Whisper
            whisper_lang : langue détectée par Whisper (optionnel)

        Returns:
            Réponse textuelle à synthétiser (TTS-ready)
        """
        t0 = time.perf_counter()
        self.state.turn += 1
        logged_text = stt_text if LOG_SENSITIVE_DATA else "[redacted]"
        logger.info("🎤 Tour %s | STT: '%s'", self.state.turn, logged_text)

        # ── Sécurité : max tours ──────────────────────────────────────────────
        if self.state.turn > MAX_TURNS:
            self.state.finished = True
            response = self._msg("max_turns")
            self._log_turn(stt_text, self.state.lang, "max_turns", 0.0, {}, response, t0)
            return response

        # ── 0. Garde-fou texte vide (STT silence/inaudible) ───────────────────
        if not stt_text or not stt_text.strip():
            response = self.fallback.handle("empty", self.state.lang)
            self._log_turn(stt_text, self.state.lang, "empty", 0.0, {}, response, t0)
            return response

        # ── 1. Preprocessing ──────────────────────────────────────────────────
        lang       = detect_lang(stt_text, whisper_lang)
        self.state.lang = lang
        clean_text = normalize(stt_text, lang)
        logger.info("   Langue: %s | Texte normalisé: '%s'", lang,
                    clean_text if LOG_SENSITIVE_DATA else "[redacted]")

        # ── 2. Intent ─────────────────────────────────────────────────────────
        intent, score = classify_intent(clean_text, lang)
        logger.info(f"   Intent: {intent} (score={score})")

        # ── 3. Entités ────────────────────────────────────────────────────────
        entities = extract_entities(clean_text, lang)
        logger.info("   Entités: %s", (
            [menu.format_item(e["item"]) for e in entities["items"]]
            if LOG_SENSITIVE_DATA else f"{len(entities['items'])} item(s)"
        ))

        # ── 4. Correction intent si entités détectées mais regex muet ─────────
        if intent == "autre" and entities["items"] and not _NEGATED_ITEM_RE.search(clean_text):
            intent = "commander"
            score  = max(score, _ENTITY_OVERRIDE_SCORE)
            logger.info("   Intent corrigé : autre → commander (items détectés)")

        # ── 5. Filtre de confiance ──────────────────────────────────────────────
        # confidence_checker ne fait ici QUE décider (ok / pas ok) ; c'est
        # fallback_handler qui produit le texte, pour ne pas avoir deux
        # sources de messages de clarification qui divergent.
        confident, _ = check_confidence(intent, score, lang)
        if not confident:
            response = self.fallback.handle("low_score", lang)
            self._log_turn(clean_text, lang, intent, score, entities, response, t0)
            return response

        # Tour compris -> on réinitialise le compteur de tours flous
        self.fallback.reset()

        # ── 6. Intents gérés sans LLM ────────────────────────────────────────
        direct_response = self._handle_intent(intent, entities, lang)
        if direct_response:
            self._add_to_history(stt_text, direct_response)
            final = format_response(direct_response, lang)
            self._log_turn(clean_text, lang, intent, score, entities, final, t0)
            return final

        # ── 7. LLM ────────────────────────────────────────────────────────────
        if not self.engine.is_available():
            response = self.fallback.handle("llm_error", lang)
            self._log_turn(clean_text, lang, intent, score, entities, response, t0)
            return response

        order_summary = self._format_order(lang)
        messages      = build_messages(
            text=clean_text,
            lang=lang,
            intent=intent,
            entities=entities,
            order_summary=order_summary,
            history=self.state.history,
        )
        raw_response  = self.engine.generate(messages, lang=lang)
        logger.info("   LLM réponse: '%s'", raw_response[:80] if LOG_SENSITIVE_DATA else "[redacted]")

        # ── 8. Nettoyage pour TTS ─────────────────────────────────────────────
        tts_response = format_response(raw_response, lang)

        # ── 9. Mise à jour historique + log ──────────────────────────────────
        self._add_to_history(clean_text, tts_response)
        self._log_turn(clean_text, lang, intent, score, entities, tts_response, t0)

        return tts_response

    def _handle_intent(self, intent: str, entities: dict, lang: str) -> str | None:
        """
        Gère les intents qui ne nécessitent pas le LLM.
        Retourne une réponse directe ou None (→ LLM prend en charge).
        """
        if self.state.confirmed and intent in {
            "commander", "ajouter", "supprimer", "modifier", "annuler_commande"
        }:
            msgs = {
                "fr": "Cette commande est déjà confirmée et ne peut plus être modifiée. Veuillez appeler le personnel.",
                "ar": "تم تأكيد هذا الطلب ولا يمكن تعديله. يرجى طلب المساعدة من الطاقم.",
                "en": "This order is already confirmed and can no longer be changed. Please ask the staff.",
            }
            return msgs.get(lang, msgs["fr"])

        if intent == "commander" and entities["items"]:
            return self._add_items(entities, lang)

        if intent == "ajouter" and entities["items"]:
            return self._add_items(entities, lang)

        # Le client veut commander/ajouter mais AUCUN item du menu n'a été
        # reconnu dans sa phrase (ex: plat qui n'existe pas au menu).
        # On ne délègue PAS ce cas au LLM : sans confirmation ferme que
        # l'item est absent du menu, un LLM peut "halluciner" une
        # correspondance avec un item existant (observé en test réel :
        # "kofta" confondu avec "Salade niçoise" à 8.500 TND, confirmé à
        # tort). On répond donc nous-mêmes, sans jamais inventer de prix
        # ni de disponibilité.
        if intent in ("commander", "ajouter") and not entities["items"]:
            # Suggestion floue disponible (mot proche d'un plat du menu) ?
            # On la PROPOSE explicitement au client sans jamais l'ajouter
            # nous-mêmes à la commande — c'est toujours lui qui confirme.
            if entities.get("suggestions"):
                nom_suggere = next(iter(entities["suggestions"].values()))["nom"].get(
                    lang, next(iter(entities["suggestions"].values()))["nom"]["fr"]
                )
                msgs = {
                    "fr": f"Désolé, je n'ai pas trouvé cet article. "
                          f"Vouliez-vous dire « {nom_suggere} » ?",
                    "ar": f"عذراً، لم أجد هذا الصنف. هل تقصد « {nom_suggere} »؟",
                    "en": f"Sorry, I couldn't find that item. Did you mean \"{nom_suggere}\"?",
                }
                return msgs.get(lang, msgs["fr"])

            msgs = {
                "fr": "Désolé, je n'ai pas trouvé cet article dans notre menu. "
                      "Pouvez-vous préciser ou choisir un autre plat ?",
                "ar": "عذراً، لم أجد هذا الصنف في قائمتنا. "
                      "هل يمكنك التوضيح أو اختيار صنف آخر؟",
                "en": "Sorry, I couldn't find that item on our menu. "
                      "Could you clarify or choose something else?",
            }
            return msgs.get(lang, msgs["fr"])

        if intent == "supprimer" and entities["items"]:
            return self._remove_items(entities, lang)

        if intent == "supprimer" and not entities["items"]:
            msgs = {
                "fr": "Quel article souhaitez-vous retirer de la commande ?",
                "ar": "ما الصنف الذي تريد إزالته من الطلب؟",
                "en": "Which item would you like to remove from the order?",
            }
            return msgs.get(lang, msgs["fr"])

        if intent == "modifier":
            return self._modify_items(entities, lang)

        if intent == "confirmer_commande":
            return self._confirm_order(lang)

        if intent == "annuler_commande":
            return self._cancel_order(lang)

        if intent == "demander_total":
            return self._get_total(lang)

        if intent == "demander_menu":
            categories = [
                category["nom"].get(lang, category["nom"].get("fr", category_id))
                for category_id, category in menu.categories.items()
                if any(item.get("disponible", True)
                       for item in menu.get_by_category(category_id))
            ]
            labels = ", ".join(categories)
            msgs = {
                "fr": f"Nous proposons : {labels}. Quelle catégorie souhaitez-vous découvrir ?",
                "ar": f"نقترح: {labels}. أي قسم تريد أن تكتشف؟",
                "en": f"We offer: {labels}. Which category would you like to explore?",
            }
            return msgs.get(lang, msgs["fr"])

        if intent == "salutation":
            msgs = {
                "fr": f"Bonjour ! Bienvenue au {menu.restaurant['nom']}. Que souhaitez-vous commander ?",
                "ar": f"مرحبا ! أهلاً بك في {menu.restaurant['nom']}. ماذا تريد أن تطلب؟",
                "en": f"Hello! Welcome to {menu.restaurant['nom']}. What would you like to order?",
            }
            return msgs.get(lang, msgs["fr"])

        if intent == "paiement":
            return self._handle_payment(entities.get("raw_text", ""), lang)

        if intent == "au_revoir":
            if self.state.order and not self.state.confirmed:
                msgs = {
                    "fr": "Votre commande n'est pas encore confirmée. Souhaitez-vous la confirmer ou l'annuler ?",
                    "ar": "لم يتم تأكيد طلبك بعد. هل تريد تأكيده أم إلغاءه؟",
                    "en": "Your order has not been confirmed yet. Would you like to confirm or cancel it?",
                }
                return msgs.get(lang, msgs["fr"])
            self.state.finished = True
            msgs = {
                "fr": "Merci pour votre visite. Bonne journée !",
                "ar": "شكراً لزيارتكم. نهارك سعيد!",
                "en": "Thank you for your visit. Have a great day!",
            }
            return msgs.get(lang, msgs["fr"])

        # Pour les autres intents → LLM
        return None

    def _add_items(self, entities: dict, lang: str) -> str:
        """Valide puis ajoute les items à la commande, retourne une confirmation."""
        requested_units = sum(e.get("quantity", 1) for e in entities["items"])
        current_units = sum(o.quantity for o in self.state.order)
        ok, msg = validate_order_size(current_units, requested_units, lang)
        if not ok:
            return msg

        added  = []
        errors = []
        for e in entities["items"]:
            item = e["item"]

            valid, err = validate_item(item["id"], e["quantity"], lang)
            if not valid:
                errors.append(err)
                continue

            order_item = OrderItem(
                item_id   = item["id"],
                nom       = item["nom"].get(lang, item["nom"]["fr"]),
                prix      = item["prix"],
                quantity  = e["quantity"],
                modifiers = e["modifiers"],
                size      = e["size"],
            )
            existing = next((o for o in self.state.order if (
                o.item_id == order_item.item_id
                and o.size == order_item.size
                and o.modifiers == order_item.modifiers
            )), None)
            if existing:
                valid, err = validate_item(item["id"], existing.quantity + e["quantity"], lang)
                if not valid:
                    errors.append(err)
                    continue
                existing.quantity += e["quantity"]
            else:
                self.state.order.append(order_item)
            added.append(f"{e['quantity']}× {order_item.nom} ({item['prix']:.3f} {menu.restaurant['devise']})")

        # Items demandés mais absents du menu (détectés par entity_extractor)
        if entities.get("unknown"):
            not_found_msgs = {
                "fr": "n'est pas au menu",
                "ar": "غير موجود في القائمة",
                "en": "is not on the menu",
            }
            suffix = not_found_msgs.get(lang, not_found_msgs["fr"])
            errors.extend(f"{name} {suffix}" for name in entities["unknown"])

        if not added:
            if errors:
                return errors[0]
            msgs = {
                "fr": "Je n'ai pas trouvé cet item dans notre menu. Pouvez-vous préciser ?",
                "ar": "لم أجد هذا الصنف في قائمتنا. هل يمكنك التوضيح؟",
                "en": "I couldn't find that item in our menu. Could you be more specific?",
            }
            return msgs.get(lang, msgs["fr"])

        items_str = " et ".join(added) if lang == "fr" else ", ".join(added)
        msgs = {
            "fr": f"J'ai ajouté {items_str} à votre commande. Autre chose ?",
            "ar": f"أضفت {items_str} إلى طلبك. هل تريد شيئاً آخر؟",
            "en": f"I've added {items_str} to your order. Anything else?",
        }
        response = msgs.get(lang, msgs["fr"])

        if errors:
            response += " (" + "; ".join(errors) + ")"

        # Pas de push dashboard ici : main.py n'accepte qu'un événement
        # "confirmed" côté commande (voir _confirm_order). Le staff n'a pas
        # besoin de voir chaque "et un café aussi" tour par tour — seulement
        # la commande complète, une fois confirmée par le client.
        return response

    def _push_order_update(self):
        """Pousse la commande confirmée vers le dashboard staff
        (best-effort, non-bloquant — voir staff_app_client.py).
        Appelé UNE SEULE FOIS, depuis _confirm_order : main.py ne connaît
        que l'événement "confirmed", pas de suivi "new"/"updated" côté commande."""
        if self.state.order_id is None:
            self.state.order_id = str(uuid.uuid4())[:8]
        total = sum(o.prix * o.quantity for o in self.state.order)
        item_lines = [self._format_item_label(o) for o in self.state.order]
        audit = log_order_audit(
            order_id=self.state.order_id,
            table=self.state.table_id,
            order_items=item_lines,
            total=total,
            devise=menu.restaurant["devise"],
            lang=self.state.lang,
        )
        if not audit.get("queued"):
            logger.error("Commande %s non persistée dans la file locale", self.state.order_id)
        return notify_order(
            order_id=self.state.order_id,
            table=self.state.table_id,
            items=item_lines,
            total=total,
            devise=menu.restaurant["devise"],
            lang=self.state.lang,
        )

    def _remove_items(self, entities: dict, lang: str) -> str:
        """Retire la quantité demandée; sans quantité explicite, retire une unité."""
        removed = []
        for e in entities["items"]:
            item_id = e["item"]["id"]
            nom     = e["item"]["nom"].get(lang, e["item"]["nom"]["fr"])
            remaining = max(1, int(e.get("quantity", 1)))
            for order_item in list(self.state.order):
                if order_item.item_id != item_id or remaining <= 0:
                    continue
                amount = min(order_item.quantity, remaining)
                order_item.quantity -= amount
                remaining -= amount
                removed.append(f"{amount}× {nom}")
                if order_item.quantity == 0:
                    self.state.order.remove(order_item)

        if not removed:
            msgs = {
                "fr": "Cet item n'est pas dans votre commande.",
                "ar": "هذا الصنف غير موجود في طلبك.",
                "en": "That item is not in your order.",
            }
            return msgs.get(lang, msgs["fr"])

        items_str = ", ".join(removed)
        msgs = {
            "fr": f"J'ai retiré {items_str} de votre commande.",
            "ar": f"أزلت {items_str} من طلبك.",
            "en": f"I've removed {items_str} from your order.",
        }
        return msgs.get(lang, msgs["fr"])

    def _modify_items(self, entities: dict, lang: str) -> str:
        """Applique réellement une taille/modification ou remplace un article."""
        requested = entities.get("items", [])
        if not requested or not self.state.order:
            msgs = {
                "fr": "Précisez l'article à modifier dans votre commande.",
                "ar": "يرجى تحديد الصنف الذي تريد تعديله في طلبك.",
                "en": "Please specify which item in your order you want to change.",
            }
            return msgs.get(lang, msgs["fr"])

        target_entity = requested[0]
        target = next((o for o in self.state.order if o.item_id == target_entity["item"]["id"]), None)
        if target is None:
            msgs = {
                "fr": "Cet article n'est pas dans votre commande.",
                "ar": "هذا الصنف غير موجود في طلبك.",
                "en": "That item is not in your order.",
            }
            return msgs.get(lang, msgs["fr"])

        if len(requested) > 1:
            replacement = requested[1]
            replacement_item = replacement["item"]
            replacement_size = replacement.get("size")
            replacement_modifiers = list(replacement.get("modifiers") or [])
            duplicate = next((o for o in self.state.order if (
                o is not target
                and o.item_id == replacement_item["id"]
                and o.size == replacement_size
                and o.modifiers == replacement_modifiers
            )), None)
            resulting_quantity = target.quantity + (duplicate.quantity if duplicate else 0)
            valid, error = validate_item(replacement_item["id"], resulting_quantity, lang)
            if not valid:
                return error
            if duplicate:
                duplicate.quantity = resulting_quantity
                self.state.order.remove(target)
                target = duplicate
            else:
                target.item_id = replacement_item["id"]
                target.nom = replacement_item["nom"].get(lang, replacement_item["nom"]["fr"])
                target.prix = replacement_item["prix"]
                target.size = replacement_size
                target.modifiers = replacement_modifiers
        else:
            if target_entity.get("size"):
                target.size = target_entity["size"]
            if target_entity.get("modifiers"):
                target.modifiers = list(target_entity["modifiers"])
            if not target_entity.get("size") and not target_entity.get("modifiers"):
                msgs = {
                    "fr": "Quelle modification souhaitez-vous appliquer ?",
                    "ar": "ما التعديل الذي تريد تطبيقه؟",
                    "en": "What change would you like to apply?",
                }
                return msgs.get(lang, msgs["fr"])

        label = self._format_item_label(target)
        msgs = {
            "fr": f"Modification appliquée : {label}.",
            "ar": f"تم تطبيق التعديل: {label}.",
            "en": f"Change applied: {label}.",
        }
        return msgs.get(lang, msgs["fr"])

    @staticmethod
    def _format_item_label(order_item: OrderItem) -> str:
        details = []
        if order_item.size:
            details.append(order_item.size)
        details.extend(order_item.modifiers or [])
        suffix = f" ({', '.join(details)})" if details else ""
        return f"{order_item.quantity}× {order_item.nom}{suffix}"

    def _confirm_order(self, lang: str) -> str:
        """Confirme la commande."""
        if self.state.confirmed:
            msgs = {
                "fr": "Votre commande est déjà confirmée.",
                "ar": "تم تأكيد طلبك بالفعل.",
                "en": "Your order is already confirmed.",
            }
            return msgs.get(lang, msgs["fr"])

        if not self.state.order:
            msgs = {
                "fr": "Votre commande est vide. Que souhaitez-vous commander ?",
                "ar": "طلبك فارغ. ماذا تريد أن تطلب؟",
                "en": "Your order is empty. What would you like to order?",
            }
            return msgs.get(lang, msgs["fr"])

        if not self.state.table_id or not str(self.state.table_id).strip():
            msgs = {
                "fr": "Je ne peux pas confirmer sans identifiant de table. Veuillez appeler le personnel.",
                "ar": "لا يمكنني تأكيد الطلب دون رقم الطاولة. يرجى طلب المساعدة من الطاقم.",
                "en": "I cannot confirm the order without a table identifier. Please ask the staff.",
            }
            return msgs.get(lang, msgs["fr"])

        total      = sum(o.prix * o.quantity for o in self.state.order)
        devise     = menu.restaurant["devise"]
        self.state.confirmed = True
        self.state.finished  = False
        # La confirmation ouvre une nouvelle phase (service/paiement) : le
        # plafond de tours de la prise de commande ne doit pas bloquer le
        # paiement juste après une conversation longue.
        self.state.turn = 0

        self._push_order_update()

        # _format_order() inclut déjà une ligne "Total : ..." -> ne pas la répéter ici
        summary    = self._format_order(lang)
        msgs = {
            "fr": f"Commande confirmée ! {summary}. Merci !",
            "ar": f"تم تأكيد طلبك! {summary}. شكراً!",
            "en": f"Order confirmed! {summary}. Thank you!",
        }
        return msgs.get(lang, msgs["fr"])

    def _cancel_order(self, lang: str) -> str:
        """Annule la commande. Rien à notifier au backend : tant que la
        commande n'est pas confirmée, le dashboard staff ne l'a jamais vue
        (main.py ne connaît la commande qu'à partir de l'événement "confirmed")."""
        self.state.order    = []
        self.state.order_id = None
        self.state.turn     = 0
        msgs = {
            "fr": "Commande annulée. Repartons de zéro. Que souhaitez-vous commander ?",
            "ar": "تم إلغاء الطلب. لنبدأ من جديد. ماذا تريد أن تطلب؟",
            "en": "Order cancelled. Let's start over. What would you like to order?",
        }
        return msgs.get(lang, msgs["fr"])

    def _handle_payment(self, raw_text: str, lang: str) -> str:
        """
        Enregistre le mode de paiement choisi par le client et informe qu'il
        sera transmis au personnel. Le robot n'encaisse JAMAIS réellement —
        pas de terminal de paiement connecté — il ne fait qu'enregistrer
        l'intention pour que le staff s'en charge physiquement.
        """
        cash = bool(_CASH_RE.search(raw_text))
        card = bool(_CARD_RE.search(raw_text))
        tip  = bool(_TIP_RE.search(raw_text))

        # Le paiement concerne uniquement une commande déjà confirmée.
        if not self.state.order:
            msgs = {
                "fr": "Votre commande est vide pour l'instant. Que souhaitez-vous commander ?",
                "ar": "طلبك فارغ في الوقت الحالي. ماذا تريد أن تطلب؟",
                "en": "Your order is empty for now. What would you like to order?",
            }
            return msgs.get(lang, msgs["fr"])

        if not self.state.confirmed or not self.state.order_id:
            msgs = {
                "fr": "Veuillez d'abord confirmer votre commande avant de demander le paiement.",
                "ar": "يرجى تأكيد طلبك أولاً قبل طلب الدفع.",
                "en": "Please confirm your order before requesting payment.",
            }
            return msgs.get(lang, msgs["fr"])

        total  = sum(o.prix * o.quantity for o in self.state.order)
        devise = menu.restaurant["devise"]

        # Pourboire mentionné SEUL (sans mode de paiement précisé) : le
        # robot ne l'encaisse pas lui-même, mais ne bloque pas non plus —
        # il redirige simplement vers le personnel.
        if tip and not (cash or card):
            msgs = {
                "fr": "Je ne peux pas encaisser de pourboire moi-même, mais le personnel "
                      "s'en chargera avec plaisir. Souhaitez-vous régler en espèces ou par carte ?",
                "ar": "لا يمكنني تحصيل الإكرامية بنفسي، لكن الطاقم سيسعد بذلك. "
                      "هل تريد الدفع نقداً أم بالبطاقة؟",
                "en": "I can't collect a tip myself, but our staff will be happy to. "
                      "Would you like to pay in cash or by card?",
            }
            return msgs.get(lang, msgs["fr"])

        # Ni cash ni carte détecté malgré l'intent "paiement" (cas rare,
        # ex: "paiement" seul sans précision) → on demande le mode.
        if not (cash or card):
            msgs = {
                "fr": "Souhaitez-vous régler en espèces ou par carte ?",
                "ar": "هل تريد الدفع نقداً أم بالبطاقة؟",
                "en": "Would you like to pay in cash or by card?",
            }
            return msgs.get(lang, msgs["fr"])

        if cash and card:
            msgs = {
                "fr": "J'ai entendu espèces et carte. Quel mode de paiement choisissez-vous ?",
                "ar": "سمعت الدفع نقداً وبالبطاقة. أي طريقة دفع تختار؟",
                "en": "I heard both cash and card. Which payment method do you choose?",
            }
            return msgs.get(lang, msgs["fr"])

        # Valeur canonique "especes"/"carte" (français) : c'est le vocabulaire
        # attendu par db.py / main.py côté backend, pour que tout le système
        # parle la même langue de bout en bout.
        self.state.payment_method = "especes" if cash else "carte"
        mode_label = {
            "fr": {"especes": "en espèces", "carte": "par carte"},
            "ar": {"especes": "نقداً",      "carte": "بالبطاقة"},
            "en": {"especes": "in cash",    "carte": "by card"},
        }[lang if lang in ("fr", "ar", "en") else "fr"][self.state.payment_method]

        order_lines = [
            f"{self._format_item_label(o)} = {o.prix * o.quantity:.3f} {devise}"
            for o in self.state.order
        ]
        # 1. Audit trail permanent (JSONL) — survit même si le dashboard est éteint
        log_payment_audit(
            session_id=self.session.session_id,
            order_id=self.state.order_id,
            payment_method=self.state.payment_method,
            order_items=order_lines,
            total=total,
            devise=devise,
            lang=lang,
            table=self.state.table_id,
        )
        # 2. Alerte d'INTENTION vers le dashboard (PAS un vrai encaissement) :
        # le robot n'a aucune idée si le plat est déjà servi, et db.py exige
        # le statut "servie" avant d'accepter un vrai paiement. Le staff voit
        # juste "le client à cette table veut payer en espèces/carte" et
        # encaisse réellement, avec le vrai montant reçu, via la modale du
        # dashboard une fois le plat servi.
        push_payment_live(
            order_id=self.state.order_id,
            table=self.state.table_id,
            payment_method=self.state.payment_method,
            items=order_lines,
            total=total,
            devise=devise,
            lang=lang,
            session_id=self.session.session_id,
        )

        self.state.finished = True
        msgs = {
            "fr": f"Demande de paiement {mode_label} enregistrée pour un total de {total:.3f} {devise}. "
                  f"Le personnel va la traiter.",
            "ar": f"تم تسجيل طلب الدفع {mode_label} بمجموع {total:.3f} {devise}. "
                  f"سيعالج الطاقم طلبك.",
            "en": f"Payment request {mode_label} recorded for a total of {total:.3f} {devise}. "
                  f"Staff will process it.",
        }
        return msgs.get(lang, msgs["fr"])

    def _get_total(self, lang: str) -> str:
        """Retourne le total de la commande."""
        if not self.state.order:
            msgs = {
                "fr": "Votre commande est vide pour l'instant.",
                "ar": "طلبك فارغ في الوقت الحالي.",
                "en": "Your order is empty for now.",
            }
            return msgs.get(lang, msgs["fr"])

        total  = sum(o.prix * o.quantity for o in self.state.order)
        devise = menu.restaurant["devise"]
        msgs = {
            "fr": f"Le total de votre commande est de {total:.3f} {devise}.",
            "ar": f"مجموع طلبك هو {total:.3f} {devise}.",
            "en": f"Your order total is {total:.3f} {devise}.",
        }
        return msgs.get(lang, msgs["fr"])

    def _format_order(self, lang: str) -> str:
        """Formate la commande en cours pour le prompt et les confirmations."""
        if not self.state.order:
            return ""
        lines = []
        total = 0.0
        for o in self.state.order:
            subtotal = o.prix * o.quantity
            total   += subtotal
            lines.append(f"{self._format_item_label(o)} = {subtotal:.3f} {menu.restaurant['devise']}")
        lines.append(f"Total : {total:.3f} {menu.restaurant['devise']}")
        return " | ".join(lines)

    def _add_to_history(self, user_text: str, assistant_text: str):
        """Ajoute un tour à l'historique LLM (fenêtre glissante de 6 tours)."""
        self.state.history.append({"role": "user",      "content": user_text})
        self.state.history.append({"role": "assistant", "content": assistant_text})
        # Garde seulement les 6 derniers tours (12 messages)
        if len(self.state.history) > 12:
            self.state.history = self.state.history[-12:]

    def _log_turn(
        self,
        stt_text:  str,
        lang:      str,
        intent:    str,
        score:     float,
        entities:  dict,
        response:  str,
        t0:        float,
    ) -> None:
        """Journalise le tour courant via session_logger."""
        duration_ms = (time.perf_counter() - t0) * 1000
        order_state = [
            f"{self._format_item_label(o)} = {o.prix * o.quantity:.3f} {menu.restaurant['devise']}"
            for o in self.state.order
        ]
        try:
            self.session.log_turn(
                turn=self.state.turn,
                stt_text=stt_text,
                lang=lang,
                intent=intent,
                score=score,
                entities=entities,
                response=response,
                order=order_state,
                duration_ms=duration_ms,
            )
        except Exception as e:
            logger.warning(f"⚠️  Journalisation échouée : {e}")

    def _msg(self, key: str) -> str:
        msgs = {
            "max_turns": {
                "fr": "Nous avons dépassé la durée maximale. Je vais réinitialiser notre conversation.",
                "ar": "تجاوزنا الحد الأقصى للمحادثة. سأعيد تشغيل المحادثة.",
                "en": "We've exceeded the maximum conversation length. I'll reset our conversation.",
            }
        }
        return msgs.get(key, {}).get(self.state.lang, "")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    dm = DialogManager()
    print("=== Test DialogManager (avec confidence_checker / order_validator / fallback_handler / session_logger) ===\n")
    turns = [
        ("bonjour", "fr"),
        ("je voudrais un couscous agneau s'il vous plaît", "fr"),
        ("et aussi un café espresso", "fr"),
        ("combien ça fait ?", "fr"),
        ("bla bla bla xyz", "fr"),          # test confidence_checker / fallback
        ("c'est tout merci", "fr"),
    ]
    for text, lang in turns:
        print(f"🎤 Client : {text}")
        response = dm.process(text, lang)
        print(f"🤖 Robot  : {response}\n")

    print("--- Résumé session ---")
    print(dm.session.summary())
