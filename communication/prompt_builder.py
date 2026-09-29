"""
prompt_builder.py
=================
Construit le prompt système et utilisateur pour le LLM.
Le prompt injecte le menu, la commande en cours et le contexte dialogue.
"""

from menu_loader import get_menu

menu = get_menu()

# ── Prompt système par langue ──────────────────────────────────────────────────
_SYSTEM_PROMPTS = {
    "fr": """Tu es NexorBot, l'assistant vocal du restaurant NexorCafé.
Tu aides les clients à passer leur commande de manière naturelle et efficace.

Règles STRICTES :
- Tu réponds UNIQUEMENT sur le sujet de la commande et du menu.
- Tu réponds en français si le client parle français.
- Tu réponds en arabe si le client parle arabe.
- Tu réponds en anglais si le client parle anglais.
- Tes réponses sont COURTES (1-2 phrases max) et adaptées à la lecture TTS.
- Tu ne génères JAMAIS de HTML, markdown, listes à puces, ou emojis.
- Si un item n'est pas dans le menu, tu dis poliment qu'il n'est pas disponible.
- Tu confirmes toujours les items ajoutés avec leur prix.
- Le paiement est géré par un autre module ; ne tente jamais toi-même une transaction.

Menu actuel :
{menu_text}

Commande en cours :
{order_summary}""",

    "ar": """أنت NexorBot، المساعد الصوتي لمطعم NexorCafé.
تساعد العملاء على تقديم طلباتهم بشكل طبيعي وفعّال.

القواعد الصارمة:
- تجيب فقط حول الطلب والقائمة.
- ردودك قصيرة (1-2 جملة فقط) ومناسبة للنطق الصوتي.
- لا تستخدم HTML أو markdown أو نقاط أو رموز تعبيرية.
- إذا لم يكن الصنف في القائمة، أخبر العميل بلطف.
- تؤكد دائماً الأصناف المضافة مع سعرها.

القائمة الحالية:
{menu_text}

الطلب الحالي:
{order_summary}""",

    "en": """You are NexorBot, the voice assistant for NexorCafé restaurant.
You help customers place their orders naturally and efficiently.

STRICT rules:
- You ONLY answer about orders and the menu.
- Your answers are SHORT (1-2 sentences max) and adapted for TTS.
- Never generate HTML, markdown, bullet points, or emojis.
- If an item is not on the menu, politely say it is not available.
- Always confirm added items with their price.
- Payment is handled by another module; never attempt a transaction yourself.

Current menu:
{menu_text}

Current order:
{order_summary}""",
}


def build_system_prompt(lang: str, order_summary: str) -> str:
    """Construit le prompt système avec menu et commande en cours."""
    template    = _SYSTEM_PROMPTS.get(lang, _SYSTEM_PROMPTS["fr"])
    menu_text   = menu.format_for_prompt(lang)
    return template.format(
        menu_text=menu_text,
        order_summary=order_summary if order_summary else _empty_order_msg(lang),
    )


def build_user_message(text: str, intent: str, entities: dict, lang: str) -> str:
    """
    Construit le message utilisateur enrichi avec le contexte NLP.
    On passe le texte brut + les entités extraites pour aider le LLM.
    """
    return text  # Le LLM reçoit le texte naturel, les entités sont dans le système


def build_messages(
    text: str,
    lang: str,
    intent: str,
    entities: dict,
    order_summary: str,
    history: list[dict],
) -> list[dict]:
    """
    Construit la liste complète de messages pour l'API Ollama.

    Format :
        [
          {"role": "system",    "content": "..."},
          {"role": "user",      "content": "..."},   # tour 1
          {"role": "assistant", "content": "..."},   # tour 1
          ...
          {"role": "user",      "content": text},    # tour actuel
        ]
    """
    system_prompt = build_system_prompt(lang, order_summary)

    messages = [{"role": "system", "content": system_prompt}]
    messages.extend(history)
    messages.append({"role": "user", "content": text})

    return messages


def _empty_order_msg(lang: str) -> str:
    msgs = {
        "fr": "Aucun item commandé pour le moment.",
        "ar": "لا توجد طلبات حتى الآن.",
        "en": "No items ordered yet.",
    }
    return msgs.get(lang, msgs["fr"])


if __name__ == "__main__":
    order = "1× Couscous agneau (18.000 TND)\n1× Café espresso (1.500 TND)\nTotal : 19.500 TND"
    prompt = build_system_prompt("fr", order)
    print("=== Test prompt_builder ===")
    print(prompt[:500], "...")