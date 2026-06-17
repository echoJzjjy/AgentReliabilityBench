"""Match agent clarification questions to masked slots."""

from __future__ import annotations

SLOT_TYPE_CUES: dict[str, list[str]] = {
    "order_id": [
        "order",
        "order id",
        "order number",
        "order #",
        "tracking",
        "which order",
        "order_id",
        "w2378156",
    ],
    "zip_code": [
        "zip",
        "zip code",
        "postal",
        "postcode",
        "area code",
        "location code",
        "what city",
        "where do you live",
    ],
    "email": [
        "email",
        "e-mail",
        "mail address",
        "email address",
    ],
}


def clarification_matches_slot(question: str, slot: dict) -> bool:
    q = question.lower()
    stype = slot.get("slot_type", "")
    for kw in slot.get("clarification_keywords", []):
        if str(kw).lower() in q:
            return True
    for cue in SLOT_TYPE_CUES.get(stype, []):
        if cue.lower() in q:
            return True
    tok = slot.get("masked_token", "").lower()
    if tok and tok in q:
        return True
    return False
