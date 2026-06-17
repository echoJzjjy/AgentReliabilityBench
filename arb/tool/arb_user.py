"""Deterministic τ-bench user simulator (no LLM API required for evaluation)."""

from __future__ import annotations

import re

from arb.tool.clarification import clarification_matches_slot

_ORDER_RE = re.compile(r"#W\d{7}", re.I)
_ZIP_RE = re.compile(r"\b(\d{5})\b")
_EMAIL_RE = re.compile(r"\b[\w.+-]+@[\w.-]+\.\w+\b", re.I)


class ArbRuleUserSimulationEnv:
    """Rule-based user: reveals masked slots on clarification; answers from instruction."""

    metadata = {"type": "arb_rule_user"}

    def __init__(
        self,
        instruction: str,
        *,
        masked_slots: list[dict] | None = None,
        clarification_enabled: bool = True,
    ) -> None:
        self.instruction = instruction
        self.masked_slots = list(masked_slots or [])
        self.clarification_enabled = clarification_enabled
        self._revealed: set[str] = set()
        self._turn = 0

    def reset(self, instruction: str | None = None) -> str:
        if instruction is not None:
            self.instruction = instruction
        self._revealed.clear()
        self._turn = 0
        return _opening_message(self.instruction)

    def step(self, content: str) -> str:
        self._turn += 1
        text = (content or "").strip()
        lower = text.lower()

        if "###STOP###" in text:
            return "###STOP###"

        if any(
            m in lower
            for m in (
                "report_failure:",
                "cannot complete",
                "can't complete",
                "unable to complete",
                "not possible to complete",
                "impossible to",
            )
        ):
            return "###STOP###"

        if self.clarification_enabled and self.masked_slots:
            for slot in self.masked_slots:
                sid = slot.get("slot_id", "")
                if sid in self._revealed:
                    continue
                if clarification_matches_slot(text, slot):
                    self._revealed.add(sid)
                    label = slot.get("slot_type", "information").replace("_", " ")
                    return f"The {label} is {slot['original_text']}."

        answer = _answer_from_instruction(text, self.instruction)
        if answer:
            return answer

        if self._turn >= 8:
            return "###STOP###"

        return (
            "I'm not sure I understand. Could you tell me what you need from me, "
            "or ask specifically about my order, zip code, or email?"
        )

    def get_total_cost(self) -> float:
        return 0.0

    @property
    def revealed_slot_ids(self) -> set[str]:
        return set(self._revealed)


def _opening_message(instruction: str) -> str:
    """Natural opener without dumping the full τ-bench instruction."""
    instr = instruction.strip()
    if not instr:
        return "Hi, I need help with my account."

    first = re.split(r"(?<=[.!?])\s+", instr, maxsplit=1)[0].strip()
    if len(first) > 220:
        first = first[:217] + "..."
    if first.lower().startswith("you are "):
        m = re.search(r"you (?:received|want|need|wish|are trying)[^.!?]*[.!?]", instr, re.I)
        if m:
            first = m.group(0).strip()
        else:
            first = "Hi, I need help with my order."

    persona = re.match(r"^You are ([^.!?]+)[.!?]", instr, re.I)
    if persona and len(first) < 40:
        name = persona.group(1).split(" in ")[0].strip()
        return f"Hi, this is {name}. I need help with my request."

    return first


def _answer_from_instruction(agent_message: str, instruction: str) -> str | None:
    """If the agent asks for a fact present in the instruction, return it."""
    q = agent_message.lower()
    instr = instruction

    if any(k in q for k in ("order id", "order number", "which order", "order #")):
        orders = _ORDER_RE.findall(instr)
        if orders:
            return f"My order id is {orders[0]}."

    if any(k in q for k in ("zip", "zip code", "postal")):
        zips = _ZIP_RE.findall(instr)
        if zips:
            return f"My zip code is {zips[0]}."

    if "email" in q:
        emails = _EMAIL_RE.findall(instr)
        if emails:
            return f"My email is {emails[0]}."

    if any(k in q for k in ("user id", "account id", "customer id")):
        m = re.search(r"\b([a-z]+_[a-z]+_\d{4})\b", instr)
        if m:
            return f"My user id is {m.group(1)}."

    if any(k in q for k in ("reservation", "confirmation", "booking")):
        m = re.search(r"\b[A-Z0-9]{6,8}\b", instr)
        if m:
            return f"The confirmation number is {m.group(0)}."

    return None
