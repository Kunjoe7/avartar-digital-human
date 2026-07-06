"""Runtime clinical state machine (P3) — the study protocol as executable code.

Until now the SBIRT flow existed only as prose inside the system prompt and the
LLM improvised its way through it. This module makes the protocol operative:
question order, skip rules, arm selection (AUDIT vs DAST), feedback routing and
the brief-intervention sequence are DETERMINISTIC transitions over coded input.
The LLM's remaining conversational jobs are emitted explicitly as `LLMSay`
directives — a single node-scoped utterance (reflect / summarize / ask the one
parameterized question) — never a protocol decision.

Contract with the pipeline:
  • `ClinicalSession` is per-user mutable state (owned by the Pipeline).
  • `advance(session, kind, value) -> Step` consumes ONE coded user input
    (kind matches `session.expect.kind`) and returns what to say next +
    what to expect. Coding free text is the NLU layer's job (llm.py).
  • `enter_crisis(session)` — the deterministic crisis net (crisis.py) or the
    coder may call this at ANY point; the protocol then pauses permanently for
    the session and every later turn is an LLM crisis-protocol turn. There is
    deliberately no automatic resume (a human-review decision, not a pattern's).
  • Unclear / ambiguous answers do NOT advance the machine: the pipeline
    re-asks via `repeat_step()` (optionally with a clarification preface).

This machine is pure Python over instruments.py / templates.py —
no LLM, no IO — so every branch is testable against the case cards.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

from . import templates
from .instruments import (assess, Assessment, BY_KEY, InvalidResponse, Item,
                          next_item_index, option_score, PRE_SCREEN)

logger = logging.getLogger(__name__)

# Which full instrument each positive pre-screen arm opens (protocol order:
# alcohol before drugs, matching the study dialogue).
ARM_INSTRUMENT = {"alcohol": "audit", "drugs": "dast_10"}
ARM_ORDER = ("alcohol", "drugs")


# --------------- Directives the machine emits ---------------

@dataclass(frozen=True)
class Say:
    """A FIXED utterance (verbatim script) — cacheable as a pre-rendered clip."""
    key: str    # stable content key, e.g. "audit.item.3" (cache identity)
    text: str


@dataclass(frozen=True)
class LLMSay:
    """One LLM-generated utterance, bounded to the current node by
    `instruction`. The LLM may phrase; it may not decide where the protocol
    goes next."""
    instruction: str


@dataclass(frozen=True)
class Expect:
    """What the next user input means (drives the NLU coder)."""
    kind: str                        # consent | option | number | open | end
    instrument: str | None = None    # for kind="option": instrument key or "prescreen"
    item_index: int | None = None    # for kind="option"


@dataclass(frozen=True)
class Step:
    node: str
    utterances: tuple
    expect: Expect


class ProtocolError(RuntimeError):
    """The pipeline fed an event that doesn't match the machine's expectation —
    always a wiring bug, never user error (user ambiguity must not advance)."""


# --------------- Per-session clinical state ---------------

@dataclass
class ClinicalSession:
    node: str = "consent"
    expect: Expect = field(default_factory=lambda: Expect("consent"))
    consent: str | None = None                 # "yes" | "no"
    prescreen: dict[str, int] = field(default_factory=dict)   # key -> code
    arms: list[str] = field(default_factory=list)             # pending arms
    arm: str | None = None                                    # active arm
    responses: dict[str, dict[int, int]] = field(default_factory=dict)
    assessments: dict[str, Assessment] = field(default_factory=dict)
    readiness: dict[str, int] = field(default_factory=dict)   # arm -> 0..10
    declined: list[str] = field(default_factory=list)         # declined permission keys (audit)
    crisis: bool = False
    last_step: Step | None = None

    def instrument(self):
        return BY_KEY[ARM_INSTRUMENT[self.arm]]

    def to_audit_dict(self) -> dict:
        """Structured, non-free-text summary for the audit record (P7b):
        codes/scores/zones only — no transcripts."""
        return {
            "node": self.node,
            "consent": self.consent,
            "prescreen": dict(self.prescreen),
            "responses": {k: dict(v) for k, v in self.responses.items()},
            "assessments": {
                k: {"score": a.score, "zone": a.zone, "complete": a.complete}
                for k, a in self.assessments.items()
            },
            "readiness": dict(self.readiness),
            "declined": list(self.declined),
            "crisis": self.crisis,
        }


# --------------- Helpers ---------------

def _fixed(key: str) -> Say:
    return Say(key, templates.FIXED[key])


def _item_say(instrument_key: str, index: int, item: Item) -> Say:
    return Say(f"{instrument_key}.item.{index}", item.text)


def _step(session: ClinicalSession, node: str, utterances: list,
          expect: Expect) -> Step:
    session.node = node
    session.expect = expect
    step = Step(node, tuple(utterances), expect)
    session.last_step = step
    logger.info("[clinical] -> %s (expect %s)", node, expect.kind)
    return step


def repeat_step(session: ClinicalSession) -> Step:
    """Re-emit the current step (after an ambiguous answer, the pipeline
    clarifies and asks again; the machine does not move)."""
    if session.last_step is None:
        return start(session)
    return session.last_step


# --------------- Entry ---------------

def start(session: ClinicalSession) -> Step:
    """The fixed greeting (config.GREETING_TEXT, delivered by the pipeline)
    already asked for consent; the machine starts by expecting that answer."""
    return _step(session, "consent", [], Expect("consent"))


def enter_crisis(session: ClinicalSession) -> Step:
    """Deterministic crisis: pause the protocol permanently for this session.
    The pipeline speaks the fixed crisis response (crisis.py); every later
    turn is an LLM crisis-protocol turn."""
    session.crisis = True
    return _step(session, "crisis", [], Expect("open"))


_CRISIS_INSTRUCTION = (
    "Crisis protocol is active. In one or two sentences, respond with empathy "
    "and urgency to what the person just said, keep them talking, and repeat "
    "the crisis lines (call or text 988; call 911 if in immediate danger) "
    "when appropriate. Do not resume any screening."
)


# --------------- Node handlers ---------------

def _on_consent(session: ClinicalSession, value) -> Step:
    if value == "no":
        session.consent = "no"
        # The fixed decline text (config.DECLINE_TEXT) is spoken by the
        # pipeline's existing decline path; the machine just terminates.
        return _step(session, "declined", [], Expect("end"))
    session.consent = "yes"
    return _ask_prescreen(session, 0)


def _ask_prescreen(session: ClinicalSession, index: int) -> Step:
    q = PRE_SCREEN[index]
    return _step(session, f"prescreen.{q.key}",
                 [Say(f"prescreen.{q.key}", q.item.text)],
                 Expect("option", instrument="prescreen", item_index=index))


def _on_prescreen(session: ClinicalSession, value) -> Step:
    index = session.expect.item_index
    q = PRE_SCREEN[index]
    code = int(value)
    if not 0 <= code < len(q.item.options):
        raise ProtocolError(f"prescreen {q.key}: invalid code {code}")
    session.prescreen[q.key] = code
    if index + 1 < len(PRE_SCREEN):
        return _ask_prescreen(session, index + 1)
    # All three answered: queue positive arms in protocol order. Pre-screen
    # options are (negative, positive) with scores 0/1, so code > 0 = positive.
    session.arms = [arm for arm in ARM_ORDER if session.prescreen.get(arm, 0) > 0]
    if not session.arms:
        return _close(session, prefix=[_fixed("prescreen.all_negative")])
    return _start_arm(session)


def _start_arm(session: ClinicalSession) -> Step:
    session.arm = session.arms.pop(0)
    if session.arm == "alcohol":
        return _step(session, "alcohol.qf", [_fixed("alcohol.qf")],
                     Expect("open"))
    return _step(session, "drugs.kind", [_fixed("drugs.kind")], Expect("open"))


def _finish_arm(session: ClinicalSession, prefix: list | None = None) -> Step:
    if session.arms:
        head = list(prefix or [])
        step = _start_arm(session)
        if head:  # prepend transition utterances to the new arm's first step
            step = Step(step.node, tuple(head) + step.utterances, step.expect)
            session.last_step = step
        return step
    return _close(session, prefix=prefix)


def _close(session: ClinicalSession, prefix: list | None = None) -> Step:
    utterances = list(prefix or []) + [_fixed("close")]
    return _step(session, "closed", utterances, Expect("end"))


# --- Alcohol arm: Q/F exploration -> education -> AUDIT permission ---

def _on_alcohol_qf(session: ClinicalSession, value) -> Step:
    return _step(session, "alcohol.edu.permission",
                 [_fixed("alcohol.edu.permission")], Expect("consent"))


def _on_alcohol_edu_perm(session: ClinicalSession, value) -> Step:
    utterances = []
    if value == "yes":
        utterances += [_fixed("alcohol.edu.standard_drink"),
                       _fixed("alcohol.edu.limits")]
    else:
        session.declined.append("alcohol.edu.permission")
    # Either way the protocol proceeds to ask permission for the AUDIT itself.
    utterances.append(_fixed("alcohol.screen.permission"))
    return _step(session, "alcohol.screen.permission", utterances,
                 Expect("consent"))


# --- Drug arm: kind -> quantity/frequency (parameterized) -> DAST permission ---

def _on_drugs_kind(session: ClinicalSession, value) -> Step:
    return _step(session, "drugs.qf", [LLMSay(
        "In one sentence, ask how much and how often the person uses the "
        "drug or drugs they just named. Ask nothing else."
    )], Expect("open"))


def _on_drugs_qf(session: ClinicalSession, value) -> Step:
    return _step(session, "drugs.screen.permission",
                 [_fixed("drugs.screen.permission")], Expect("consent"))


# --- Screening: administer the instrument item by item (skip rules apply) ---

def _on_screen_permission(session: ClinicalSession, value) -> Step:
    if value != "yes":
        session.declined.append(f"{session.arm}.screen.permission")
        return _finish_arm(session, prefix=[_fixed("permission.declined")])
    instrument = session.instrument()
    session.responses.setdefault(instrument.key, {})
    utterances = []
    if instrument.preamble:
        utterances.append(Say(f"{instrument.key}.preamble", instrument.preamble))
    return _ask_next_item(session, utterances)


def _ask_next_item(session: ClinicalSession, prefix: list | None = None) -> Step:
    instrument = session.instrument()
    responses = session.responses[instrument.key]
    idx = next_item_index(instrument, responses)
    if idx is None:
        return _screen_complete(session)
    item = instrument.items[idx]
    utterances = list(prefix or []) + [_item_say(instrument.key, idx, item)]
    return _step(session, f"screening.{instrument.key}.{idx}", utterances,
                 Expect("option", instrument=instrument.key, item_index=idx))


def _on_screen_item(session: ClinicalSession, value) -> Step:
    instrument = session.instrument()
    idx = session.expect.item_index
    code = int(value)
    # Validates the code against the instrument (raises InvalidResponse on a
    # coder bug rather than silently mis-scoring).
    option_score(instrument, idx, code)
    session.responses[instrument.key][idx] = code
    return _ask_next_item(session)


def _screen_complete(session: ClinicalSession) -> Step:
    instrument = session.instrument()
    assessment = assess(instrument, session.responses[instrument.key])
    session.assessments[instrument.key] = assessment
    logger.info("[clinical] %s complete: score=%d zone=%s",
                instrument.key, assessment.score, assessment.zone)
    return _step(session, f"{session.arm}.feedback.permission",
                 [_fixed(f"{session.arm}.feedback.permission")],
                 Expect("consent"))


# --- Feedback -> (healthy: done) / (else: brief intervention) ---

def _on_feedback_permission(session: ClinicalSession, value) -> Step:
    instrument = session.instrument()
    assessment = session.assessments[instrument.key]
    if value != "yes":
        session.declined.append(f"{session.arm}.feedback.permission")
        return _finish_arm(session, prefix=[_fixed("permission.declined")])
    zone = assessment.zone
    feedback = Say(f"feedback.{instrument.key}.{zone}",
                   templates.feedback_text(instrument.key, zone))
    if zone == "healthy":
        return _finish_arm(session, prefix=[feedback])
    utterances = [feedback]
    if (instrument.key, zone) not in templates.FEEDBACK_ASKS_BI:
        # e.g. the source's dependent-drug text doesn't end with the BI ask.
        utterances.append(Say(f"bi.permission.{session.arm}",
                              templates.bi_permission(session.arm)))
    return _step(session, f"bi.permission.{session.arm}", utterances,
                 Expect("consent"))


# --- Brief intervention: decisional balance -> readiness ruler ---

def _on_bi_permission(session: ClinicalSession, value) -> Step:
    if value != "yes":
        session.declined.append(f"{session.arm}.bi.permission")
        return _finish_arm(session, prefix=[_fixed("permission.declined")])
    return _step(session, "bi.likes",
                 [Say(f"bi.likes.{session.arm}", templates.bi_likes(session.arm))],
                 Expect("open"))


def _on_bi_likes(session: ClinicalSession, value) -> Step:
    return _step(session, "bi.dislikes",
                 [Say(f"bi.dislikes.{session.arm}", templates.bi_dislikes(session.arm))],
                 Expect("open"))


def _on_bi_dislikes(session: ClinicalSession, value) -> Step:
    return _step(session, "bi.ruler", [
        LLMSay("In one or two sentences, summarize back first what the person "
               "said they LIKE about their use, then what they DISLIKE, using "
               "their own words. Do not add advice or new clinical content."),
        Say(f"bi.recommend.{session.arm}", templates.bi_recommend(session.arm)),
        Say(f"bi.ruler.{session.arm}", templates.bi_ruler(session.arm)),
    ], Expect("number"))


def _on_bi_ruler(session: ClinicalSession, value) -> Step:
    value = int(value)
    if not 0 <= value <= 10:
        raise ProtocolError(f"readiness ruler out of range: {value}")
    session.readiness[session.arm] = value
    # Key carries the value: each of the 11 variants is its own cached clip.
    return _step(session, "bi.why_not_lower",
                 [Say(f"bi.why_not_lower.{value}",
                      templates.bi_why_not_lower(value))],
                 Expect("open"))


def _on_bi_why_lower(session: ClinicalSession, value) -> Step:
    value = session.readiness[session.arm]
    return _step(session, "bi.why_not_higher",
                 [Say(f"bi.why_not_higher.{value}",
                      templates.bi_why_not_higher(value))],
                 Expect("open"))


def _on_bi_why_higher(session: ClinicalSession, value) -> Step:
    return _step(session, "bi.leaves_you", [
        LLMSay("In one or two sentences, summarize the person's reasons for "
               "not being a 9 or 10, then their reasons for not being a 1 or "
               "2, using their own words. Do not add advice."),
        Say("bi.leaves_you", templates.BI_LEAVES_YOU),
    ], Expect("open"))


def _on_bi_leaves(session: ClinicalSession, value) -> Step:
    reflect = LLMSay("In one brief sentence, reflect what the person just "
                     "said about where this leaves them. No new questions.")
    return _finish_arm(session, prefix=[reflect])


_HANDLERS = {
    "consent": _on_consent,
    "prescreen.tobacco": _on_prescreen,
    "prescreen.alcohol": _on_prescreen,
    "prescreen.drugs": _on_prescreen,
    "alcohol.qf": _on_alcohol_qf,
    "alcohol.edu.permission": _on_alcohol_edu_perm,
    "alcohol.screen.permission": _on_screen_permission,
    "drugs.kind": _on_drugs_kind,
    "drugs.qf": _on_drugs_qf,
    "drugs.screen.permission": _on_screen_permission,
    "alcohol.feedback.permission": _on_feedback_permission,
    "drugs.feedback.permission": _on_feedback_permission,
    "bi.permission.alcohol": _on_bi_permission,
    "bi.permission.drugs": _on_bi_permission,
    "bi.likes": _on_bi_likes,
    "bi.dislikes": _on_bi_dislikes,
    "bi.ruler": _on_bi_ruler,
    "bi.why_not_lower": _on_bi_why_lower,
    "bi.why_not_higher": _on_bi_why_higher,
    "bi.leaves_you": _on_bi_leaves,
}

# --------------- The transition function ---------------

def advance(session: ClinicalSession, kind: str, value) -> Step:
    """Consume ONE coded user input and move the protocol forward.
    kind mirrors session.expect.kind; value is the coded payload
    (consent: "yes"|"no"; option: int code; number: int; open: transcript)."""
    if session.crisis:
        return _step(session, "crisis", [LLMSay(_CRISIS_INSTRUCTION)],
                     Expect("open"))
    if session.expect.kind != kind:
        raise ProtocolError(
            f"machine at node {session.node!r} expects "
            f"{session.expect.kind!r}, got {kind!r}")
    # Screening item nodes are dynamic ("screening.<instrument>.<idx>").
    handler = _HANDLERS.get(session.node) or (
        _on_screen_item if session.node.startswith("screening.") else None)
    if handler is None:
        raise ProtocolError(f"no handler for node {session.node!r}")
    return handler(session, value)
