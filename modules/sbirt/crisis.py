"""Deterministic crisis safety net (P2) — the non-LLM backstop.

Crisis detection must NOT depend on an LLM judgment call: this module scans the
user's transcribed words with fixed, reviewable patterns and, on a hit, the
pipeline speaks a FIXED response (hardcoded below, pre-rendered to a cached
clip) and routes the session into the crisis protocol. The LLM's own crisis
handling (prompt) stays active — detection is the UNION of both, so the
deterministic net catches what the model misses, and the model catches subtle
cues no pattern can. Over-triggering is acceptable by design; missing is not.

Categories mirror referral.py CRISIS_PROTOCOL. Patterns are word-boundary,
case-insensitive regexes tuned for ASR text (lowercase, unreliable
punctuation). Deliberate scope notes:
  • "withdrawal" alone is NOT a trigger — DAST item 9 asks "Have you ever
    experienced withdrawal symptoms...", so a normal screening answer would
    fire on every positive patient. The withdrawal category targets the
    ACUTE danger presentation instead (seizures, DTs, hallucinations).
  • Negations ("I'm not suicidal") still trigger — a false positive costs one
    empathetic safety message; a false negative can cost a life.

The RESPONSES texts are clinical safety copy assembled from referral.py
(crisis lines 988/911/SAMHSA) — flagged for clinician review; they are fixed
strings, never LLM-generated or paraphrased at runtime.
"""

from __future__ import annotations

import re
from dataclasses import dataclass


@dataclass(frozen=True)
class CrisisHit:
    category: str      # "suicide" | "overdose" | "withdrawal" | "acute_danger"
    pattern: str       # the pattern that fired (safe for logs: no user text)


# Category order = check order = severity order; first hit wins.
_PATTERNS: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("suicide", (
        r"\bsuicid\w*",
        # 'myself' only — '...is killing me' is a common idiom (esp. in a
        # drinking context: 'this hangover is killing me') and must not fire.
        r"\bkill(?:ing)?\s+myself\b",
        r"\bend(?:ing)?\s+(?:my|it)\s+(?:life|all)\b",
        r"\btak(?:e|ing)\s+my\s+(?:own\s+)?life\b",
        r"\b(?:hurt|harm|cutt?)(?:ing)?\s+myself\b",
        r"\bself[\s-]?harm\w*",
        r"\bwant(?:ed)?\s+to\s+die\b",
        r"\bbetter\s+off\s+dead\b",
        r"\b(?:don'?t|do\s+not|no\s+reason\s+to)\s+want\s+to\s+(?:live|be\s+alive)\b",
        r"\bno\s+reason\s+to\s+(?:live|keep\s+going)\b",
    )),
    ("overdose", (
        r"\boverdos\w*",
        r"\btook\s+too\s+many\s+(?:pills|of\s+them)\b",
        r"\btook\s+(?:a\s+)?whole\s+bottle\b",
        r"\bcan'?t\s+wake\s+(?:him|her|them)\s+up\b",
        r"\bnot\s+breathing\b",
    )),
    ("withdrawal", (
        # Acute alcohol/benzo withdrawal danger — abrupt cessation can be fatal.
        r"\bseizures?\b",
        r"\bdelirium\s+tremens\b",
        r"\bthe\s+dts\b",
        r"\bhallucinat\w*",
        r"\bshak(?:ing|es)\s+(?:real\s+)?bad(?:ly)?\b",
    )),
    ("acute_danger", (
        r"\b(?:kill|hurt|shoot|stab)(?:ing)?\s+(?:him|her|them|someone|somebody|my\s+\w+)\b",
        r"\bgoing\s+to\s+hurt\s+\w+\b",
        r"\bpassed\s+out\s+and\s+won'?t\s+wake\b",
    )),
)

_COMPILED = tuple(
    (category, tuple(re.compile(p, re.IGNORECASE) for p in patterns))
    for category, patterns in _PATTERNS
)


def detect(text: str) -> CrisisHit | None:
    """Scan one utterance; return the first (most severe) crisis hit, else None.
    Pure function — no LLM, no IO — so it can never be down when needed."""
    if not text or not text.strip():
        return None
    for category, patterns in _COMPILED:
        for rx in patterns:
            if rx.search(text):
                return CrisisHit(category=category, pattern=rx.pattern)
    return None


# Fixed spoken responses — one per category, pre-rendered to cached clips so a
# crisis answer plays instantly with no LLM/TTS/FLOAT on the hot path.
# Copy assembled from referral.py CRISIS_PROTOCOL / CRISIS_LINES; pending
# clinician sign-off, tracked as a human decision point.
RESPONSES: dict[str, str] = {
    "suicide": (
        "Thank you for telling me — I'm really glad you said that, and I want "
        "to make sure you're safe. Please reach the 988 Suicide and Crisis "
        "Lifeline right now by calling or texting 9 8 8. If you are in "
        "immediate danger, please call 9 1 1. Would you like to talk about "
        "what's going on?"
    ),
    "overdose": (
        "That sounds like it could be a medical emergency. If you or someone "
        "with you may have overdosed, please call 9 1 1 right away. If "
        "opioids may be involved and naloxone, also called Narcan, is "
        "available, use it. Please get medical help now — we can continue "
        "talking after you are safe."
    ),
    "withdrawal": (
        "What you're describing can be a sign of serious withdrawal, and "
        "stopping alcohol or sedatives suddenly can be dangerous. Please seek "
        "urgent medical care now, or call 9 1 1 if it's severe. Please don't "
        "try to get through this alone or quit cold turkey without medical "
        "support."
    ),
    "acute_danger": (
        "It sounds like someone may be in immediate danger. Please call "
        "9 1 1 right now. If you can, stay somewhere safe. You can also call "
        "or text 9 8 8 to talk to a crisis counselor at any time."
    ),
}

# Every category must have a response — a KeyError in a crisis is unacceptable.
assert set(RESPONSES) == {c for c, _ in _PATTERNS}
