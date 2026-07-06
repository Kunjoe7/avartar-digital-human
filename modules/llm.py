import json
import logging
import re
import threading
from openai import OpenAI
import config

logger = logging.getLogger(__name__)

# Lazily built so an empty/misconfigured OPENROUTER_API_KEY doesn't crash the
# whole server at import time (OpenAI('') raises on construction).
_client_obj = None
_client_lock = threading.Lock()


def _client() -> OpenAI:
    global _client_obj
    if _client_obj is None:
        with _client_lock:
            if _client_obj is None:
                _client_obj = OpenAI(
                    base_url=config.OPENROUTER_BASE_URL,
                    api_key=config.OPENROUTER_API_KEY,
                )
    return _client_obj


def build_messages(history: list[dict], patient: dict | None = None) -> list[dict]:
    """Build the API message list: system prompt (plus any known patient facts) on
    top, then a snapshot of the caller-owned conversation history.

    The patient profile is injected fresh EVERY turn, independent of the history
    window, so key facts (age, sex, screening answers) survive the sliding-window
    trim and the model never loses them even after old turns scroll out."""
    system = config.SYSTEM_PROMPT
    if patient:
        system += ("\n\n=== KNOWN PATIENT (persists for the whole session) ===\n"
                   "These were established earlier — do NOT re-ask a field that is "
                   "already filled; use them to tailor screening and tone:\n"
                   + json.dumps(patient, ensure_ascii=False))
    return [{"role": "system", "content": system}] + list(history)


_EXTRACT_SYSTEM = (
    "You extract structured facts for an SBIRT substance-use screening from a "
    "counseling conversation. Output ONLY a JSON object with any of these keys you "
    "can determine from what the USER explicitly said; omit any you are unsure "
    "about. Keys: age (int), sex ('male'|'female'|'other'), substances (list of "
    "substances actually used), alcohol_use, tobacco_use, drug_use, rx_misuse "
    "(short strings), screening (object mapping instrument -> answers/score so far), "
    "readiness_stage, risk_level, notes (one short string). Never invent anything; "
    "if the user gave no new factual info, output {}."
)


def extract_patient_facts(history: list[dict]) -> dict:
    """Best-effort structured extraction of patient facts from recent conversation.
    Returns {} on ANY failure — this is a non-critical add-on that must never break
    a turn (it runs off the hot path and only updates the profile for next turn)."""
    convo = "\n".join(f"{m['role']}: {m.get('content', '')}" for m in history[-8:])
    if not convo.strip():
        return {}
    try:
        resp = _client().chat.completions.create(
            model=config.LLM_MODEL,
            messages=[{"role": "system", "content": _EXTRACT_SYSTEM},
                      {"role": "user", "content": convo}],
            temperature=0,
        )
        raw = (resp.choices[0].message.content or "").strip()
        start, end = raw.find("{"), raw.rfind("}")
        if start == -1 or end == -1 or end < start:
            return {}
        data = json.loads(raw[start:end + 1])
        return data if isinstance(data, dict) else {}
    except Exception as e:
        logger.info("patient extraction skipped (%s)", e)
        return {}


def chat(messages: list[dict]) -> str:
    """Non-streaming completion. `messages` is the full API message list (system +
    history + current user) built by the caller. This function does NOT mutate any
    history — the caller (Pipeline) owns and updates the conversation state."""
    response = _client().chat.completions.create(
        model=config.LLM_MODEL,
        messages=messages,
    )
    return response.choices[0].message.content or ""


def chat_stream(messages: list[dict],
                cancel_event: threading.Event | None = None):
    """Yield complete sentences as they arrive from the LLM stream.

    `messages` is the full API message list built by the caller. This function is
    pure: it does NOT touch any history — the caller accumulates the yielded
    sentences and owns the conversation state. Checks cancel_event between chunks.
    """
    stream = _client().chat.completions.create(
        model=config.LLM_MODEL,
        messages=messages,
        stream=True,
    )

    buffer = ""
    # Split ONLY on sentence-final punctuation so each spoken clip is a whole
    # sentence. The system prompt's short-acknowledgment-first rule keeps the first
    # clip short enough for a fast start without sub-sentence (comma) splitting.
    sentence_endings = {"。", "！", "？", ".", "!", "?", "\n"}

    for chunk in stream:
        if cancel_event and cancel_event.is_set():
            break

        if not chunk.choices:
            continue
        delta = chunk.choices[0].delta.content
        if delta:
            buffer += delta

            # Extract ALL complete sentences from the buffer.
            while True:
                split_pos = -1
                for i, ch in enumerate(buffer):
                    if ch in sentence_endings:
                        split_pos = i
                        break

                if split_pos == -1:
                    break

                sentence = buffer[: split_pos + 1].strip()
                buffer = buffer[split_pos + 1 :]
                if sentence:
                    yield sentence

    # Flush remaining buffer
    if buffer.strip() and not (cancel_event and cancel_event.is_set()):
        yield buffer.strip()


_CONSENT_SYSTEM = (
    "A health-screening tool asked the user for permission with the question: "
    "{question!r}. Read ONLY the user's reply and classify their intent as "
    "exactly one lowercase word:\n"
    "  yes — they clearly agree to proceed (incl. 'sure', 'ok', 'go ahead', 'no problem')\n"
    "  no — they clearly decline ('no', 'not now', \"I'd rather not\")\n"
    "  unclear — anything else, OR any sign of distress/crisis/off-topic.\n"
    "Output only that one word."
)

_DEFAULT_CONSENT_QUESTION = "May I ask you some questions about your health?"


def classify_consent(text: str, question: str = _DEFAULT_CONSENT_QUESTION) -> str:
    """Classify the user's reply to a permission question as
    'yes' | 'no' | 'unclear'. Conservative: distress/crisis/ambiguous -> 'unclear'
    so the caller clarifies instead of guessing. Returns 'unclear' on any failure."""
    if not text or not text.strip():
        return "unclear"
    try:
        resp = _client().chat.completions.create(
            model=config.LLM_MODEL,
            messages=[{"role": "system",
                       "content": _CONSENT_SYSTEM.format(question=question)},
                      {"role": "user", "content": text}],
            temperature=0,
        )
        out = (resp.choices[0].message.content or "").strip().lower()
        if out.startswith("yes"):
            return "yes"
        if out.startswith("no"):
            return "no"
        return "unclear"
    except Exception as e:
        logger.info("consent classification skipped (%s)", e)
        return "unclear"


# =====================================================================
# Constrained NLU coding (P4): free text -> option code, or AMBIGUOUS.
# The coder NEVER guesses: quantities/timeframes the user did not state
# route to a clarification, not to a code (case-card AUDIT item 10 rule).
# =====================================================================

AMBIGUOUS = "AMBIGUOUS"

_YES_RE = re.compile(r"^\s*(yes|yeah|yep|yup|sure|correct|i do|i have)\b", re.I)
_NO_RE = re.compile(r"^\s*(no|nope|nah|never|not really|i don'?t|i do not|i haven'?t)\b", re.I)

_WORD_NUMBERS = {
    "zero": 0, "one": 1, "two": 2, "three": 3, "four": 4, "five": 5,
    "six": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10,
}
_DIGIT_RE = re.compile(r"\b(10|[0-9])\b")

# Confidence below this -> clarify rather than code.
CODE_CONFIDENCE_MIN = 0.55

_CODE_SYSTEM = (
    "You code a patient's spoken answer onto EXACTLY ONE of the allowed "
    "options of a health-screening question. The transcript comes from speech "
    "recognition and may be informal.\n"
    "Question: {question}\n"
    "Options (code = number):\n{options}\n"
    "Rules:\n"
    "- Choose the single best-matching option code.\n"
    "- NEVER guess a timeframe, frequency or quantity the patient did not "
    "state. If the answer does not determine one option (e.g. no timeframe "
    "where options differ by timeframe, or an off-topic reply), it is "
    "ambiguous.\n"
    "- Output ONLY compact JSON, nothing else: "
    '{{"option": <int>, "confidence": <0..1>}} or '
    '{{"status": "AMBIGUOUS", "reason": "<short>"}}'
)


def _prematch_option(options, text: str):
    """Deterministic pre-pass: exact label match, and yes/no shortcuts for
    binary No/Yes items. Returns a code or None."""
    t = " ".join(text.strip().lower().split())
    if not t:
        return None
    for i, opt in enumerate(options):
        if t == opt.label.lower():
            return i
    labels = [o.label for o in options]
    if labels == ["No", "Yes"]:
        if _YES_RE.match(t):
            return 1
        if _NO_RE.match(t):
            return 0
    return None


def code_option(question: str, options, user_text: str) -> dict:
    """Code `user_text` onto one of `options` (tuple of instruments.Option).

    Returns {"code": int} on success or {"status": "AMBIGUOUS"} when the
    answer does not determine one option (caller clarifies — never guesses).
    Strategy: deterministic pre-match first; then a strict-JSON LLM call with
    one corrective retry; low confidence or any failure -> AMBIGUOUS."""
    pre = _prematch_option(options, user_text)
    if pre is not None:
        return {"code": pre}

    option_lines = "\n".join(f"  {i}: {o.label}" for i, o in enumerate(options))
    system = _CODE_SYSTEM.format(question=question, options=option_lines)
    messages = [{"role": "system", "content": system},
                {"role": "user", "content": user_text}]
    for attempt in range(2):
        try:
            resp = _client().chat.completions.create(
                model=config.LLM_MODEL, messages=messages, temperature=0)
            raw = (resp.choices[0].message.content or "").strip()
            start, end = raw.find("{"), raw.rfind("}")
            data = json.loads(raw[start:end + 1])
            if data.get("status") == AMBIGUOUS:
                return {"status": AMBIGUOUS}
            code = data["option"]
            confidence = float(data.get("confidence", 0))
            if (isinstance(code, int) and 0 <= code < len(options)
                    and confidence >= CODE_CONFIDENCE_MIN):
                return {"code": code}
            if isinstance(code, int) and 0 <= code < len(options):
                return {"status": AMBIGUOUS}      # valid but low confidence
            raise ValueError(f"option {code!r} out of range")
        except Exception as e:
            logger.info("code_option attempt %d failed (%s)", attempt + 1, e)
            messages.append({"role": "user", "content":
                             "Your last output was invalid. Output ONLY the "
                             "JSON object, with an in-range integer option."})
    return {"status": AMBIGUOUS}


def code_number(user_text: str, low: int = 0, high: int = 10) -> dict:
    """Code a spoken 0-10 ruler answer. Deterministic digit/word scan first;
    exactly one distinct in-range number -> that value, else AMBIGUOUS.
    (No LLM: a readiness number the user didn't clearly say must be re-asked.)"""
    t = user_text.lower()
    found = {int(m) for m in _DIGIT_RE.findall(t)}
    found |= {v for w, v in _WORD_NUMBERS.items()
              if re.search(rf"\b{w}\b", t)}
    found = {n for n in found if low <= n <= high}
    if len(found) == 1:
        return {"value": found.pop()}
    return {"status": AMBIGUOUS}


# --------------- Bounded single-utterance generation (P4) ---------------

_UTTER_SYSTEM = (
    "You are the voice of a structured SBIRT screening avatar. The clinical "
    "protocol — which question comes next, scores, risk zones, referrals — is "
    "decided by external code, NEVER by you. Produce exactly the single short "
    "utterance the INSTRUCTION asks for: one or two sentences, warm, "
    "plain-spoken, conversational, no clinical jargon, no scores or zone "
    "names, no new questions unless the instruction says to ask one. "
    "Output only the utterance text."
)


def phrase_utterance(instruction: str, history: list[dict],
                     patient: dict | None = None) -> str:
    """One bounded LLM utterance at the current protocol node (summaries,
    reflections, clarifications). Falls back to '' on failure — the caller
    treats an empty utterance as skippable, the protocol continues."""
    system = _UTTER_SYSTEM
    if patient:
        system += "\nKnown patient facts: " + json.dumps(patient, ensure_ascii=False)
    messages = ([{"role": "system", "content": system}]
                + list(history[-6:])
                + [{"role": "user", "content": f"INSTRUCTION: {instruction}"}])
    try:
        resp = _client().chat.completions.create(
            model=config.LLM_MODEL, messages=messages)
        return (resp.choices[0].message.content or "").strip()
    except Exception as e:
        logger.warning("phrase_utterance failed (%s); skipping utterance", e)
        return ""


if __name__ == "__main__":
    msgs = build_messages([{"role": "user", "content": "Hello, please briefly introduce yourself"}])
    print(f"LLM reply: {chat(msgs)}")
