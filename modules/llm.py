import json
import logging
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


def _trim_history(history: list[dict]):
    """Sliding window: keep only the most recent N messages so long sessions
    don't grow the prompt without bound (which slows first-token latency).
    Mutates `history` in place. After trimming, never let the window START on an
    assistant turn — that would orphan a reply from its user prompt."""
    max_msgs = getattr(config, "LLM_HISTORY_MAX_MESSAGES", 0)
    if max_msgs and len(history) > max_msgs:
        del history[:-max_msgs]
        if history and history[0]["role"] == "assistant":
            del history[0]


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
    "A health-screening tool asked the user: 'May I ask you some questions about your "
    "health?'. Read ONLY the user's reply and classify their intent as exactly one "
    "lowercase word:\n"
    "  yes — they clearly agree to proceed (incl. 'sure', 'ok', 'go ahead', 'no problem')\n"
    "  no — they clearly decline the screening ('no', 'not now', \"I'd rather not\")\n"
    "  unclear — anything else, OR any sign of distress/crisis/off-topic.\n"
    "Output only that one word."
)


def classify_consent(text: str) -> str:
    """Classify the user's reply to the greeting's consent question as
    'yes' | 'no' | 'unclear'. Conservative: distress/crisis/ambiguous -> 'unclear'
    so it is routed to the full counselor (with the crisis protocol), never to the
    fixed decline. Returns 'unclear' on any failure."""
    if not text or not text.strip():
        return "unclear"
    try:
        resp = _client().chat.completions.create(
            model=config.LLM_MODEL,
            messages=[{"role": "system", "content": _CONSENT_SYSTEM},
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


if __name__ == "__main__":
    msgs = build_messages([{"role": "user", "content": "Hello, please briefly introduce yourself"}])
    print(f"LLM reply: {chat(msgs)}")
