"""Patient-data protection: keep PHI out of logs, record consent, and hold
the encrypted at-rest session state.

Three related jobs live together here because they answer the same
question — "what do we do to protect the person's data?":

  • phi() / phi_keys() — everything a patient says (ASR transcripts),
    everything the counselor says back (LLM sentences), and the extracted
    patient profile are PHI. Wrap any such value in a log line with phi();
    it renders a content-free shape summary ("<phi 7w/41c>") so logs stay
    debuggable without ever leaking content. There is NO opt-out: PHI never
    reaches the logs, not even in local debugging.

  • record_consent() — append-only JSONL audit trail: one line per consent
    decision (what, when, and the exact greeting wording by content hash).
    No transcripts, no screening codes, no names — the session key is the
    browser-generated pseudonymous UUID. A write failure logs a warning and
    returns False; it must never break a conversation turn.

  • save/load/clear_session_state() (T24) — the authoritative clinical
    state (coded answers, covered units, node, consent) persisted per sid
    so a crash or reconnect RESUMES instead of restarting. The state dict
    contains PHI (open captures), so it is Fernet-encrypted at rest and
    the on-disk filename is a hash of the sid. Serialization to/from the
    dict is runtime.py's job; this module only encrypts, stores, loads.
    Every failure path degrades to "no saved state" — persistence must
    never break a turn.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import threading
from datetime import datetime, timezone

import config

logger = logging.getLogger(__name__)


# --------------- PHI-safe logging ---------------

def phi(value) -> str:
    """Render user/clinical content for a log line without leaking it."""
    s = str(value)
    return f"<phi {len(s.split())}w/{len(s)}c>"


def phi_keys(mapping) -> str:
    """Render a PHI-bearing dict for logs as its key list only."""
    try:
        return "keys=" + repr(sorted(mapping.keys()))
    except Exception:
        return "<phi mapping>"


# --------------- Consent audit trail ---------------

_lock = threading.Lock()


def _greeting_version() -> str:
    """Short content hash of the exact consent wording the user answered to."""
    return hashlib.sha256(config.GREETING_TEXT.encode("utf-8")).hexdigest()[:12]


# --------------- Encrypted at-rest session state (T24) ---------------

_fernet_obj = None
_fernet_lock = threading.Lock()


def _fernet():
    """The Fernet cipher for session state, built once. Key precedence:
    SBIRT_STATE_KEY env (production: inject from a secrets manager) →
    records/state.key (single-host fallback, generated once, mode 0600)."""
    global _fernet_obj
    if _fernet_obj is None:
        with _fernet_lock:
            if _fernet_obj is None:
                from cryptography.fernet import Fernet
                if config.SESSION_STATE_KEY:
                    key = config.SESSION_STATE_KEY.encode("ascii")
                else:
                    path = config.SESSION_STATE_KEY_PATH
                    try:
                        with open(path, "rb") as f:
                            key = f.read().strip()
                    except FileNotFoundError:
                        key = Fernet.generate_key()
                        os.makedirs(os.path.dirname(path), exist_ok=True)
                        fd = os.open(path,
                                     os.O_WRONLY | os.O_CREAT | os.O_EXCL,
                                     0o600)
                        with os.fdopen(fd, "wb") as f:
                            f.write(key)
                        logger.info("session-state key generated at %s", path)
                _fernet_obj = Fernet(key)
    return _fernet_obj


def _state_path(session_key: str) -> str:
    """On-disk location for one sid's state — the filename is a hash so the
    (client-generated) sid never appears in a directory listing."""
    name = hashlib.sha256(session_key.encode("utf-8")).hexdigest()[:24]
    return os.path.join(config.SESSION_STATE_DIR, name + ".bin")


def save_session_state(session_key: str, state: dict) -> bool:
    """Encrypt + atomically persist one session's state dict. Returns False
    on any failure (logged without content — the dict is PHI)."""
    try:
        blob = _fernet().encrypt(
            json.dumps(state, ensure_ascii=False).encode("utf-8"))
        path = _state_path(session_key)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        tmp = path + ".tmp"
        with open(tmp, "wb") as f:
            f.write(blob)
        os.replace(tmp, path)              # atomic: never a torn state file
        return True
    except Exception as e:
        logger.warning("session-state save failed (%s)", type(e).__name__)
        return False


def load_session_state(session_key: str) -> dict | None:
    """Decrypt one session's persisted state; None when there is none, or on
    ANY failure (wrong key, tampered/corrupt file) — a bad file must degrade
    to a fresh session, never crash a connection."""
    path = _state_path(session_key)
    try:
        with open(path, "rb") as f:
            blob = f.read()
    except FileNotFoundError:
        return None
    except OSError as e:
        logger.warning("session-state read failed (%s)", type(e).__name__)
        return None
    try:
        data = json.loads(_fernet().decrypt(blob).decode("utf-8"))
        return data if isinstance(data, dict) else None
    except Exception as e:
        logger.warning("session-state decrypt failed (%s) — starting fresh",
                       type(e).__name__)
        return None


def clear_session_state(session_key: str) -> None:
    """Forget a session's persisted state (explicit reset)."""
    try:
        os.remove(_state_path(session_key))
    except FileNotFoundError:
        pass
    except OSError as e:
        logger.warning("session-state clear failed (%s)", type(e).__name__)


def record_consent(session_key: str, decision: str) -> bool:
    """Append one consent decision. Returns True if written, False on failure
    (logged, never raised — a turn must not break on an audit write)."""
    entry = {
        "ts": datetime.now(timezone.utc).isoformat(),
        "session": session_key,
        "decision": decision,                    # "yes" | "no"
        "greeting_sha": _greeting_version(),
    }
    try:
        path = config.CONSENT_LOG_PATH
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with _lock, open(path, "a", encoding="utf-8") as f:
            f.write(json.dumps(entry) + "\n")
        return True
    except Exception as e:
        logger.warning("consent audit write failed (%s) — decision=%s", e, decision)
        return False
