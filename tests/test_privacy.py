"""Patient-data protection: PHI never reaches logs (no opt-out), and consent
decisions are recorded without any transcript."""

import json

import pytest

pytest.importorskip("openai")  # privacy -> config chain parity with the llm tests

import config
from modules.privacy import phi, phi_keys, record_consent


# ---------------- PHI redaction (always on, no opt-out) ----------------

def test_phi_hides_content():
    secret = "I drink a fifth of vodka every night and my name is Jane Doe"
    out = phi(secret)
    assert "vodka" not in out and "Jane" not in out
    assert out == f"<phi {len(secret.split())}w/{len(secret)}c>"


def test_phi_keys_hides_values():
    profile = {"age": 34, "substances": ["heroin"], "notes": "uses daily"}
    out = phi_keys(profile)
    assert "heroin" not in out and "34" not in out and "daily" not in out
    assert "age" in out and "substances" in out  # keys visible for flow debugging


def test_phi_has_no_opt_out():
    # Redaction is unconditional — there is no config flag that could turn it
    # off and leak a transcript into the logs.
    assert not hasattr(config, "LOG_PHI")
    assert "raw secret text" not in phi("raw secret text")


# ---------------- Consent audit trail ----------------

def test_record_consent_writes_versioned_entry(tmp_path, monkeypatch):
    log = tmp_path / "records" / "consent_log.jsonl"
    monkeypatch.setattr(config, "CONSENT_LOG_PATH", str(log))
    assert record_consent("sid-abc", "yes") is True
    assert record_consent("sid-abc", "no") is True

    lines = [json.loads(l) for l in log.read_text().splitlines()]
    assert [l["decision"] for l in lines] == ["yes", "no"]
    for l in lines:
        assert l["session"] == "sid-abc"
        assert "T" in l["ts"]                      # ISO timestamp
        assert len(l["greeting_sha"]) == 12        # exact-wording version anchor


def test_no_transcript_ever_in_entry(tmp_path, monkeypatch):
    log = tmp_path / "consent.jsonl"
    monkeypatch.setattr(config, "CONSENT_LOG_PATH", str(log))
    record_consent("sid-x", "yes")
    entry = json.loads(log.read_text())
    assert set(entry) == {"ts", "session", "decision", "greeting_sha"}


def test_write_failure_does_not_raise(monkeypatch):
    monkeypatch.setattr(config, "CONSENT_LOG_PATH",
                        "/proc/definitely/not/writable/consent.jsonl")
    assert record_consent("sid-x", "no") is False  # logged, not raised
