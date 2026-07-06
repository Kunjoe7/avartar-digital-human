"""Constrained NLU coding (P4): deterministic pre-pass, strict JSON handling,
never-guess semantics. LLM calls are faked — no network in tests."""

import json
from types import SimpleNamespace

import pytest

pytest.importorskip("openai")

from modules import llm
from modules.sbirt.instruments import AUDIT, DAST_10


def fake_client(replies):
    """A stand-in OpenAI client yielding canned completions in order."""
    replies = list(replies)
    calls = []

    def create(**kwargs):
        calls.append(kwargs)
        content = replies.pop(0)
        return SimpleNamespace(choices=[SimpleNamespace(
            message=SimpleNamespace(content=content))])

    client = SimpleNamespace(chat=SimpleNamespace(
        completions=SimpleNamespace(create=create)))
    client.calls = calls
    return client


# ---------------- deterministic pre-match (no LLM at all) ----------------

def test_prematch_exact_label():
    opts = AUDIT.items[0].options
    assert llm._prematch_option(opts, "2 to 3 times a week") == 3
    assert llm._prematch_option(opts, "  NEVER ") == 0
    assert llm._prematch_option(opts, "a couple times, I guess") is None


def test_prematch_binary_yes_no():
    opts = DAST_10.items[0].options       # (No, Yes)
    assert llm._prematch_option(opts, "yeah") == 1
    assert llm._prematch_option(opts, "Yes I have") == 1
    assert llm._prematch_option(opts, "nope") == 0
    assert llm._prematch_option(opts, "never") == 0
    assert llm._prematch_option(opts, "well sometimes") is None


def test_prematch_binary_shortcut_only_for_no_yes_items():
    # AUDIT item 9 options are timeframed (No / Yes-not-last-year / Yes-last-
    # year): a bare "yes" must NOT shortcut — the timeframe is undetermined.
    opts = AUDIT.items[8].options
    assert llm._prematch_option(opts, "yes") is None
    assert llm._prematch_option(opts, "no") == 0   # exact label match is fine


def test_code_option_uses_prematch_without_llm(monkeypatch):
    def boom():
        raise AssertionError("LLM must not be called for an exact match")
    monkeypatch.setattr(llm, "_client", boom)
    out = llm.code_option("q", AUDIT.items[0].options, "monthly or less")
    assert out == {"code": 1}


# ---------------- strict-JSON LLM coding ----------------

def test_code_option_valid_json(monkeypatch):
    monkeypatch.setattr(llm, "_client", lambda: fake_client(
        ['{"option": 2, "confidence": 0.9}']))
    out = llm.code_option("q", AUDIT.items[0].options, "a few times a month")
    assert out == {"code": 2}


def test_code_option_low_confidence_is_ambiguous(monkeypatch):
    monkeypatch.setattr(llm, "_client", lambda: fake_client(
        ['{"option": 2, "confidence": 0.3}']))
    out = llm.code_option("q", AUDIT.items[0].options, "hmm sometimes")
    assert out == {"status": "AMBIGUOUS"}


def test_code_option_explicit_ambiguous(monkeypatch):
    monkeypatch.setattr(llm, "_client", lambda: fake_client(
        ['{"status": "AMBIGUOUS", "reason": "no timeframe"}']))
    # The case-card item-10 answer: "told me once or twice ... sometimes"
    out = llm.code_option("q", AUDIT.items[9].options,
                          "my spouse has told me once or twice that I drink too much")
    assert out == {"status": "AMBIGUOUS"}


def test_code_option_retries_invalid_then_succeeds(monkeypatch):
    client = fake_client(["not json at all", '{"option": 1, "confidence": 0.8}'])
    monkeypatch.setattr(llm, "_client", lambda: client)
    out = llm.code_option("q", AUDIT.items[0].options, "monthly-ish")
    assert out == {"code": 1}
    assert len(client.calls) == 2


def test_code_option_out_of_range_twice_is_ambiguous(monkeypatch):
    monkeypatch.setattr(llm, "_client", lambda: fake_client(
        ['{"option": 9, "confidence": 0.9}', '{"option": -1, "confidence": 0.9}']))
    out = llm.code_option("q", AUDIT.items[0].options, "whatever")
    assert out == {"status": "AMBIGUOUS"}


def test_code_option_llm_error_is_ambiguous(monkeypatch):
    def create(**kwargs):
        raise RuntimeError("network down")
    client = SimpleNamespace(chat=SimpleNamespace(
        completions=SimpleNamespace(create=create)))
    monkeypatch.setattr(llm, "_client", lambda: client)
    out = llm.code_option("q", AUDIT.items[0].options, "some answer")
    assert out == {"status": "AMBIGUOUS"}, "coder failure must clarify, never guess"


# ---------------- readiness ruler (deterministic only) ----------------

@pytest.mark.parametrize("text,expected", [
    ("8", {"value": 8}),
    ("I'd say a five", {"value": 5}),
    ("probably ten", {"value": 10}),
    ("zero honestly", {"value": 0}),
    ("a 7, maybe 8", {"status": "AMBIGUOUS"}),
    ("I don't know", {"status": "AMBIGUOUS"}),
    ("twenty", {"status": "AMBIGUOUS"}),
    ("about 15", {"status": "AMBIGUOUS"}),
])
def test_code_number(text, expected):
    assert llm.code_number(text) == expected


def test_consent_question_is_parameterized(monkeypatch):
    client = fake_client(["yes"])
    monkeypatch.setattr(llm, "_client", lambda: client)
    assert llm.classify_consent("sure go ahead",
                                question="May I provide you some more information?") == "yes"
    sent = client.calls[0]["messages"][0]["content"]
    assert "May I provide you some more information?" in sent
