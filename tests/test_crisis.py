"""Deterministic crisis safety net: these phrases MUST trigger the fixed path
regardless of what any LLM decides — that is the whole point of the net."""

import pytest

from modules.sbirt import crisis


@pytest.mark.parametrize("text,category", [
    # Suicide / self-harm — including mid-sentence, mixed case, ASR-ish text
    ("I've been thinking about suicide", "suicide"),
    ("sometimes i just want to kill myself", "suicide"),
    ("I have thought about ending my life", "suicide"),
    ("honestly I want to die", "suicide"),
    ("i keep hurting myself when i drink", "suicide"),
    ("I've been cutting myself", "suicide"),
    ("everyone would be better off dead without me around", "suicide"),
    ("I don't want to be alive anymore", "suicide"),
    ("Suicidal thoughts, yeah, sometimes", "suicide"),
    # Explicit denial still triggers — over-trigger by design, never miss.
    ("I'm not suicidal or anything", "suicide"),
    # Overdose
    ("I think I overdosed last night", "overdose"),
    ("my friend took too many pills", "overdose"),
    ("she took a whole bottle of them", "overdose"),
    ("he's not breathing right now", "overdose"),
    # Acute withdrawal danger
    ("I had a seizure when I stopped drinking", "withdrawal"),
    ("they said it was delirium tremens", "withdrawal"),
    ("i start hallucinating when i quit", "withdrawal"),
    ("I get the DTs bad", "withdrawal"),
    # Danger to others / immediate danger
    ("I feel like hurting someone", "acute_danger"),
    ("i want to kill him", "acute_danger"),
    ("he passed out and won't wake up", "acute_danger"),
])
def test_crisis_phrases_trigger(text, category):
    hit = crisis.detect(text)
    assert hit is not None, f"MISSED crisis phrase: {text!r}"
    assert hit.category == category


@pytest.mark.parametrize("text", [
    "",
    "   ",
    "I drink to relax after work",
    "maybe two or three times a week",
    "my back hurts after work",                       # 'hurt' but not self-harm
    "I quit smoking last year",
    "yes I have felt guilty about my drug use",       # DAST item 5 answer
    # DAST item 9 echo: 'withdrawal symptoms' in a normal screening answer must
    # NOT hard-fire the crisis path (see scope note in crisis.py).
    "yeah I've had withdrawal symptoms when I stopped",
    "no never had blackouts",
    "I was dead tired yesterday",
    "this job is killing me",                          # idiom, no myself/target
])
def test_normal_screening_answers_do_not_trigger(text):
    assert crisis.detect(text) is None, f"FALSE POSITIVE on: {text!r}"


def test_every_category_has_fixed_response():
    categories = {c for c, _ in crisis._PATTERNS}
    assert set(crisis.RESPONSES) == categories
    for cat, text in crisis.RESPONSES.items():
        assert len(text) > 50, f"{cat}: response text suspiciously short"
    # Safety copy must point at the right lines (voice-spelled numbers).
    assert "9 8 8" in crisis.RESPONSES["suicide"]
    assert "9 1 1" in crisis.RESPONSES["overdose"]
    assert "9 1 1" in crisis.RESPONSES["acute_danger"]
    # Cold-turkey danger: withdrawal copy must warn against abrupt cessation.
    assert "cold turkey" in crisis.RESPONSES["withdrawal"]


def test_first_hit_is_most_severe():
    # Suicide outranks withdrawal when both appear in one utterance.
    hit = crisis.detect("after the seizure I wanted to kill myself")
    assert hit.category == "suicide"


def test_detection_is_pure_and_fast():
    import time
    t0 = time.perf_counter()
    for _ in range(1000):
        crisis.detect("I usually have two or three beers with friends on weekends")
    per_call_ms = (time.perf_counter() - t0)
    assert per_call_ms < 1.0, "1000 negative scans should take well under 1s"
