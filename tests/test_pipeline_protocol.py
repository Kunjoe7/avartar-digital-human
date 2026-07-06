"""End-to-end protocol wiring through the REAL Pipeline (P4 acceptance).

Runs only where the heavy stack imports (the float conda env); GPU renders and
network LLM calls are stubbed, everything else — coder pre-match, state
machine, scoring, histories, video queue — is the real code path.
"""

import json
import os
import queue

import pytest

pytest.importorskip("funasr")
pytest.importorskip("torch")

import modules.pipeline as P
from modules import llm
from modules.sbirt import runtime

FIXTURES = os.path.join(os.path.dirname(__file__), "fixtures")


def load(name):
    with open(os.path.join(FIXTURES, name), encoding="utf-8") as f:
        return json.load(f)


@pytest.fixture()
def pipeline(monkeypatch):
    # No GPU / no network: renders return fake paths; bounded LLM canned.
    monkeypatch.setattr(P, "ensure_fixed_clip",
                        lambda text, path: "/fake/" + os.path.basename(path))
    monkeypatch.setattr(P.tts, "synthesize", lambda text, out, ev=None: "/fake.wav")
    monkeypatch.setattr(P.avatar, "generate_video", lambda *a, **k: "/fake.mp4")
    monkeypatch.setattr(P.llm, "phrase_utterance",
                        lambda instruction, history, patient=None: "[reflection]")
    monkeypatch.setattr(P.llm, "extract_patient_facts", lambda h: {})
    monkeypatch.setattr(P.llm, "classify_consent",
                        lambda text, question=None:
                        "yes" if text.strip().lower().startswith(("yes", "sure", "ok"))
                        else ("no" if text.strip().lower().startswith("no") else "unclear"))
    # Full-counselor path must NEVER run in a non-crisis protocol session.
    def no_full_llm(*a, **k):
        raise AssertionError("full LLM synthesis ran during a protocol turn")
    monkeypatch.setattr(P.llm, "chat_stream", no_full_llm)
    monkeypatch.setattr(P.llm, "chat", no_full_llm)
    p = P.Pipeline()
    return p


def drain(p):
    """Pop everything currently queued (up to the end-of-response marker)."""
    spoken = []
    while True:
        try:
            item = p.video_queue.get_nowait()
        except queue.Empty:
            break
        if item is None:
            break
        spoken.append(item["sentence"])
    return spoken


def say_turn(p, text):
    """Synchronous user turn via the text path; returns spoken sentences."""
    p.cancel_event.clear()
    p._turn += 1
    p._process_text(text, p._turn)
    return drain(p)


def test_full_alcohol_bi_protocol(pipeline):
    p = pipeline
    fix = load("alcohol_bi_case.json")
    p.start_greeting()
    import time
    for _ in range(50):
        if p.clinical.node == "consent" and p.chat_history:
            break
        time.sleep(0.05)
    greeting = drain(p)          # the fixed greeting clip queued by the thread
    assert greeting and greeting[0].startswith("Hello, I am an AI assistant")

    # Consent -> 3 pre-screens (exact option labels: coded WITHOUT any LLM).
    assert say_turn(p, "yes, that's fine") and p.clinical.node == "prescreen.tobacco"
    say_turn(p, "no")
    say_turn(p, "within the last year")
    spoken = say_turn(p, "none")
    assert p.clinical.node == "alcohol.qf"
    assert "What do you like to drink?" in spoken[0]

    say_turn(p, "wine mostly, two or three glasses")     # open Q/F
    spoken = say_turn(p, "sure")                          # edu permission
    assert any("12-ounce beer" in s for s in spoken), "standard-drink education"
    spoken = say_turn(p, "yes")                           # AUDIT permission
    assert any("How often do you have a drink" in s for s in spoken)

    # The ten AUDIT answers, phrased as exact labels (deterministic coding).
    answers = ["2 to 3 times a week", "3 or 4", "less than monthly", "never",
               "less than monthly", "never", "less than monthly", "never",
               "no", "yes, but not in the last year"]
    for a in answers[:-1]:
        say_turn(p, a)
    spoken = say_turn(p, answers[-1])
    a = p.clinical.assessments["audit"]
    assert a.score == 9 and a.zone == "risky", "deterministic score, zero LLM"
    assert any("feedback" in s.lower() for s in spoken)

    spoken = say_turn(p, "yes")                           # feedback permission
    assert any("using alcohol at risky levels" in s for s in spoken)
    say_turn(p, "yes")                                    # BI entry
    say_turn(p, "helps me relax")                         # likes
    spoken = say_turn(p, "spend too much")                # dislikes -> ruler
    assert any("scale from 0 to 10" in s for s in spoken)
    spoken = say_turn(p, "an 8 I guess")                  # ruler (deterministic)
    assert p.clinical.readiness["alcohol"] == 8
    assert any("a 8 and not a 1 or 2" in s for s in spoken)
    say_turn(p, "because I know it costs too much")
    say_turn(p, "weekends are hard")
    spoken = say_turn(p, "I think I'll try cutting back")
    assert p.clinical.node == "closed"
    assert any("Thank you for participating" in s for s in spoken)

    # ONE history: the API view is a derived sliding-window suffix of the
    # chat, and the deterministic score lives in session state so window
    # trimming can never corrupt triage.
    tail = [(m["role"], m["content"]) for m in p._api_window()]
    full = [(m["role"], m["content"]) for m in p.chat_history]
    assert 0 < len(tail) <= P.config.LLM_HISTORY_MAX_MESSAGES
    assert full[-len(tail):] == tail
    assert tail[0][0] == "user", "API window must never start on an assistant turn"


def test_ambiguous_answer_clarifies_and_does_not_advance(pipeline, monkeypatch):
    p = pipeline
    runtime.start(p.clinical)
    say_turn(p, "yes")            # consent
    node_before = p.clinical.node
    assert node_before == "prescreen.tobacco"
    # A fuzzy answer with the option-coder forced ambiguous:
    monkeypatch.setattr(P.llm, "code_option",
                        lambda q, o, t: {"status": "AMBIGUOUS"})
    spoken = say_turn(p, "well, you know how it is")
    assert p.clinical.node == node_before, "ambiguity must not move the machine"
    assert spoken == ["[reflection]"], "one bounded clarification is spoken"
    # Machine still accepts a clean answer afterwards.
    monkeypatch.undo()
    say_turn(p, "no")
    assert p.clinical.node == "prescreen.alcohol"


def test_crisis_phrase_mid_screen_fires_fixed_path(pipeline):
    p = pipeline
    runtime.start(p.clinical)
    say_turn(p, "yes")
    say_turn(p, "no")
    spoken = say_turn(p, "honestly I've been thinking about ending my life")
    assert p.clinical.crisis
    from modules.sbirt import crisis as crisis_mod
    assert spoken and spoken[0] == crisis_mod.RESPONSES["suicide"]


def test_consent_decline_uses_fixed_decline(pipeline, monkeypatch):
    p = pipeline
    recorded = []
    monkeypatch.setattr(P.privacy, "record_consent",
                        lambda key, decision: recorded.append((key, decision)))
    runtime.start(p.clinical)
    spoken = say_turn(p, "no thanks")
    assert p.clinical.node == "declined" and p.clinical.consent == "no"
    assert spoken and spoken[0] == P.config.DECLINE_TEXT
    assert p.ended, "decline ends the session (mic off via poller)"
    assert recorded == [("default", "no")], "decline must hit the audit trail"


def test_generation_budget_full_session(pipeline, monkeypatch):
    """P6 acceptance: with a warm clip cache, a COMPLETE screening session
    performs zero FLOAT renders for fixed content — the only runtime renders
    are the bounded LLM utterances (the dynamic face)."""
    p = pipeline
    renders = []
    monkeypatch.setattr(P.avatar, "generate_video",
                        lambda *a, **k: renders.append(1) or "/fake.mp4")
    runtime.start(p.clinical)

    say_turn(p, "yes")
    say_turn(p, "no")
    say_turn(p, "within the last year")
    say_turn(p, "none")
    say_turn(p, "wine")                                   # open
    say_turn(p, "yes")                                    # edu perm
    say_turn(p, "yes")                                    # AUDIT perm
    for a in ["2 to 3 times a week", "3 or 4", "less than monthly", "never",
              "less than monthly", "never", "less than monthly", "never",
              "no", "yes, but not in the last year"]:     # 9 -> risky -> BI runs
        say_turn(p, a)
    say_turn(p, "yes")                                    # feedback perm
    say_turn(p, "yes")                                    # BI entry
    say_turn(p, "relaxing")                               # likes
    say_turn(p, "money")                                  # dislikes -> LLM summary
    say_turn(p, "8")                                      # ruler
    say_turn(p, "reasons")                                # why lower
    say_turn(p, "reasons")                                # why higher -> LLM summary
    say_turn(p, "we'll see")                              # leaves -> LLM reflect
    assert p.clinical.node == "closed"

    # Dynamic face of this walk: 3 LLMSay utterances (decisional-balance
    # summary, ruler summary, closing reflection). Fixed content: 0 renders.
    assert len(renders) == 3, f"FLOAT ran {len(renders)}x; budget is 3"


def test_every_protocol_clip_is_prewarmed(pipeline, monkeypatch):
    """prewarm_fixed_clips must cover exactly the machine-emittable keys."""
    from modules.sbirt import templates
    warmed = []
    monkeypatch.setattr(P, "ensure_fixed_clip",
                        lambda text, path: warmed.append(path) or path)
    P.prewarm_fixed_clips()
    warmed_names = {os.path.basename(w) for w in warmed}
    for key in templates.all_fixed_utterances():
        expected = os.path.basename(P.protocol_clip_path(key))
        assert expected in warmed_names, f"not pre-warmed: {key}"
    assert os.path.basename(P.config.GREETING_VIDEO_PATH) in warmed_names
    assert "crisis_suicide.mp4" in warmed_names
