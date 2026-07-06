"""Runtime clinical state machine: given coded answer sequences, the protocol
must walk the correct branches deterministically (P3 acceptance)."""

import json
import os

import pytest

from modules.sbirt import runtime, templates
from modules.sbirt.instruments import BY_KEY
from modules.sbirt.runtime import ClinicalSession, LLMSay, Say

FIXTURES = os.path.join(os.path.dirname(__file__), "fixtures")


def load(name):
    with open(os.path.join(FIXTURES, name), encoding="utf-8") as f:
        return json.load(f)


def new_session():
    s = ClinicalSession()
    runtime.start(s)
    return s


def say_keys(step):
    return [u.key for u in step.utterances if isinstance(u, Say)]


def drive_prescreen(s, tobacco, alcohol, drugs):
    step = runtime.advance(s, "consent", "yes")
    assert say_keys(step) == ["prescreen.tobacco"]
    step = runtime.advance(s, "option", tobacco)
    assert say_keys(step) == ["prescreen.alcohol"]
    step = runtime.advance(s, "option", alcohol)
    assert say_keys(step) == ["prescreen.drugs"]
    return runtime.advance(s, "option", drugs)


def drive_screening(s, instrument_key, codes):
    """Answer instrument items in the order the machine asks them."""
    step = s.last_step
    while step.expect.kind == "option" and step.expect.instrument == instrument_key:
        idx = step.expect.item_index
        step = runtime.advance(s, "option", codes[idx])
    return step


def test_alcohol_bi_case_full_walk():
    fix = load("alcohol_bi_case.json")
    s = new_session()

    # Pre-screen: tobacco NO, alcohol positive, drugs none -> alcohol arm only.
    step = drive_prescreen(s, fix["prescreen"]["tobacco"],
                           fix["prescreen"]["alcohol"], fix["prescreen"]["drugs"])
    assert s.arms == [] and s.arm == "alcohol"
    assert say_keys(step) == ["alcohol.qf"] and step.expect.kind == "open"

    # Q/F exploration (open) -> education permission -> education -> AUDIT perm.
    step = runtime.advance(s, "open", "usually 2-3 glasses of wine")
    assert say_keys(step) == ["alcohol.edu.permission"]
    step = runtime.advance(s, "consent", "yes")
    assert say_keys(step) == ["alcohol.edu.standard_drink", "alcohol.edu.limits",
                              "alcohol.screen.permission"]

    # AUDIT permission -> preamble + item 0; then all 10 items in order (no skips).
    step = runtime.advance(s, "consent", "yes")
    assert say_keys(step) == ["audit.preamble", "audit.item.0"]
    asked = [0]
    while step.expect.kind == "option":
        idx = step.expect.item_index
        step = runtime.advance(s, "option", fix["codes"][idx])
        if step.expect.kind == "option":
            asked.append(step.expect.item_index)
    assert asked == list(range(10)), "AUDIT must ask all 10 items in order"

    # Deterministic assessment recorded; zone drives the branch.
    a = s.assessments["audit"]
    assert a.score in fix["expected_scores"] and a.zone == "risky"

    # Feedback permission -> risky feedback (ends with the BI ask) -> BI.
    assert say_keys(step) == ["alcohol.feedback.permission"]
    step = runtime.advance(s, "consent", "yes")
    assert say_keys(step) == ["feedback.audit.risky"]
    assert step.expect.kind == "consent"          # BI entry ask is inside the text

    step = runtime.advance(s, "consent", "yes")
    assert say_keys(step) == ["bi.likes.alcohol"]
    step = runtime.advance(s, "open", fix["likes"])
    assert say_keys(step) == ["bi.dislikes.alcohol"]
    step = runtime.advance(s, "open", fix["dislikes"])
    # Decisional-balance summary is the LLM's ONE bounded job here.
    assert isinstance(step.utterances[0], LLMSay)
    assert say_keys(step) == ["bi.recommend.alcohol", "bi.ruler.alcohol"]
    assert step.expect.kind == "number"

    step = runtime.advance(s, "number", fix["readiness"])
    assert s.readiness["alcohol"] == 8
    assert say_keys(step) == [f"bi.why_not_lower.{fix['readiness']}"]
    assert f"a {fix['readiness']} and not a 1 or 2" in step.utterances[0].text
    step = runtime.advance(s, "open", "I know I should cut back")
    assert say_keys(step) == [f"bi.why_not_higher.{fix['readiness']}"]
    step = runtime.advance(s, "open", "not sure I can on weekends")
    assert isinstance(step.utterances[0], LLMSay)
    assert say_keys(step) == ["bi.leaves_you"]
    step = runtime.advance(s, "open", "I guess I'll try cutting down")

    # Single positive arm -> close.
    assert s.node == "closed" and step.expect.kind == "end"
    assert say_keys(step)[-1] == "close"


def test_drug_bi_case_full_walk():
    fix = load("drug_bi_case.json")
    s = new_session()
    step = drive_prescreen(s, 0, 0, fix["prescreen"]["drugs"])
    assert s.arm == "drugs"
    assert say_keys(step) == ["drugs.kind"]

    step = runtime.advance(s, "open", "crack and marijuana")
    # Parameterized quantity/frequency question is LLM-phrased, bounded.
    assert len(step.utterances) == 1 and isinstance(step.utterances[0], LLMSay)
    step = runtime.advance(s, "open", "4-5 times per week")
    assert say_keys(step) == ["drugs.screen.permission"]

    step = runtime.advance(s, "consent", "yes")
    assert say_keys(step) == ["dast_10.preamble", "dast_10.item.0"]
    step = drive_screening(s, "dast_10", fix["codes"])

    a = s.assessments["dast_10"]
    assert a.score == 7 and a.zone == "harmful"
    assert say_keys(step) == ["drugs.feedback.permission"]
    step = runtime.advance(s, "consent", "yes")
    assert say_keys(step) == ["feedback.dast_10.harmful"]

    step = runtime.advance(s, "consent", "yes")       # BI entry
    step = runtime.advance(s, "open", "relax")         # likes -> dislikes
    step = runtime.advance(s, "open", "money")         # -> ruler
    step = runtime.advance(s, "number", fix["readiness"])
    assert s.readiness["drugs"] == 5
    runtime.advance(s, "open", "a")
    runtime.advance(s, "open", "b")
    step = runtime.advance(s, "open", "c")
    assert s.node == "closed"


def test_dual_positive_runs_alcohol_then_drugs():
    s = new_session()
    step = drive_prescreen(s, 0, 1, 1)
    assert s.arm == "alcohol" and s.arms == ["drugs"]
    assert say_keys(step) == ["alcohol.qf"]
    # Decline the alcohol screening -> machine moves ON to the drug arm.
    runtime.advance(s, "open", "wine")
    runtime.advance(s, "consent", "no")                # no education
    step = runtime.advance(s, "consent", "no")         # decline AUDIT
    assert s.arm == "drugs"
    assert say_keys(step) == ["permission.declined", "drugs.kind"]
    assert "alcohol.screen.permission" in s.declined


def test_audit_q1_never_skips_to_item_9():
    s = new_session()
    drive_prescreen(s, 0, 1, 0)
    runtime.advance(s, "open", "beer")
    runtime.advance(s, "consent", "no")
    step = runtime.advance(s, "consent", "yes")        # start AUDIT
    assert step.expect.item_index == 0
    step = runtime.advance(s, "option", 0)             # Q1 = Never
    assert step.expect.item_index == 8, "skip rule must jump to item 9"
    step = runtime.advance(s, "option", 0)
    assert step.expect.item_index == 9
    step = runtime.advance(s, "option", 0)
    a = s.assessments["audit"]
    assert a.score == 0 and a.zone == "healthy" and a.complete


def test_healthy_zone_skips_brief_intervention():
    s = new_session()
    drive_prescreen(s, 0, 1, 0)
    runtime.advance(s, "open", "beer")
    runtime.advance(s, "consent", "no")
    runtime.advance(s, "consent", "yes")
    runtime.advance(s, "option", 0)                    # Q1 Never -> skip
    runtime.advance(s, "option", 0)                    # item 9
    step = runtime.advance(s, "option", 0)             # item 10 -> feedback perm
    step = runtime.advance(s, "consent", "yes")
    assert say_keys(step) == ["feedback.audit.healthy", "close"]
    assert s.node == "closed", "healthy zone must NOT enter BI"


def test_all_negative_prescreen_closes():
    s = new_session()
    step = drive_prescreen(s, 0, 0, 0)
    assert s.node == "closed"
    assert say_keys(step) == ["prescreen.all_negative", "close"]


def test_consent_decline_terminates():
    s = new_session()
    step = runtime.advance(s, "consent", "no")
    assert s.node == "declined" and step.expect.kind == "end"
    assert s.consent == "no"


def test_crisis_pauses_protocol_permanently():
    s = new_session()
    drive_prescreen(s, 0, 1, 0)
    runtime.enter_crisis(s)
    assert s.crisis and s.node == "crisis"
    step = runtime.advance(s, "open", "I feel awful")
    assert isinstance(step.utterances[0], LLMSay)
    assert step.expect.kind == "open"
    # No screening resumption, ever (deliberate: human decision, not pattern's).
    step = runtime.advance(s, "open", "ok")
    assert s.node == "crisis"


def test_wrong_event_kind_raises():
    s = new_session()
    with pytest.raises(runtime.ProtocolError):
        runtime.advance(s, "number", 5)    # machine expects consent


def test_repeat_step_is_stable():
    s = new_session()
    step1 = drive_prescreen(s, 0, 1, 0)
    step2 = runtime.repeat_step(s)
    assert step1 == step2, "ambiguity must not move the machine"


def test_audit_dict_has_no_free_text():
    fix = load("drug_bi_case.json")
    s = new_session()
    drive_prescreen(s, 0, 0, 1)
    runtime.advance(s, "open", "SECRET DRUG NAME")
    runtime.advance(s, "open", "SECRET AMOUNT")
    runtime.advance(s, "consent", "yes")
    drive_screening(s, "dast_10", fix["codes"])
    d = json.dumps(s.to_audit_dict())
    assert "SECRET" not in d, "audit record must never carry transcripts"
    assert '"score": 7' in d


def test_every_fixed_key_the_machine_emits_exists_in_templates():
    # Walk all four case cards, collecting every Say the machine produced.
    # Each (key, text) must EXACTLY match the pre-warm enumeration — same key
    # must always mean same text, or the clip cache would serve stale audio.
    catalog = templates.all_fixed_utterances()
    seen = set()

    def collect(step):
        for u in step.utterances:
            if isinstance(u, Say):
                assert u.key in catalog, f"Say key not pre-warmable: {u.key}"
                assert catalog[u.key] == u.text, f"key/text drift: {u.key}"
                seen.add(u.key)

    for fixture in ("alcohol_bi_case.json", "drug_bi_case.json",
                    "alcohol_complete_case_3.json", "drug_complete_case_3.json"):
        fix = load(fixture)
        s = new_session()
        collect(drive_prescreen(s, fix["prescreen"]["tobacco"],
                                fix["prescreen"]["alcohol"],
                                fix["prescreen"]["drugs"]))
        while s.node not in ("closed", "declined"):
            kind = s.expect.kind
            if kind == "consent":
                step = runtime.advance(s, "consent", "yes")
            elif kind == "option":
                step = runtime.advance(s, "option",
                                                fix["codes"][s.expect.item_index])
            elif kind == "number":
                step = runtime.advance(s, "number", fix["readiness"])
            else:
                step = runtime.advance(s, "open", "x")
            collect(step)
    assert "close" in seen
