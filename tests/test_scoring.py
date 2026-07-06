"""Deterministic scoring engine vs the case-card gold standard.

Fixtures under tests/fixtures/ were hand-scored from SBIRT_Reference/ source
documents BEFORE the engine existed (see fixtures/README.md). If one of these
fails, the code is wrong — do not edit the fixture.
"""

import itertools
import json
import os

import pytest

from modules.sbirt import instruments
from modules.sbirt.instruments import AUDIT, DAST_10, BY_KEY, PRE_SCREEN, Item

FIXTURES = os.path.join(os.path.dirname(__file__), "fixtures")


def load(name):
    with open(os.path.join(FIXTURES, name), encoding="utf-8") as f:
        return json.load(f)


CASES = [
    "alcohol_bi_case.json",
    "alcohol_complete_case_3.json",
    "drug_bi_case.json",
    "drug_complete_case_3.json",
]


def coding_matrix(fix):
    """Yield every admissible coding: canonical codes with each ambiguous /
    variant item swept across its admissible codes."""
    base = list(fix["codes"])
    variants = {int(k): v for k, v in fix["code_variants"].items()}
    if not variants:
        yield base
        return
    keys = sorted(variants)
    for combo in itertools.product(*(variants[k] for k in keys)):
        codes = list(base)
        for k, c in zip(keys, combo):
            codes[k] = c
        yield codes


@pytest.mark.parametrize("name", CASES)
def test_case_card_gold_standard(name):
    fix = load(name)
    instrument = BY_KEY[fix["instrument"]]
    for codes in coding_matrix(fix):
        responses = dict(enumerate(codes))
        a = instruments.assess(instrument, responses)
        assert a.complete, f"{name}: all items answered -> must be complete"
        assert a.score in fix["expected_scores"], \
            f"{name} codes={codes}: score {a.score} not in {fix['expected_scores']}"
        # Zone is the robust anchor: it must hold for EVERY admissible coding.
        assert a.zone == fix["expected_zone"], \
            f"{name} codes={codes}: zone {a.zone!r} != {fix['expected_zone']!r}"


def test_canonical_scores_match_plan():
    """The canonical coding reproduces the exact hand-computed totals."""
    expected = {
        "alcohol_bi_case.json": 9,
        "alcohol_complete_case_3.json": 33,
        "drug_bi_case.json": 7,
        "drug_complete_case_3.json": 7,
    }
    for name, want in expected.items():
        fix = load(name)
        instrument = BY_KEY[fix["instrument"]]
        got = instruments.total_score(instrument, dict(enumerate(fix["codes"])))
        assert got == want, f"{name}: {got} != {want}"


# ---------------- AUDIT skip rules (WHO Box 4) ----------------

def test_audit_q1_never_skips_to_q9():
    responses = {0: 0}  # Q1 = Never
    assert instruments.next_item_index(AUDIT, responses) == 8
    responses[8] = 0
    assert instruments.next_item_index(AUDIT, responses) == 9
    responses[9] = 0
    assert instruments.is_complete(AUDIT, responses)
    a = instruments.assess(AUDIT, responses)
    assert a.score == 0 and a.zone == "healthy"


def test_audit_q2_q3_zero_skips_to_q9():
    responses = {0: 1, 1: 0, 2: 0}  # drinks monthly or less; 1-2 typical; never 6+
    assert instruments.next_item_index(AUDIT, responses) == 8
    responses.update({8: 0, 9: 0})
    assert instruments.is_complete(AUDIT, responses)
    assert instruments.total_score(AUDIT, responses) == 1


def test_audit_no_skip_when_q2q3_positive():
    responses = {0: 1, 1: 1, 2: 0}
    assert instruments.next_item_index(AUDIT, responses) == 3  # Q4 next, no skip


def test_dast_has_no_skips():
    responses = {0: 0}
    assert instruments.next_item_index(DAST_10, responses) == 1
    all_no = {i: 0 for i in range(10)}
    assert instruments.is_complete(DAST_10, all_no)
    a = instruments.assess(DAST_10, all_no)
    assert a.score == 0 and a.zone == "healthy"


# ---------------- Zone boundaries (study app-dialogue bands) ----------------

@pytest.mark.parametrize("score,zone", [
    (0, "healthy"), (7, "healthy"), (8, "risky"), (15, "risky"),
    (16, "harmful"), (19, "harmful"), (20, "dependent"), (40, "dependent"),
])
def test_audit_zone_boundaries(score, zone):
    from modules.sbirt.instruments import risk_band_for
    assert risk_band_for(AUDIT, score).zone == zone


@pytest.mark.parametrize("score,zone", [
    (0, "healthy"), (1, "risky"), (5, "risky"),
    (6, "harmful"), (8, "harmful"), (9, "dependent"), (10, "dependent"),
])
def test_dast_zone_boundaries(score, zone):
    from modules.sbirt.instruments import risk_band_for
    assert risk_band_for(DAST_10, score).zone == zone


# ---------------- Validation: coding bugs must surface ----------------

def test_invalid_code_raises():
    with pytest.raises(instruments.InvalidResponse):
        instruments.option_score(AUDIT, 0, 5)
    with pytest.raises(instruments.InvalidResponse):
        instruments.option_score(AUDIT, 99, 0)
    with pytest.raises(instruments.InvalidResponse):
        instruments.total_score(DAST_10, {0: 2})


# ---------------- Structured data sanity (source-document invariants) -------

def test_audit_structure_matches_official_form():
    assert len(AUDIT.items) == 10
    for i in range(8):
        assert [o.score for o in AUDIT.items[i].options] == list(range(5))[:len(AUDIT.items[i].options)]
    for i in (8, 9):
        assert [o.score for o in AUDIT.items[i].options] == [0, 2, 4]
    assert instruments.total_score(AUDIT, {i: len(AUDIT.items[i].options) - 1
                                       for i in range(10)}) == 40


def test_dast_structure():
    assert len(DAST_10.items) == 10
    for item in DAST_10.items:
        assert isinstance(item, Item)
        assert [o.score for o in item.options] == [0, 1]
    # Study wording for item 3: positive phrasing ("unable to stop"), Yes = 1.
    assert "unable to stop" in DAST_10.items[2].text


def test_prescreen_structure():
    assert [q.key for q in PRE_SCREEN] == ["tobacco", "alcohol", "drugs"]
    for q in PRE_SCREEN:
        assert {o.score for o in q.item.options} == {0, 1}


# --------- T1: the data-model migration (skip rules as declarative data) ---------

def test_audit_skip_rules_are_declarative_data():
    """AUDIT's two WHO Box-4 skip rules live ON the instrument as data; the
    engine (_skipped_items/next_item_index) has no per-instrument branches."""
    assert len(AUDIT.skip_rules) == 2
    assert DAST_10.skip_rules == ()
    # Rule 1: item 1 'Never' skips 2-8.
    assert AUDIT.skip_rules[0].skip == tuple(range(1, 8))
    assert AUDIT.skip_rules[0].when({0: 0}) is True
    assert AUDIT.skip_rules[0].when({0: 1}) is False
    assert AUDIT.skip_rules[0].when({}) is False          # unanswered → no skip
    # Rule 2: items 2+3 totalling 0 skips 4-8 — and only when BOTH answered.
    assert AUDIT.skip_rules[1].skip == tuple(range(3, 8))
    assert AUDIT.skip_rules[1].when({1: 0, 2: 0}) is True
    assert AUDIT.skip_rules[1].when({1: 0}) is False
    assert AUDIT.skip_rules[1].when({1: 1, 2: 0}) is False


def test_item_kind_derived_from_options():
    """kind can never drift from the options it is derived from."""
    assert DAST_10.items[0].kind == "yesno"
    assert AUDIT.items[0].kind == "scale"
    assert AUDIT.items[8].kind == "scale"     # No/Yes-timeframed is NOT yesno
    assert PRE_SCREEN[0].item.kind == "yesno"


def test_stems_verbatim_by_default():
    """Conservative default pending the study's exact-wording ruling: every
    administered stem is spoken verbatim; the engine may wrap, never rephrase."""
    for ins in (AUDIT, DAST_10):
        assert all(it.verbatim for it in ins.items)
    assert all(q.item.verbatim for q in PRE_SCREEN)


def test_option_aliases_ship_empty():
    """Aliases are a clinical coding decision (PENDING CLINICIAN REVIEW) —
    the field exists as a contract but must ship empty until reviewed."""
    for ins in (AUDIT, DAST_10):
        for it in ins.items:
            assert all(o.aliases == () for o in it.options)
