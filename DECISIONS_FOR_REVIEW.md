# Human decision points (clinical / compliance review required)

This refactor deliberately did NOT resolve the items below. They need a
clinician / study-owner decision; code references show where each lands.

## 1. Consent wording vs third-party data flow (over-promise) — HIGH

`config.GREETING_TEXT` promises: answers "treated as confidential and as
protected health information". The actual data flow sends user content to
external processors:

- ASR transcript text → OpenRouter → Google (`gemini-2.5-flash`)
  (`modules/llm.py`: consent classification, NLU option coding, bounded
  utterance phrasing, patient-fact extraction)
- Counselor reply text (incl. clinical wording) → Microsoft (edge-tts)
  (`modules/tts.py`)

Decision needed: revise the consent wording, execute BAAs / change providers,
or move ASR/LLM/TTS on-prem. Per the refactor's ground rules the greeting
text was NOT modified. (Note: fixed clips are pre-rendered once, so the fixed
script itself no longer flows to Microsoft per session — only the bounded
LLM utterances do.)

## 2. Clinical copy written during the refactor — needs sign-off

- Crisis responses (4 texts): `modules/sbirt/crisis.py::RESPONSES` —
  assembled from `referral.py` protocol lines (988/911, naloxone,
  cold-turkey warning).
- Gap-fill lines the source dialogue does not provide:
  `templates.FIXED["prescreen.all_negative"]`, `FIXED["permission.declined"]`.
- Typo normalizations of the source script (4, listed in
  `modules/sbirt/templates.py` docstring).

## 3. DAST-10 deviations inherited from the study protocol — confirm intended

- Item 3 wording flipped to "Are you unable to stop..." (positive scoring,
  Yes = 1) per the case cards; standard Skinner DAST-10 reverse-scores an
  "always able" phrasing. `modules/sbirt/instruments.py` (item note).
- Risk banding 0 / 1-5 / 6-8 / 9-10 per the app dialogue, vs Skinner's
  0 / 1-2 / 3-5 / 6-8 / 9-10.

## 4. Protocol edges the source dialogue does not specify

- Crisis pauses the protocol PERMANENTLY for the session (no auto-resume):
  `modules/sbirt/runtime.py::enter_crisis`.
- Declining a mid-protocol permission (education / screening / feedback / BI)
  skips that stage or arm and continues: `runtime.py::_on_screen_permission`
  et al., recorded in `session.declined`.
- Readiness ruler answers 0-2 still ask "why not a 1 or 2?" (kept verbatim
  from the source; sounds odd for low values).
- Dual-positive pre-screen runs the full BI per arm (alcohol arm first, then
  drugs) — repetitive for the patient; confirm intended.
- Tobacco pre-screen answer is recorded but has no follow-up instrument
  (matches the source dialogue, which defines none).

## 5. Data retention

- Consent audit trail: `records/consent_log.jsonl` (decision + timestamp +
  wording version + pseudonymous session id; no transcripts). Retention /
  encryption-at-rest policy is a deployment decision.
- Screening results (`ClinicalSession.to_audit_dict()`: codes, scores, zones)
  currently live in memory only and are NOT persisted; define the export
  path to the provider (the greeting says results are shared with them).
