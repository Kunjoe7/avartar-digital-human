# Case-card gold-standard fixtures

Provenance: every number here was transcribed and hand-scored from the study's
authoritative source documents in `SBIRT_Reference/` **before** the scoring
engine was written. Do NOT edit these to make code pass — fix the code.

| File | Source |
|---|---|
| `alcohol_bi_case.json` | "Case Cards UH - SBIRT Client Generic Provider.pdf", p.1 "Alcohol - BI Case" |
| `drug_bi_case.json` | same PDF, p.2 "Drug - BI Case" |
| `alcohol_complete_case_3.json` | same PDF, p.3 "Alcohol - Complete Case 3" |
| `drug_complete_case_3.json` | same PDF, p.4 "Drug - Complete Case 3" |

Option codes were mapped through the official option lists:

- AUDIT: WHO manual Box 4 interview version (`AUDIT.pdf`). Codes are the
  option INDEX (0-based): items 1–8 index == score; items 9–10 indexes 0/1/2
  score 0/2/4.
- DAST-10: item wording per the case cards themselves. **Item 3 red flag**:
  this study asks "Are you unable to stop using drugs when you want to?"
  (positive scoring, Yes=1), unlike the standard Skinner DAST-10 which asks
  "always able to stop" reverse-scored (No=1). The case answers ("Yes") map to
  1 point under the study wording — equivalent under either wording.

Known ambiguities (kept, not resolved away — they are test material):

- **Alcohol BI, AUDIT item 10**: "Yeah, my (spouse, partner, friend) has told
  me once or twice that I drink too much sometimes." — no timeframe, so code 1
  (score 2, "not in the last year") and code 2 (score 4, "during the last
  year") are both admissible → total 9 or 11. Both land in Risky (8–15). The
  fixture encodes both variants; the zone assertion must hold for each. An
  NLU coder given this answer should return AMBIGUOUS (clarify), not guess.
- **Alcohol BI, AUDIT item 1**: card says "2-3 or more times a week"
  (quantity/frequency section: "2 to 3 times per week") → canonical code 3;
  code 4 also plausible for the "or more" reading. Zone stays Risky either
  way; encoded as a variant.

Zone keys (`healthy|risky|harmful|dependent`) follow the study's app dialogue
feedback sections ("AI SBIRT app dialogue for study.docx"): AUDIT 0-7 / 8-15 /
16-19 / 20-40; DAST 0 / 1-5 / 6-8 / 9-10.

Readiness values are the "Client Readiness to Change" numbers on each card.
