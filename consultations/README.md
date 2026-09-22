# Consultations

Five rounds of external-model consultation, kept as a research record: the prompts sent (`prompts/`) and the reports
received (`ai-assistant-suggestions/`, with extracted text under `_txt/`).

| Round | Question put to the models | Outcome |
|---|---|---|
| 1 | How is the deep third cumulant represented, and what is cheaper? | Candidate representations, all later measured |
| 2 | New formulations, given the first round's failures | Aggregate states, sketches, recursions - all falsified |
| 3 | Given the explicit propagator structure, is 1.2 units per source-layer reachable? | Two new proposals (source-index coding, birth-operator compression); both falsified |
| 4 | Public-code audit: has anyone published a method at this cost? | One real gain found (the spherical-input reparameterisation, now shipped); the rest confirmed the dead list |
| 5 | What exists beyond the sum-of-sources formulation? | Twenty alternative families assessed; the affordable ones were measured and closed |

Every proposal was run as a falsifier before being accepted or rejected; the verdicts are in `writeup/UNTESTED_IDEAS.md`
and in Part K of the handbook. These documents are third-party output and are not covered by this repository's licence.
