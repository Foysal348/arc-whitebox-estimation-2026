# Write-up requirements and judging (collected 11 Oct 2026)

Sources:
- ARC forum post "Algorithmic contribution prize guidelines: How ARC judges these prizes — discretion, technical
  writeups & LLM usage" (discourse.aicrowd.com/t/18041), including the organiser reply of 6 Jul and the edit of 7 Sep;
- the LessWrong announcement "Announcing the ARC White-Box Estimation Challenge";
- the Phase 2 launch post (discourse.aicrowd.com/t/18197);
- the ARC blog "Mechanistic estimation for wide random MLPs" and "ARC progress update: Competing with sampling".

## Prize and judging
- **Algorithmic contribution prize:** Phase 1 $10,000; Phase 2 $20,000. Awarded at ARC's discretion to the method that
  most improves understanding of white-box estimation for random MLPs.
- **How ARC reads:** the highest-scoring submissions' write-ups are read first. The prize goes to the submission whose
  novel mechanistic ideas produced the largest gain over previously known methods.
- **Valued:** "mechanistic" estimation methods (ARC blog posts); clever sampling built on structural observations is
  an exception.
- **Valued less:** methods that rely heavily on sampling, fine-tuned constants, careful performance optimisation, or
  opaque LLM-optimised code. LLM-written code is acceptable if it can be deciphered.
- **The write-up "greatly increases" the chance of winning.** No length or format limit is published.

## LLM disclosure (mandatory)
- LLM use is encouraged, but the write-up must state how much LLMs contributed to the code and the write-up text.
- Explanations the authors do not fully understand must be hedged or labelled as guesswork, saying whether they were
  validated.
- **"Unhedged dubious claims" may make ARC more sceptical of the whole submission, possibly leading to it being
  skipped.**

## Submission mechanics
- Required: a PDF write-up and exactly ONE submission ID that was graded successfully in Phase 2 (ours: #334870).
- Channel: email to arc-whestbench@aicrowd.com (private) or a forum post (public; public write-ups may also be
  considered for community contribution prizes).
- Deadline: one week after Phase 2 closes, i.e. 24 Oct 2026 23:59 UTC (25 Oct, 05:59 Bangladesh time). The referenced
  submission must already be graded.

## Implications for our document
- Lead with the mechanistic contributions: spherical identity, parity theorem, the error map and channel oracles, the
  cubic-star (Hermite) feature, and the C·T channel analysis.
- Present the cost engineering and the fitted map (V55) honestly as engineering or fitted.
- Every explanatory claim must be marked measured, derived or conjecture.
- The LLM disclosure must be specific: who wrote the code (AI assistants under direction), which ideas came from
  outside AIs, and what we validated.
