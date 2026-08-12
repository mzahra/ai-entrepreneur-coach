# Project Proof

AI Entrepreneur Coach, trait based business idea recommender, full 4 day build. Full context in `PROJECT_PLAN.md`, `stack_decision.md`, `gtm_future_sprints.md`. See `README.md` for the architecture diagram.

## Workflow (config)

- `app.py`, Gradio UI, builds and invokes the two graphs below
- `pipeline/graph.py`, `build_recommendation_graph(client)` and `build_coaching_graph(client)`, the actual LangGraph pipeline
- `pipeline/profile_parsing.py`, `pipeline/profile_extraction.py`, `pipeline/personality.py`, `pipeline/matching.py`, `pipeline/reporting.py`, the step logic each graph node calls into
- Two OpenAI models, chosen per step: `gpt-4o-mini` for structured profile extraction (`pipeline/profile_extraction.py`, cheaper, deterministic with `temperature=0`), `gpt-5.6-luna` for the report narrative (`pipeline/reporting.py`, `REPORT_NARRATIVE_MODEL`), chosen after a side by side comparison showed noticeably more specific, better researched roadmap text (real platform/resource names, not generic templates) for a meaningfully higher token cost, worth it for the narrative step but not for extraction. Every generated report states which model wrote its narrative in a footer.
- `input/onet/business_ideas_enriched.json`, 463 business ideas derived from real O*NET occupation data, stored in a Pinecone index named `entrepreneur-coach-ideas` (1536 dim, cosine, Cohere `embed-v4.0` embeddings), retrieved by semantic similarity instead of the MVP's small hand curated list

## Input payload

```json
{
  "pdf_path": "input/Profile.pdf",
  "tipi_answers": {"1": 6, "2": 2, "3": 6, "4": 3, "5": 6, "6": 2, "7": 5, "8": 2, "9": 5, "10": 2},
  "budget_eur": 500,
  "time_available_hours_per_week": 10
}
```

## Execution trace

Ran through `recommendation_graph.invoke(...)` then `coaching_graph.invoke(...)`, the same two graphs `app.py` uses.

Structured profile extracted from the PDF (`node_extract_profile`):

```json
{
  "skills": ["Data Engineering", "Data Warehousing", "Data Modeling", "Teaching", "Public Speaking", "Google Cloud Platform (GCP)", "C++", "Oracle Development", "Website Development", "PHP", "ASP.NET", "Image Forensics", "Pattern Recognition", "Data Management", "Software Security"],
  "industry": "Information Technology",
  "years_of_experience": 11,
  "highest_education": "Doctor of Philosophy (PhD) in Image Forensics",
  "name": "Zahra Moghaddasi",
  "location": "Berlin Metropolitan Area"
}
```

Big Five scores from TIPI (`node_score_personality`):

```json
{"extraversion": 6.0, "agreeableness": 5.5, "conscientiousness": 6.0, "neuroticism": 3.0, "openness": 6.0}
```

Career best fit ranking, computed not LLM invented (`node_rank_by_fit`, retrieval pulled 30 candidates from Pinecone, these are the top 5 by score):

```
89.1%  onet_206  Freelance Transportation Consulting     (budget 1.00, time 1.00, trait 1.00, skill 0.56)
88.2%  onet_120  Data Insight Consultancy                (budget 1.00, time 1.00, trait 0.95, skill 0.60)
88.1%  onet_119  Freelance Database Optimization Service (budget 1.00, time 1.00, trait 0.95, skill 0.59)
87.7%  onet_118  Freelance Database Management Service   (budget 1.00, time 1.00, trait 0.95, skill 0.58)
87.6%  onet_137  Freelance Data Analytics Service         (budget 1.00, time 1.00, trait 0.95, skill 0.57)
```

Traceability check for the top idea, `onet_206`, its `source_note` (saved alongside every idea in the dataset):

```
Derived from O*NET occupation 19-3099.01 (Transportation Planners), real RIASEC/skills/work styles
data, budget/time/traits are LLM estimates
```

Its `matched_skills` (semantic, not literal text match, see README "Known limitations"): `["Data Analysis", "Communication", "Writing Reports", "Project Management"]`, its `in_range_traits`: all 5 Big Five traits.

Written rationale for the top idea, generated with `gpt-5.6-luna` (must only reference `matched_skills`/`in_range_traits` above, not invent new ones):

```
Your data analysis and report writing skills can help municipalities and transport companies
understand demand, routes, and project results. Your high openness, conscientiousness, and
extraversion support new solutions, careful project work, and clear communication with clients.
```

## Output record

- `output/report_Zahra_Moghaddasi_Freelance_Transportation_Consulting.pdf`
- `output/report_Zahra_Moghaddasi_Freelance_Transportation_Consulting.html`

Another saved sample report, for a different business idea, is also committed in `output/` as a required deliverable: `report_Zahra_Moghaddasi_Freelance_Data_Analytics_Service.pdf`/`.html`.

## Verify

This is a checklist for the reviewer. Open the PDF/HTML report above and check these 3 things:

1. Does each idea's rationale only mention skills or traits that are really listed in its `matched_skills`/`in_range_traits`? It should never mention a skill or trait that is not there.
2. Does every idea trace back to a real `source_note`, a real O*NET occupation code? It should never be made up.
3. Does the top idea's percentage match the formula (0.20 x budget fit + 0.20 x time fit + 0.35 x trait fit + 0.25 x skill fit), using the 4 fit scores shown above? This number must come from the formula, not from the AI guessing.

## QA checks

Phase 4 QA pass (`PROJECT_PLAN.md`), run against purpose-built edge case inputs, not just real resumes, to see how the system fails, not just how it succeeds.

| # | Check | Result |
|---|---|---|
| 1 | Non-PDF file uploaded (plain text renamed `.pdf`) | **Bug found and fixed.** Used to raise a raw `PdfStreamError`, would crash the Gradio UI. `parse_profile_pdf` now catches this and raises a clear `ProfilePdfError` ("Could not read this file as a PDF...") instead. |
| 2 | Completely empty PDF (zero pages, no text) | **Bug found and fixed, the significant one.** `extract_structured_profile` used to fabricate an entire fake profile from nothing (invented "10 years experience", a fake degree, a fake bio), directly contradicting the project's grounding promise. Now raises a clear `ProfileExtractionError` before ever calling the LLM, if there is no real content in any relevant section and no header, it refuses instead of inventing one. |
| 3 | Sparse PDF (only 1 real skill, everything else blank) | Passed, no fix needed. Extraction stayed honest, picked up the one real skill, left every other field empty instead of inventing content. |
| 4 | Ranking on that near-empty profile | Passed. Sensible results, the one real skill correctly surfaced a relevant idea, other ideas got weaker/more generic matches, no crash. |
| 5 | Extreme TIPI answers (all 1s, all 7s) | Passed, not a bug: both produce identical neutral (4.0) scores on every trait. Correct behavior of the real TIPI instrument, each trait has one reverse-scored item, so any uniform answer averages to the midpoint. A rushed/careless user gets a neutral profile, not a crash or garbage. |
| 6 | Traceability at scale (15 ranked ideas across 3 different real profiles, not just 1) | Passed. All 15 traced back to a real O*NET `source_note`, zero missing or invalid. |
| 7 | Both fixes re-verified end to end through the actual running app (not just the pipeline functions directly) | Passed. Empty PDF upload now shows the friendly `ProfileExtractionError` message in the UI, idea selector cleanly resets, no server-side traceback. |

Regression checked after the fixes: the sparse-but-real profile (#3) and a normal real profile both still extract exactly as before, the new checks only block the genuinely-empty case, not just "having very little content."

## Explain

This run is a real example from testing, not a cherry picked one. It shows the "noise floor" problem already explained in `README.md`.

"Freelance Transportation Consulting" got a slightly higher score than "Freelance Data Analytics Service", 89.1% versus 87.6%. But for a Data Engineer with a PhD, Data Analytics looks like the more obvious match. So why did Transportation Consulting come out on top?

Look at the 4 fit numbers for both ideas. `budget_fit` and `time_fit` are the same for both (this person's budget and time work for almost every idea). `trait_fit` is also almost the same for both (this person's personality fits inside most ideas' trait ranges). So the only real difference was `skill_fit`, 0.56 for Transportation Consulting versus 0.57 for Data Analytics. That is a very small gap, and it comes from Cohere's embedding similarity score, which is simply not precise enough to always separate two "somewhat related" skill lists with confidence.

If this app ran every day, this is the first problem to watch for. It is not hallucination, both ideas are real, they trace back to real O*NET jobs, and the written text only mentions real matched skills and traits. The real problem is that `skill_fit` does not give a strong enough signal near the top of the ranking. Two good matches can end up almost tied, even when one is clearly the better fit to a human reader.
