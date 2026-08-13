# Business ideas dataset

This document explains how `input/onet/business_ideas_enriched.json` gets built, and why. The script that builds it is `build_dataset.py`, one command:

```
python build_dataset.py
```

## What this dataset is

463 business ideas, one idea derived from one real O*NET occupation each. Each idea has a computed id (`onet_001`, `onet_002`, and so on), a `source_note` pointing back to the real O*NET code and title, and the actual idea fields (name, description, budget range, time range, risk level, skills needed, ideal Big Five traits).

Every idea mixes two kinds of data, and this matters for how much to trust each field:

- Real O*NET data: occupation title, description, RIASEC scores, top skills, top work styles, job zone, typical work week.
- LLM estimates on top of that real data: the business framing, the budget range, the time range, and the personality trait ranges.

The `source_note` on every idea says this openly, so it stays visible which part is real data and which part is an estimate.

## Where the occupations come from

Two sources, then combined into one list.

**1. 20 occupations picked by hand.** These come from Phase 1, picked to cover different RIASEC profiles and to avoid occupations with heavy licensing. See `OCCUPATION_CODES` in `build_dataset.py`.

**2. 472 more occupations, picked by a filter.** Same filter used to scale from 20 to the full dataset:

- Keep only 13 major SOC groups that translate reasonably to a solo, low budget business.
- Keep only Job Zone 1 to 4. Job Zone 5 needs extensive preparation (advanced degree, licensing), like a doctor or a lawyer, so it does not fit a solo low budget business.

The 13 groups kept, and the ones left out, and why:

| Kept (13 groups) | Left out (10 groups) | Why left out |
|---|---|---|
| Management | Legal | needs licensing |
| Business / Financial | Healthcare Practitioners | needs licensing |
| Computer / Math | Healthcare Support | needs licensing |
| Engineering | Protective Service | needs licensing |
| Science | Food Prep & Serving | mostly employed only roles |
| Community / Social Service | Building / Grounds Cleaning | mostly employed only roles |
| Education | Farming, Fishing, Forestry | mostly employed only roles |
| Arts / Design | Production | mostly employed only roles |
| Personal Care | Transportation | mostly employed only roles |
| Sales | Military | not applicable |
| Office / Admin | | |
| Construction | | |
| Installation / Repair | | |

In numbers: O*NET has 1016 occupations total. 632 of them are in these 13 groups. Of those 632, 97 are Job Zone 5 and get excluded, so 490 remain after the filter. 20 of those 490 are already in the hand picked list, so the filter adds 472 new ones. 20 (hand picked) + 472 (filter) = 492 occupations get enriched in total, before any cleanup.

## What the O*NET files give us

`load_onet_tables()` in `build_dataset.py` reads only the files whose signal actually ends up in a business idea:

- `occupation_data.csv`: title, description
- `job_zones.csv`: job zone (1 to 5)
- `career_interest_types.csv`: RIASEC interest scores
- `essential_skills.csv` + `transferable_skills.csv`: top skills, by importance
- `work_styles.csv`: top work styles (used for `trait_anchors`, see below)
- `work_context_duration.csv`: typical work week score

A few other O*NET files were explored during Phase 1 but not used in the end, see the README "Known limitations" section.

## What the LLM invents, and what it does not

`enrich_occupation()` sends one real occupation record to `gpt-4o-mini`, with a strict JSON schema (`IDEA_SCHEMA`), and asks it to turn the occupation into a small solo business idea, not a job posting.

One part is not left free to the LLM: `ideal_traits`, the Big Five range for the idea. Early in the build, we found that many unrelated ideas got the exact same trait range (for example, Web Developer and Carpenter both got the same range), because the LLM was inventing ranges from scratch instead of using real signal from the occupation. The fix is `compute_trait_anchors()`, a deterministic function (no LLM involved) that turns the occupation's real `top_work_styles` into a numeric anchor per trait, using `WORK_STYLE_TRAIT_BOOST`. The enrichment prompt is told to center each trait's range on this anchor, not invent one freely. This makes the trait ranges reflect real O*NET data, not just LLM guesswork.

## Cleanup after enrichment

492 occupations get enriched, then two cleanup steps run before saving:

**1. Drop ideas that do not fit, found by inspection.** 9 ideas passed the Job Zone / category filter above, but turned out unsuitable once we actually read them, for example ideas needing heavy industry equipment (mining, nuclear, petroleum, demolition) or subject matter that does not fit a solo low budget business (sports betting). These ids are listed by hand in `EXCLUDE_IDEA_IDS`, with the reason in a comment.

**2. Drop exact duplicate names.** Some closely related O*NET occupations (for example two kinds of security analyst) can independently land on the same business idea name, since each occupation gets enriched with no knowledge of the others. 20 duplicates get dropped this way, keeping the first one enriched.

492 minus 9 minus 20 leaves the final 463.

## Budget outlier recalibration

2 of the original 20 hand picked ideas (Childcare Workers, Maids and Housekeeping Cleaners) came back with a budget clearly higher than the rest of that batch, when each occupation was estimated on its own with no idea of the other 19. These 2 get re-enriched a second time, with a calibration note added to the prompt showing the real budget range of the rest of the batch, so the estimate is relative to the batch instead of an isolated guess. See `RECALIBRATE_CODES`.

## Embedding and Pinecone

Once the 463 ideas are final, each one gets turned into one embedding text (name, description, category, skills needed), embedded with Cohere (`embed-v4.0`), and upserted into the Pinecone index `entrepreneur-coach-ideas`. Any id dropped in the cleanup step above also gets deleted from Pinecone, in case it exists there from an earlier run.

## Re running the script

`input/onet/business_ideas_enriched.json` is already committed with the current 463 ideas, so `build_dataset.py` only needs to run again if the Pinecone index is empty or wiped, or to regenerate the dataset with fresh LLM estimates. The enrichment calls do not use `temperature=0` on purpose (a bit of variety in wording is fine), so re running will not reproduce the exact same wording as before, but the same real O*NET occupations, the same filtering and QA logic, and the same overall dataset shape.
