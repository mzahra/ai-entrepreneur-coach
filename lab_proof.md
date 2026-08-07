# Lab Proof

Entry point for the reviewer. AI Entrepreneur Coach, trait based business idea recommender. Full context in `PROJECT_PLAN.md`.

## Workflow (config)

- `pipeline.py`, the actual pipeline: parse PDF, extract structured profile, score TIPI, rank business ideas by computed career best fit, generate report, export PDF
- `app.py`, Gradio UI wrapping the pipeline
- `mvp-ai-entrepreneur-coach.ipynb`, same pipeline built step by step with inline output at each step
- `input/business_ideas.json`, the hand curated knowledge base used for ranking (8 entries, MVP scope, Pinecone is the target for Project 3)

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

Structured profile extracted from the PDF (Step 2 to 3):

```json
{
  "skills": ["Data Engineering", "Data Warehousing", "Data Modeling", "Google Cloud Platform (GCP)", "Mathematics", "C++", "..."],
  "industry": "Data Engineering",
  "years_of_experience": 11,
  "highest_education": "Doctor of Philosophy (PhD) in Image Forensics",
  "name": "Zahra Moghaddasi",
  "location": "Berlin Metropolitan Area"
}
```

Big Five scores from TIPI (Step 4):

```json
{"extraversion": 6.0, "agreeableness": 5.5, "conscientiousness": 6.0, "neuroticism": 3.0, "openness": 6.0}
```

Career best fit ranking, computed not LLM invented (Step 6):

```
83.8% - Freelance Data Engineering / Data Warehousing Consulting
75.0% - Online Course Creator / Technical Instructor
75.0% - Social Media Management for Local Businesses
```

## Output record

`output/entrepreneur_coach_report.pdf`, the final report: working style summary, ranked ideas with fit percentage and rationale, 90 day roadmap for the top match (split into Days 1-30, 31-60, 61-90, each a full paragraph with concrete real world platforms named).

## Verify

Open `output/entrepreneur_coach_report.pdf` and check: does each recommended idea connect to at least two of the person's stated skills or traits, does the top idea's fit percentage match what `pipeline.compute_career_best_fit()` would output for this input (traceable to a formula, not a made up number).

## Explain

First failure mode to monitor if this ran daily: the narrative rationale drifting from the data it is supposed to reference, silent ungrounding. The fit percentage itself is safe (computed, not generated), but the LLM writes the rationale text, and it could describe a skill or trait match that is not actually in `matched_skills` or `in_range_traits` for that idea. Worth adding an automated check that every rationale only mentions skills/traits present in its own grounding data.
