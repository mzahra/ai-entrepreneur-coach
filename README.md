# AI Entrepreneur Coach

A trait based business idea recommender. Takes a LinkedIn PDF export (or a regular CV), a Big Five personality quiz (TIPI), a budget, and time available, then recommends small business ideas ranked by a computed "career best fit" percentage, each with a grounded rationale and a 90 day roadmap for the one the user picks. The user can also give written feedback and regenerate the report, without changing the underlying computed numbers.

This is the **Project 3** full build (4 days). The one day intro lab that came before it is preserved unchanged in `MVP/`. See `PROJECT_PLAN.md` for the full use case, tech stack justification, scope, and risk assessment, `stack_decision.md` for why LangGraph was chosen over n8n, `gtm_future_sprints.md` for the proposed next steps toward real users, and `DATASET.md` for how the 463 business ideas get built from O*NET.

## Tool list

| Tool | Role |
|---|---|
| OpenAI (`gpt-4o-mini`) | Structured profile extraction from the parsed PDF, strict JSON schema output, deterministic (`temperature=0`) |
| OpenAI (`gpt-5.6-luna`) | The report narrative, working style summary, per-idea rationale, and the 90 day roadmap, also strict JSON schema output. Chosen over `gpt-4o-mini` here after a side by side comparison showed noticeably more specific, better researched roadmap text, at a meaningfully higher token cost, worth it for narrative quality but not for the cheaper, more deterministic extraction step |
| Cohere (`embed-v4.0`) | Embeddings for retrieval (matching a person's profile to business ideas) and for semantic skill matching (replacing literal text matching, which does not work well for real resumes) |
| Pinecone | Vector database storing 463 business ideas derived from O*NET occupation data, queried by embedding similarity |
| LangGraph | Orchestrates the fixed step sequence as two graphs (see diagram below), see `stack_decision.md` for why this was chosen over n8n |
| Gradio | The UI |
| pypdf | Local PDF text extraction, before any LLM call |
| fpdf2 | PDF report export |

## Architecture

The pipeline is split into two LangGraph graphs, at the point where the app needs real user input (which idea to build a roadmap for), instead of one graph with an `interrupt()`.

```mermaid
flowchart TD
    subgraph "recommendation_graph"
        A[score_personality<br/>TIPI to Big Five] --> B[parse_pdf<br/>LinkedIn or generic CV]
        B --> C[extract_profile<br/>OpenAI gpt-4o-mini, structured JSON]
        C --> D[retrieve_candidates<br/>Cohere embed + Pinecone query]
        D --> E[rank_by_fit<br/>budget + time + trait + skill fit]
        E -->|top fit below 50%<br/>and not retried yet| Br[broaden_retrieval<br/>drop skills, industry-only query]
        Br --> D
        E -->|top fit ok, or already retried| Fi[finalize_ideas<br/>matched_skills + in_range_traits]
    end
    Fi -->|user picks one idea| F
    subgraph "coaching_graph"
        F[generate_report<br/>OpenAI gpt-5.6-luna, rationale + 90 day roadmap] --> G[export_pdf]
        F --> H[export_html]
    end
    F -.->|optional, up to 3 rounds| F
```

The user's written feedback (if any) loops back into `generate_report` only, it can reshape the narrative and roadmap but never the computed `career_best_fit_percentage`, `budget_range_eur`, or `time_range_hours_per_week`, those come from `pipeline/matching.py`, not from the LLM.

## Structure

- `app.py`, Gradio UI, run this to use the app
- `pipeline/`, the pipeline logic, one module per stage:
  - `clients.py`, Cohere/Pinecone singleton clients
  - `profile_parsing.py`, PDF to sections (LinkedIn export parser, plus a generic fallback for other CV formats)
  - `profile_extraction.py`, sections to structured profile (the OpenAI call)
  - `personality.py`, TIPI questions and Big Five scoring
  - `matching.py`, retrieval and the 4 part fit calculation (budget, time, trait, skill)
  - `reporting.py`, report narrative generation and PDF/HTML export
  - `graph.py`, the two LangGraph graphs described above
  - `full_pipeline.py`, a plain non-graph entry point (`run_full_pipeline`), same steps as one function call
  - `__init__.py`, re-exports everything, so `import pipeline; pipeline.some_function(...)` works without knowing which submodule it lives in
- `build_dataset.py`, builds the O*NET-derived business idea dataset end to end: filters ~490 occupations, enriches each into a business idea (one OpenAI call each), applies the QA fixes found during development (trait-anchor grounding, budget outlier recalibration, dropping ideas that don't fit a solo/low-budget business, deduping exact name collisions), then embeds and upserts everything to Pinecone. One command (`python build_dataset.py`) to rebuild the dataset from scratch. See `DATASET.md` for the full explanation of how and why. Needs the O*NET source CSVs locally in `input/onet/` (not committed, see O*NET's own site), not needed just to run the app
- `seed_pinecone.py`, the fast path for a fresh Pinecone account: reads the already-committed `business_ideas_enriched.json`, embeds and upserts it, no OpenAI calls, about a minute
- `DATASET.md`, how the 463 business ideas get built from O*NET: occupation selection, what is real data vs LLM estimate, the trait_anchors fix, and the cleanup steps
- `input/onet/business_ideas_enriched.json`, the O*NET-derived business idea dataset, 463 ideas, committed so a fresh setup does not need to regenerate it
- `input/`, sample profiles used for testing (`Profile.pdf`, a LinkedIn export, and a plain CV)
- `.env.example`, copy to `.env` and fill in your own API keys
- `output/`, generated reports land here, named `report_<user_name>_<business_idea_name>.pdf` / `.html`
- `MVP/`, the one day lab build, preserved unchanged as a reference (hand curated 8-idea list, no retrieval, no LangGraph)
- `PROJECT_PLAN.md`, full project plan (use case, tech stack, scope, risks, phases)
- `stack_decision.md`, LangGraph vs n8n, required by the project brief
- `gtm_future_sprints.md`, proposed go-to-market sprints for after Project 3, required by the project brief

## Setup

Copy `.env.example` to `.env` and fill in your own three keys, `OPENAI_API_KEY`, `COHERE_API_KEY`, `PINECONE_API_KEY` (each person running this needs their own accounts/keys, `.env` is gitignored and not shared).

```
pip install -r requirements.txt
```

## Populate Pinecone (first time only, per Pinecone account)

The Pinecone index needs to exist and be populated before `app.py` can retrieve anything. `input/onet/business_ideas_enriched.json` (463 ideas) is already committed, so a fresh Pinecone account just needs that data embedded and upserted, no need to regenerate it:

```
python seed_pinecone.py
```

Takes about a minute, only Cohere and Pinecone calls, no OpenAI cost. Creates the index if it does not exist yet.

To instead regenerate the dataset itself from scratch (new LLM enrichment over the O*NET source data, not needed just to run the app), see `DATASET.md` and run `build_dataset.py`, that path takes 15-25 minutes and close to 500 OpenAI calls.

## Run it

```
python app.py
```

Then open `http://127.0.0.1:7860`, upload a profile PDF (LinkedIn export or a regular CV), fill in the TIPI quiz plus budget and time available, and submit.

## Known limitations

See `PROJECT_PLAN.md` Risk Assessment for the full list. The two biggest:

- Budget and time ranges per business idea are still an AI estimate. The AI is guided by two real O*NET signals (`job_zone`, how much training the real job needs, and `typical_work_week_score`, how many hours/week people in that real job tend to work), but O*NET has no direct data on business startup cost or solo side-business time commitment, so those two numbers are a proxy, not a direct measurement.
- Skill/idea semantic matching (Cohere embeddings) has a natural "noise floor", two unrelated but both professional-sounding skill lists can still score a moderate similarity, not zero.
