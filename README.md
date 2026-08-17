# AI Entrepreneur Coach

A trait based business idea recommender. Takes a LinkedIn PDF export (or a regular CV), a Big Five personality quiz (TIPI), a budget, and time available, then recommends small business ideas ranked by a computed "career best fit" percentage, each with a grounded rationale and a 90 day roadmap for the one the user picks.

This is the **Project 3** full build. The one day intro lab that came before it is preserved unchanged in `MVP/`. See `PROJECT_PLAN.md` (full use case, tech stack, scope, risks), `stack_decision.md` (LangGraph vs n8n, architecture diagram), `DATASET.md` (how the 463 business ideas get built from O*NET), and `gtm_future_sprints.md` (next steps toward real users).

## Requirements

- Python, `pip install -r requirements.txt`
- Three API keys, OpenAI, Cohere, Pinecone

## Setup

Copy `.env.example` to `.env` and fill in your three keys.

## Populate Pinecone (first time only, per Pinecone account)

`input/onet/business_ideas_enriched.json` (463 ideas) is already committed, so a fresh Pinecone account just needs it embedded and upserted, no need to regenerate the dataset:

```
python seed_pinecone.py
```

Takes about a minute, only Cohere and Pinecone calls, no OpenAI cost. Creates the index if it does not exist yet. Expected output:

```
[5/6] Embedding 463 ideas with Cohere and upserting to Pinecone...
     {'dimension': 1536,
 'index_fullness': 0.0,
 'metric': 'cosine',
 'namespaces': {'': {'vector_count': 463}},
 'total_vector_count': 463,
 'vector_type': 'dense'}
Seeded 463 ideas into Pinecone index 'entrepreneur-coach-ideas'.
```

## Run it

```
python app.py
```

Then open `http://127.0.0.1:7860`, upload a profile PDF (LinkedIn export or a regular CV), fill in the TIPI quiz plus budget and time available, and submit.
