# AI Entrepreneur Coach

Trait based business idea recommender. Takes a LinkedIn PDF export (or CV), a Big Five personality quiz (TIPI), a budget, and time available, then recommends business ideas ranked by a computed career best fit percentage, with a rationale per idea and a 90 day roadmap for the top match.

This is the one day lab build (2026-08-06 to 2026-08-07). The full 4.5 day build is **Project 3**, see `PROJECT_PLAN.md` for the complete plan (use case, tech stack, MVP scope, risks, phases).

## Structure

- `app.py`, Gradio UI, run this to use the app
- `pipeline.py`, the actual pipeline logic (PDF parsing, structured extraction, TIPI scoring, career best fit ranking, report generation, PDF export)
- `mvp-ai-entrepreneur-coach.ipynb`, the same pipeline built step by step with inline output at each step
- `input/business_ideas.json`, hand curated business idea knowledge base (8 entries, MVP scope, Pinecone is the target vector DB for Project 3)
- `input/Profile.pdf`, sample LinkedIn export used for testing
- `output/`, generated PDF reports land here
- `PROJECT_PLAN.md`, full project plan
- `lab_proof.md`, lab submission entry point (input, execution trace, output)

## Setup

Needs `OPENAI_API_KEY` in `.env` (already set up in this repo). Dependencies: `openai`, `python-dotenv`, `pypdf`, `fpdf2`, `gradio`.

```
pip install openai python-dotenv pypdf fpdf2 gradio
```

## Run it

```
python app.py
```

Then open `http://127.0.0.1:7860`, upload `input/Profile.pdf` (or your own LinkedIn export / CV), fill in the TIPI quiz plus budget and time available, and submit.

## Known limitations

See `PROJECT_PLAN.md` Risk Assessment for the full list. The two biggest:

- The business idea list is hand curated (8 entries) with no vector retrieval yet, Project 3 adds Pinecone and Cohere embeddings over a bigger dataset.
- PDF parsing is written specifically for LinkedIn's "Save to PDF" export layout, Project 3 needs to handle other CV formats too.
