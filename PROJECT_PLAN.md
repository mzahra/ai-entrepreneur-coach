# Autonomous Agent Project Plan

**Project name:** AI Entrepreneur Coach, A Trait-Based Business Idea Recommender

**Project Planning and MVP:** 2026-08-06 to 2026-08-07

**Full build:** Project 3

## 1. Use Case

**Problem statement:** A lot of people who want to start a business do not know what business fit them. They search generic "best business ideas" lists that do not care about their skills, budget, time, or risk tolerance. This agent take a real profile from the person (Big Five personality, resume/LinkedIn background, budget, time available) and recommend a small set of business ideas that match, with reason for each one, grounded in a real knowledge source instead of invented on the spot.

**Target users:** People who think about starting a business but are not sure what direction to go, career changers, students, side hustlers, first time founders.

**Verify:** Give the agent one input profile and check if the output ideas logically connect to at least two stated traits or constraints, and if each recommendation trace back to a retrieved source, not just invented text.

**Current process today:** People ask friends, browse generic listicles, or pay a business coach for one on one session. It is slow, not personalized, and depend a lot on who you ask.

## 2. Technology Stack

- **Core LLM:** OpenAI
- **RAG components:** Cohere embeddings, **Pinecone**: vector DB to store the curated business idea knowledge base
- **Agent framework:** LangGraph, chosen over a plain LangChain ReAct agent because the flow is a fixed sequence (parse, extract, retrieve, rank by career best fit, explain), not something the model should decide on its own, and LangGraph let us checkpoint each step for the Verify requirement. For this fixed flow, a plain chain of Python function calls would give the same result today, LangGraph is worth it for what it enables beyond that:
  - **Conditional branching:** a graph edge can route based on the data, for example retry retrieval with a different query if candidates come back empty, not easy to express as cleanly with plain function calls
  - **Checkpointing and resume:** a compiled graph with a checkpointer can resume from the last completed step if a run crashes mid way, instead of restarting from the beginning
  - **Human in the loop:** `interrupt()` can pause a graph mid run for real user input, right now we work around this with two separate graphs (one for ranking, one for the report), split at the point the user picks an idea
  - **Tracing and visualization:** the graph structure itself can be drawn and traced (for example with LangSmith), useful for debugging and for showing the pipeline in the presentation
- **Orchestration:** none, standalone Python app with a Gradio UI
- **Tools/integrations:** none for MVP, could add a domain name availability check as a stretch goal later


**Justification and alternatives considered:**
- Plain ReAct agent, rejected because less predictable, harder to guarantee retrieval happen before generation
- n8n, rejected because there is no external business system to integrate with
- Managed vector DB was first considered "not for today", but the user wants Project 3 to be reusable and deployable, so Pinecone is the target for the real build. A local Chroma folder does not survive redeploy, which matter if we host the app.
- Live LinkedIn scraping or API, rejected for Terms of Service and legal risk. Replaced with a self exported PDF (user clicks "More" then "Save to PDF" on their own LinkedIn profile, or uploads a CV), no scraping needed.
- Personality framework: considered MBTI, DISC, Enneagram, rejected for weak scientific validity and/or licensing issues. Chose **Big Five via TIPI** (Ten Item Personality Inventory, 10 items, validated, public), because it has real research linking it to entrepreneurial success.

**PDF parsing:** raw text is extracted with `pypdf`, then split into known LinkedIn export sections (Contact, Top Skills, Languages, Certifications, Honors-Awards, Summary, Experience, Education) before any LLM call, so parsing stays separate from reasoning and each section is inspectable on its own. This parser is written for the LinkedIn export layout specifically, Project 3 needs to add handling for other CV formats and structures.

## 3. MVP Scope

**Must-have (Project 3 full build):**
- Upload one background document (LinkedIn PDF export preferred, CV as fallback if no LinkedIn)
- Complete 10 item Big Five quiz (TIPI)
- Enter budget and time available
- Extract structured profile (skills, experience, industry) from the PDF
- Curated business idea knowledge base stored in Pinecone, retrieved by embedding similarity
- Rank and filter candidates by fit against budget, time, and trait
- Return top 3 to 5 ideas with reasoning that reference the retrieved source
- Generate the output report (see structure below)
- Simple Gradio UI to run the whole thing end to end

**Output report structure:**
- **Working style summary:** 2 to 4 sentences on key traits, generated from the Big Five scores plus resume/LinkedIn extraction
- **Ranked business ideas with fit percentage:** the percentage comes from our own fit calculation step (weighted combination of budget fit, time fit, trait fit, skill fit), not asked from the LLM as a made up number, so it stays defensible for Verify
- **Brief rationale per idea:** 1 to 2 sentences, cites the retrieved source, connects to at least two stated traits
- **90-day roadmap for the top match only:** broken into phases (days 1 to 30, 31 to 60, 61 to 90), with concrete first actions
- **Related study area / job area v2:** short section suggesting fields of study or job areas that also fit the profile, using O*NET occupation data (RIASEC codes) once that source is added in Project 3

**Should-have (v2):**
- RIASEC interest matching alongside Big Five
- Grit scale questions for commitment signal
- Save and revisit past runs
- Related study area / job area suggestions, using O*NET occupation data (RIASEC codes) as an added RAG source, shown as a small extra section in the output report alongside the business ideas

**Nice-to-have (v3+):**
- Domain name availability check tool
- Market saturation lookup
- Export recommendation as a PDF report
- Multi-language support

**Out of scope:**
- Real LinkedIn API or scraping
- Full business plan generation (financials, legal formation)
- Payment or subscription system
- Multi-user accounts and auth
- Any CRM or business system integration

**MVP:** no Pinecone setup, use a small hardcoded list of 5 to 8 business idea entries directly in code, just to prove one input moves to one output correctly. The output report structure above (working style summary, ranked ideas with computed fit percentage, rationale, 90-day roadmap for the top match) is included, since it needs no new infrastructure, just one more generation step on data we already have.

**Success metrics:**
- Each returned idea connects logically to at least two stated traits or constraints (checked against a simple rubric)
- Each recommendation is traceable to a retrieved source, not hallucinated
- PDF extraction correctly captures the person's top skills and industry (spot check)
- Fit percentage on each idea traces back to the fit calculation, not an LLM invented number
- Full run completes in a reasonable time without manual steps in between

## 4. Risk Assessment

| Category | Risk | Probability | Impact | Mitigation |
|---|---|---|---|---|
| Technical | Model accuracy/hallucination, recommendation not actually grounded in retrieved data | Medium | High | Force the explanation step to cite which retrieved idea it used, reject ungrounded output |
| Technical | PDF extraction failure on formats other than the LinkedIn export layout | Medium | Medium | Parser targets the LinkedIn section structure specifically, Project 3 needs fallback handling for other CV formats |
| Technical | API cost from PDF token usage | Low | Low | Parsing happens locally before any LLM call, only clean section text is sent to the model |
| Business | Scope creep, "entrepreneur coach" can expand endlessly into legal, finance, marketing advice | High | High | Enforce the out of scope list above strictly |
| Business | Users give sensitive personal data (real resume, real personality data) | Medium | Medium | No persistence of uploaded files beyond the session for now |
| Data | Small hardcoded or curated business idea dataset may not cover enough industries, biasing recommendations | High | Medium | Fine for today's proof of concept, Project 3 needs a bigger and more diverse dataset before Pinecone ingestion |
| Data | Big Five plus resume data is sensitive personal data | Low | High | Do not log raw PDF content |

## 5. Implementation Plan

**Phase 1: Setup and data preparation**
- build the real curated business idea dataset
- set up Pinecone index
- set up Cohere embeddings

**Phase 2: Core agent development**
- Build the LangGraph pipeline: parse, extract, combine traits, retrieve, rank by career best fit, explain
- Build the TIPI quiz and profile intake form in Gradio

**Phase 3: Integration and testing**
- Extend PDF parsing to handle CV formats beyond the LinkedIn export layout
- Test with multiple real resumes/LinkedIn exports and profiles
- Check recommendations against the success metrics rubric

**Phase 4: QA and presentation**
- QA pass: run the full pipeline end to end with several fresh test profiles, check edge cases (missing PDF sections, extreme trait scores, no skill matches), confirm Pinecone retrieval stays grounded (recommendations trace back to a real retrieved idea, not invented)
- Prepare the presentation, live demo

## 6. Success Metrics

Same as Step 3 success metrics above, plus one operational question to monitor: if this workflow ran daily, the first failure mode to watch is the LLM producing a recommendation that does not actually match a retrieved source (silent ungrounding), since this is the hardest failure to notice just by reading the output text.