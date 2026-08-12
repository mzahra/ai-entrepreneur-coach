# Stack Decision: LangGraph vs n8n

**Primary stack chosen: LangGraph**

## Why LangGraph fits this problem

AI Entrepreneur Coach has a fixed sequence of steps, not something an agent should improvise on its own: parse the profile PDF, extract a structured profile, score the Big Five personality quiz, retrieve candidate business ideas from Pinecone, rank them by a computed fit score, then write the explanation and 90 day roadmap. This is predictable, code-level orchestration with real Python data structures moving between steps (a parsed profile dict, a ranked list, computed fit scores), not a workflow made of pre-built connector nodes pointing at external business systems. LangGraph lets this run as an actual typed Python graph (`StateGraph`, with `TypedDict` state) with each step (node) independently testable and inspectable, which matters both for debugging during a 4-day build and for demonstrating the pipeline structure in a presentation.

Beyond just running the fixed sequence (a plain chain of Python function calls would already do that correctly), LangGraph gives this project real headroom it does not need yet but can grow into without a rewrite: conditional branching (for example retrying retrieval with a different query if candidates come back empty), checkpointing so a crashed run could resume instead of restarting, a real `interrupt()` for human-in-the-loop pausing (right now approximated with two separate graphs, one for ranking and one for the coaching report, split at the point the user picks an idea), and tracing/visualizing the graph structure for debugging.

## Why n8n was not chosen (and is out of scope for Project 3)

n8n is strongest when a workflow's job is to move data between existing business systems (a CRM, a ticketing system, a Slack channel, a billing platform) through pre-built connector nodes, with branching logic expressed visually. This project has no such systems to integrate with, everything relevant lives inside one Python codebase (PDF parsing, OpenAI calls, Cohere embeddings, Pinecone queries, PDF/HTML report generation), and the actual "logic" is closer to a typed data pipeline than a business process automation. Building it in n8n would mean either wrapping most of this logic in Code nodes anyway (losing n8n's main advantage) or fighting the visual builder to express conditional ranking math and structured LLM output schemas that are naturally expressed in Python. n8n is not used anywhere in this project, not even as a light helper, there was no trigger/webhook use case that needed it.
