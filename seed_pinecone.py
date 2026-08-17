"""
Loads the already-built input/onet/business_ideas_enriched.json (463 ideas, committed to
the repo) and embeds + upserts it into your own Pinecone account. Use this the first time
you set up Pinecone for this project.

    python seed_pinecone.py

This is the fast path: only Cohere + Pinecone calls, no OpenAI, and it does not touch the
O*NET CSVs (not needed, the dataset is already built). Takes about a minute for 463 ideas.
Creates the Pinecone index if it does not exist yet.

To instead rebuild the dataset itself from scratch (new LLM enrichment, ~15-25 min, close
to 500 OpenAI calls), run build_dataset.py, not this script.

Needs COHERE_API_KEY and PINECONE_API_KEY in .env, see .env.example.
"""

import os
import json
import time

from dotenv import load_dotenv
import cohere
from pinecone import Pinecone, ServerlessSpec

from build_dataset import ENRICHED_IDEAS_PATH, PINECONE_INDEX_NAME, step_embed_and_upsert


def main():
    load_dotenv()
    co = cohere.ClientV2(api_key=os.getenv("COHERE_API_KEY"))
    pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))

    if PINECONE_INDEX_NAME not in pc.list_indexes().names():
        print(f"Creating Pinecone index '{PINECONE_INDEX_NAME}'...")
        pc.create_index(name=PINECONE_INDEX_NAME, dimension=1536, metric="cosine", spec=ServerlessSpec(cloud="aws", region="us-east-1"))
        while not pc.describe_index(PINECONE_INDEX_NAME).status["ready"]:
            time.sleep(1)
    index = pc.Index(PINECONE_INDEX_NAME)

    with open(ENRICHED_IDEAS_PATH) as f:
        ideas = json.load(f)

    step_embed_and_upsert(co, index, ideas, removed_ids=[])
    print(f"Seeded {len(ideas)} ideas into Pinecone index '{PINECONE_INDEX_NAME}'.")


if __name__ == "__main__":
    main()
