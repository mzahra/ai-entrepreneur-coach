"""
Builds the O*NET-derived business idea dataset end to end: reads the local O*NET CSVs,
turns a filtered set of ~490 occupations into business ideas (one OpenAI call each),
applies the QA fixes found during development (trait-anchor grounding, budget outlier
recalibration, dropping ideas that don't fit a solo/low-budget business, deduping exact
name collisions), saves the result to input/onet/business_ideas_enriched.json, embeds
everything with Cohere, and upserts it into the Pinecone index.

This consolidates the whole Phase 1 dataset build into one script, so rebuilding the
dataset from scratch (e.g. after wiping the Pinecone index, or to regenerate with fresh
LLM estimates) is one command instead of running many steps by hand in order:

    python build_dataset.py

Note: the enrichment step is not deterministic (no temperature=0 on those calls, on
purpose, some variety in phrasing is fine), so re-running this will not reproduce the
exact same wording as before, but the same ~463 real O*NET occupations, the same
filtering/QA logic, and the same overall dataset shape.
"""

import os
import json
import time

import pandas as pd
from dotenv import load_dotenv
from openai import OpenAI
import cohere
from pinecone import Pinecone, ServerlessSpec

ONET_DIR = "input/onet"
ENRICHED_IDEAS_PATH = os.path.join(ONET_DIR, "business_ideas_enriched.json")
COHERE_EMBED_MODEL = "embed-v4.0"
PINECONE_INDEX_NAME = "entrepreneur-coach-ideas"

# the original 20, picked by hand, spanning different RIASEC profiles, avoiding heavy licensing
OCCUPATION_CODES = [
    "15-1254.00",  # Web Developers
    "27-1024.00",  # Graphic Designers
    "39-9031.00",  # Exercise Trainers and Group Fitness Instructors
    "25-3041.00",  # Tutors
    "35-2014.00",  # Cooks, Restaurant
    "39-5012.00",  # Hairdressers, Hairstylists, and Cosmetologists
    "13-2011.00",  # Accountants and Auditors
    "27-3043.00",  # Writers and Authors
    "27-1026.00",  # Merchandise Displayers and Window Trimmers
    "39-9011.00",  # Childcare Workers
    "37-2012.00",  # Maids and Housekeeping Cleaners
    "47-2031.00",  # Carpenters
    "13-1161.00",  # Market Research Analysts and Marketing Specialists
    "27-1022.00",  # Fashion Designers
    "15-1211.00",  # Computer Systems Analysts
    "27-2022.00",  # Coaches and Scouts
    "13-1071.00",  # Human Resources Specialists
    "43-6014.00",  # Secretaries and Administrative Assistants
    "15-1232.00",  # Computer User Support Specialists
    "27-1013.00",  # Fine Artists
]

# 2 ideas that came back with a clearly higher budget than the rest of the original 20 batch
RECALIBRATE_CODES = ["39-9011.00", "37-2012.00"]  # Childcare Workers, Maids and Housekeeping Cleaners

# job categories that translate reasonably to a solo, low budget business, see stack_decision.md /
# PROJECT_PLAN.md for the reasoning, same filter used to scale from 20 to ~490 candidates
KEEP_MAJOR_GROUPS = ["11", "13", "15", "17", "19", "21", "25", "27", "39", "41", "43", "47", "49"]

# ideas that passed the Job Zone/category filter above but turned out, on inspection, to need
# heavy industry credentials/equipment (mining, nuclear, petroleum, demolition) or subject matter
# that doesn't fit a solo low budget business recommender (sports betting), found during QA
EXCLUDE_IDEA_IDS = [
    "onet_167", "onet_168", "onet_169", "onet_186", "onet_289",
    "onet_436", "onet_439", "onet_441", "onet_486",
]

WORK_STYLE_TRAIT_BOOST = {
    "Innovation": ("openness", 1), "Achievement Orientation": ("conscientiousness", 1),
    "Intellectual Curiosity": ("openness", 1), "Tolerance for Ambiguity": ("openness", 1),
    "Initiative": ("extraversion", 1), "Adaptability": ("neuroticism", -1),
    "Self-Confidence": ("neuroticism", -1), "Perseverance": ("conscientiousness", 1),
    "Leadership Orientation": ("extraversion", 1), "Humility": ("agreeableness", 1),
    "Sincerity": ("agreeableness", 1), "Empathy": ("agreeableness", 1),
    "Cooperation": ("agreeableness", 1), "Optimism": ("neuroticism", -1),
    "Social Orientation": ("extraversion", 1), "Cautiousness": ("conscientiousness", 1),
    "Attention to Detail": ("conscientiousness", 1), "Dependability": ("conscientiousness", 1),
    "Integrity": ("conscientiousness", 1), "Stress Tolerance": ("neuroticism", -1),
    "Self-Control": ("conscientiousness", 1),
}

IDEA_SCHEMA = {
    "type": "object",
    "properties": {
        "name": {"type": "string"},
        "description": {"type": "string"},
        "category": {"type": "string"},
        "budget_range_eur": {"type": "array", "items": {"type": "number"}, "minItems": 2, "maxItems": 2},
        "time_range_hours_per_week": {"type": "array", "items": {"type": "number"}, "minItems": 2, "maxItems": 2},
        "risk_level": {"type": "string", "enum": ["low", "medium", "high"]},
        "skills_needed": {"type": "array", "items": {"type": "string"}},
        "ideal_traits": {
            "type": "object",
            "properties": {
                "openness": {"type": "array", "items": {"type": "number"}, "minItems": 2, "maxItems": 2},
                "conscientiousness": {"type": "array", "items": {"type": "number"}, "minItems": 2, "maxItems": 2},
                "extraversion": {"type": "array", "items": {"type": "number"}, "minItems": 2, "maxItems": 2},
                "agreeableness": {"type": "array", "items": {"type": "number"}, "minItems": 2, "maxItems": 2},
                "neuroticism": {"type": "array", "items": {"type": "number"}, "minItems": 2, "maxItems": 2},
            },
            "required": ["openness", "conscientiousness", "extraversion", "agreeableness", "neuroticism"],
            "additionalProperties": False,
        },
    },
    "required": ["name", "description", "category", "budget_range_eur", "time_range_hours_per_week", "risk_level", "skills_needed", "ideal_traits"],
    "additionalProperties": False,
}

ENRICHMENT_SYSTEM_PROMPT = (
    "Turn the given O*NET occupation into a small, low cost business idea a solo person could start, not a job posting. "
    "name: a short business idea name (e.g. 'Freelance Web Development Service', not the occupation title itself). "
    "description: 1 to 2 sentences describing the business, not the job. "
    "category: a short category label. "
    "budget_range_eur: realistic typical startup budget range in EUR for a solo person starting this small, [min, max], "
    "use the job_zone and top_skills given (more preparation/specialized software usually means a higher budget). "
    "time_range_hours_per_week: realistic typical time commitment range to run this as a side business or small operation, [min, max], "
    "use job_zone and typical_work_week_score as signals (typical_work_week_score is 1 to 3, 1 means the full time version of this "
    "job usually takes under 40h/week, 3 means usually over 40h/week, a higher score suggests this type of work tends to be more "
    "time intensive even in a smaller version). "
    "risk_level: low, medium, or high. "
    "skills_needed: 3 to 5 practical skills, simplified from the given top_skills list, plain language. "
    "ideal_traits: Big Five ranges on a 1 to 7 scale, one [min, max] range per trait. The user message gives you trait_anchors, a "
    "pre-computed numeric estimate (1 to 7) for each trait, already derived from this occupation's real top_work_styles data. "
    "You MUST center each trait's range on its trait_anchor value, roughly anchor minus 1.5 to anchor plus 1.5 (clipped to 1-7), "
    "do NOT default to a generic middle of scale range like 4-7 or 5-7 for every occupation, the anchors are already different per "
    "occupation because the real work style data is different, your range must reflect that difference. Only deviate from an anchor "
    "if the occupation's specific description or skills give a clear concrete reason to."
)


def chunked(items, size):
    # Cohere and Pinecone both cap how much can go in one request, so batch calls into chunks
    for i in range(0, len(items), size):
        yield items[i:i + size]


def compute_trait_anchors(top_work_styles):
    # deterministic, not LLM invented, this is the number the enrichment prompt must build
    # ideal_traits around, fixes ideal_traits being identical across unrelated occupations
    anchors = {t: 4.0 for t in ["openness", "conscientiousness", "extraversion", "agreeableness", "neuroticism"]}
    for style in top_work_styles:
        if style in WORK_STYLE_TRAIT_BOOST:
            trait, direction = WORK_STYLE_TRAIT_BOOST[style]
            anchors[trait] += direction * 0.6
    return {t: round(max(1.0, min(7.0, v)), 1) for t, v in anchors.items()}


# only the O*NET files whose signal actually ends up in a business idea, several other O*NET
# files exist but were explored and not used in the end (see README "Known limitations")
def load_onet_tables():
    occupation_df = pd.read_csv(os.path.join(ONET_DIR, "occupation_data.csv"))
    job_zones_df = pd.read_csv(os.path.join(ONET_DIR, "job_zones.csv"))
    interests_df = pd.read_csv(os.path.join(ONET_DIR, "career_interest_types.csv"))
    riasec_pivot = interests_df[interests_df["Scale ID"] == "OI"].pivot(
        index="O*NET-SOC Code", columns="Element Name", values="Data Value"
    )
    essential_df = pd.read_csv(os.path.join(ONET_DIR, "essential_skills.csv"))
    transferable_df = pd.read_csv(os.path.join(ONET_DIR, "transferable_skills.csv"))
    skills_df = pd.concat([essential_df, transferable_df], ignore_index=True)
    importance_df = skills_df[skills_df["Scale ID"] == "IM"]
    work_styles_df = pd.read_csv(os.path.join(ONET_DIR, "work_styles.csv"))
    work_styles_wi_df = work_styles_df[work_styles_df["Scale ID"] == "WI"]
    work_context_duration_df = pd.read_csv(os.path.join(ONET_DIR, "work_context_duration.csv"))
    return {
        "occupation_df": occupation_df,
        "job_zones_df": job_zones_df,
        "riasec_pivot": riasec_pivot,
        "importance_df": importance_df,
        "work_styles_wi_df": work_styles_wi_df,
        "work_context_duration_df": work_context_duration_df,
    }


# one row of real O*NET signal per occupation, this whole record is what the enrichment
# prompt sees, everything in it is real data except trait_anchors (computed above)
def build_occupation_record(code, tables):
    occ_row = tables["occupation_df"].loc[tables["occupation_df"]["O*NET-SOC Code"] == code].iloc[0]
    riasec = tables["riasec_pivot"].loc[code].to_dict()
    zone = tables["job_zones_df"].loc[tables["job_zones_df"]["O*NET-SOC Code"] == code, "Job Zone"].values[0]
    top_skill_rows = (
        tables["importance_df"][tables["importance_df"]["O*NET-SOC Code"] == code]
        .sort_values("Data Value", ascending=False).head(5)
    )
    top_work_style_rows = (
        tables["work_styles_wi_df"][tables["work_styles_wi_df"]["O*NET-SOC Code"] == code]
        .sort_values("Data Value", ascending=False).head(5)
    )
    duration_rows = tables["work_context_duration_df"].loc[
        (tables["work_context_duration_df"]["O*NET-SOC Code"] == code) & (tables["work_context_duration_df"]["Scale ID"] == "CT"),
        "Data Value",
    ]
    top_work_styles = top_work_style_rows["Element Name"].tolist()
    return {
        "code": code,
        "title": occ_row["Title"],
        "description": occ_row["Description"],
        "riasec": riasec,
        "job_zone": int(zone),
        "top_skills": top_skill_rows["Element Name"].tolist(),
        "top_work_styles": top_work_styles,
        "trait_anchors": compute_trait_anchors(top_work_styles),
        "typical_work_week_score": float(duration_rows.values[0]) if len(duration_rows) else None,
    }


# the only step that invents anything: business framing, budget/time estimate, and the
# personality range (must be centered on trait_anchors, enforced by the prompt, not by code)
def enrich_occupation(client, occupation_record, extra_prompt=""):
    response = client.responses.create(
        model="gpt-4o-mini",
        input=[
            {"role": "system", "content": ENRICHMENT_SYSTEM_PROMPT + extra_prompt},
            {"role": "user", "content": json.dumps(occupation_record)},
        ],
        text={"format": {"type": "json_schema", "name": "business_idea", "schema": IDEA_SCHEMA, "strict": True}},
    )
    return json.loads(response.output_text)


# same filter used to scale from the original 20 to ~490 candidates, see stack_decision.md /
# PROJECT_PLAN.md for why these groups and not others
def build_candidate_codes(tables):
    merged = tables["occupation_df"].merge(
        tables["job_zones_df"][["O*NET-SOC Code", "Job Zone"]], on="O*NET-SOC Code", how="left"
    )
    merged["major_group"] = merged["O*NET-SOC Code"].str[:2]
    filtered = merged[(merged["major_group"].isin(KEEP_MAJOR_GROUPS)) & (merged["Job Zone"] < 5)]
    filtered_codes = set(filtered["O*NET-SOC Code"])
    all_target_codes = filtered_codes | set(OCCUPATION_CODES)
    new_codes = sorted(all_target_codes - set(OCCUPATION_CODES))
    return all_target_codes, new_codes


def step_enrich_original_20(client, tables):
    print(f"[1/6] Enriching the original {len(OCCUPATION_CODES)} hand-picked occupations...")
    occupation_records = [build_occupation_record(code, tables) for code in OCCUPATION_CODES]
    enriched_ideas = []
    for i, occ in enumerate(occupation_records, start=1):
        idea = enrich_occupation(client, occ)
        idea["id"] = f"onet_{i:03d}"
        idea["source_note"] = f"Derived from O*NET occupation {occ['code']} ({occ['title']}), real RIASEC/skills/work styles data, budget/time/traits are LLM estimates"
        enriched_ideas.append(idea)
    print(f"    done, {len(enriched_ideas)} ideas")

    # these 2 came back priced well above the rest of this batch when estimated in isolation,
    # re-run with a hint showing the rest of the batch's real range so the estimate is relative
    print("    Recalibrating known budget outliers (Childcare, Housekeeping Cleaning)...")
    other_budgets = [idea["budget_range_eur"] for idea, occ in zip(enriched_ideas, occupation_records) if occ["code"] not in RECALIBRATE_CODES]
    budget_floor = min(b[0] for b in other_budgets)
    budget_ceiling = max(b[1] for b in other_budgets)
    calibration_hint = (
        f"\n\nCalibration note: this occupation is part of a batch of similar solo, low cost business ideas. "
        f"The other ideas in the batch have budget_range_eur values between EUR {budget_floor} and EUR {budget_ceiling}. "
        f"Estimate this one relative to that range, only go noticeably higher if there is a specific, concrete cost driver "
        f"(for example required certification, insurance, or specialized equipment), and if so keep it as close to that "
        f"range as the real cost driver allows, do not inflate the estimate without a concrete reason."
    )
    for idx, occ in enumerate(occupation_records):
        if occ["code"] in RECALIBRATE_CODES:
            idea = enrich_occupation(client, occ, extra_prompt=calibration_hint)
            idea["id"] = enriched_ideas[idx]["id"]
            idea["source_note"] = enriched_ideas[idx]["source_note"]
            enriched_ideas[idx] = idea
    return enriched_ideas


def step_enrich_new_codes(client, tables, new_codes, next_id_number):
    print(f"[2/6] Enriching {len(new_codes)} more occupations (this is the slow part, ~15-25 min)...")
    new_enriched_ideas = []
    for i, code in enumerate(new_codes):
        occ = build_occupation_record(code, tables)
        idea = enrich_occupation(client, occ)
        idea["id"] = f"onet_{next_id_number + i:03d}"
        idea["source_note"] = f"Derived from O*NET occupation {occ['code']} ({occ['title']}), real RIASEC/skills/work styles data, budget/time/traits are LLM estimates"
        new_enriched_ideas.append(idea)
        if (i + 1) % 25 == 0 or (i + 1) == len(new_codes):
            print(f"    {i + 1}/{len(new_codes)} done")
    return new_enriched_ideas


def step_clean_dataset(all_enriched_ideas):
    # these passed the Job Zone/category filter but turned out unsuitable on inspection, see
    # EXCLUDE_IDEA_IDS above for why each one specifically
    print("[3/6] Dropping ideas that don't fit a solo low-budget business...")
    before = len(all_enriched_ideas)
    all_enriched_ideas = [idea for idea in all_enriched_ideas if idea["id"] not in EXCLUDE_IDEA_IDS]
    print(f"    dropped {before - len(all_enriched_ideas)}, {len(all_enriched_ideas)} remain")

    # closely related O*NET occupations (e.g. two kinds of security analyst) can independently
    # land on the same business idea name, since each one is enriched with no knowledge of the others
    print("[4/6] Dropping exact duplicate idea names (keeping the first one enriched)...")
    seen_names = set()
    deduped = []
    removed_ids = []
    for idea in all_enriched_ideas:
        if idea["name"] in seen_names:
            removed_ids.append(idea["id"])
        else:
            seen_names.add(idea["name"])
            deduped.append(idea)
    print(f"    dropped {len(removed_ids)} duplicates, {len(deduped)} remain")
    return deduped, removed_ids


def step_embed_and_upsert(co, index, all_enriched_ideas, removed_ids):
    print(f"[5/6] Embedding {len(all_enriched_ideas)} ideas with Cohere and upserting to Pinecone...")

    def build_idea_embedding_text(idea):
        return (
            f"{idea['name']}. {idea['description']} "
            f"Category: {idea['category']}. "
            f"Skills needed: {', '.join(idea['skills_needed'])}."
        )

    all_idea_texts = [build_idea_embedding_text(idea) for idea in all_enriched_ideas]
    all_idea_embeddings = []
    for batch in chunked(all_idea_texts, 90):
        embed_response = co.embed(texts=batch, model=COHERE_EMBED_MODEL, input_type="search_document", embedding_types=["float"])
        all_idea_embeddings.extend(embed_response.embeddings.float_)

    all_vectors = [
        {"id": idea["id"], "values": embedding, "metadata": {"data": json.dumps(idea), "name": idea["name"], "category": idea["category"]}}
        for idea, embedding in zip(all_enriched_ideas, all_idea_embeddings)
    ]
    for batch in chunked(all_vectors, 100):
        index.upsert(vectors=batch)

    # ids dropped in step_clean_dataset never got upserted this run, but may already exist in
    # Pinecone from a previous run, delete them so the index doesn't keep stale vectors around
    if removed_ids:
        index.delete(ids=removed_ids)

    time.sleep(2)
    print("    ", index.describe_index_stats())


def step_test_retrieval(co, index):
    print("[6/6] Test retrieval...")
    test_query_text = "Data engineer with programming, data modeling, and problem solving skills, high conscientiousness and openness, moderate budget"
    embed_response = co.embed(texts=[test_query_text], model=COHERE_EMBED_MODEL, input_type="search_query", embedding_types=["float"])
    query_embedding = embed_response.embeddings.float_[0]
    results = index.query(vector=query_embedding, top_k=5, include_metadata=True)
    for match in results["matches"]:
        print(f"    {match['score']:.3f}  {match['id']}  {match['metadata']['name']}")


def main():
    load_dotenv()
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    co = cohere.ClientV2(api_key=os.getenv("COHERE_API_KEY"))
    pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))

    # safe to re-run, only creates the index the first time, dimension matches embed-v4.0's output
    if PINECONE_INDEX_NAME not in pc.list_indexes().names():
        pc.create_index(name=PINECONE_INDEX_NAME, dimension=1536, metric="cosine", spec=ServerlessSpec(cloud="aws", region="us-east-1"))
        while not pc.describe_index(PINECONE_INDEX_NAME).status["ready"]:
            time.sleep(1)
    index = pc.Index(PINECONE_INDEX_NAME)

    tables = load_onet_tables()

    enriched_ideas = step_enrich_original_20(client, tables)
    _, new_codes = build_candidate_codes(tables)
    new_enriched_ideas = step_enrich_new_codes(client, tables, new_codes, next_id_number=len(enriched_ideas) + 1)

    all_enriched_ideas = enriched_ideas + new_enriched_ideas
    all_enriched_ideas, removed_ids = step_clean_dataset(all_enriched_ideas)

    os.makedirs(ONET_DIR, exist_ok=True)
    with open(ENRICHED_IDEAS_PATH, "w") as f:
        json.dump(all_enriched_ideas, f, indent=2)
    print(f"Saved {len(all_enriched_ideas)} ideas to {ENRICHED_IDEAS_PATH}")

    step_embed_and_upsert(co, index, all_enriched_ideas, removed_ids)
    step_test_retrieval(co, index)
    print("\nDone.")


if __name__ == "__main__":
    main()
