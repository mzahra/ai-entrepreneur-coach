import json

from .clients import get_cohere_client, get_pinecone_index, COHERE_EMBED_MODEL

FIT_WEIGHTS = {"budget": 0.2, "time": 0.2, "trait": 0.35, "skill": 0.25}
RETRIEVAL_TOP_K = 30


# --- Step 5/6: retrieve candidates from Pinecone, then rank by computed career best fit ---

def build_profile_query_text(structured_profile: dict) -> str:
    return (
        f"{structured_profile['industry']} professional with skills in {', '.join(structured_profile['skills'])}. "
        f"{structured_profile['experience_summary']}"
    )


def retrieve_candidate_ideas(structured_profile: dict, top_k: int = RETRIEVAL_TOP_K) -> list:
    co = get_cohere_client()
    query_text = build_profile_query_text(structured_profile)
    embed_response = co.embed(
        texts=[query_text],
        model=COHERE_EMBED_MODEL,
        input_type="search_query",
        embedding_types=["float"],
    )
    query_embedding = embed_response.embeddings.float_[0]

    index = get_pinecone_index()
    results = index.query(vector=query_embedding, top_k=top_k, include_metadata=True)
    return [json.loads(match["metadata"]["data"]) for match in results["matches"]]


def lower_bound_fit(value: float, low: float, high: float) -> float:
    if low == 0 or value >= low:
        return 1.0
    return max(0.0, value / low)


def range_fit(value: float, low: float, high: float, max_scale: float = 6) -> float:
    if low <= value <= high:
        return 1.0
    distance = low - value if value < low else value - high
    return max(0.0, 1 - distance / max_scale)


def cosine_similarity(a: list, b: list) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = sum(x * x for x in a) ** 0.5
    norm_b = sum(y * y for y in b) ** 0.5
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)


def compute_skill_fits(user_skills: list, candidates: list) -> dict:
    # semantic similarity, not literal substring match, so e.g. "PHP"/"C++" register as related to "Programming"
    co = get_cohere_client()
    texts = [", ".join(user_skills)] + [", ".join(idea["skills_needed"]) for idea in candidates]
    embed_response = co.embed(
        texts=texts,
        model=COHERE_EMBED_MODEL,
        input_type="classification",
        embedding_types=["float"],
    )
    embeddings = embed_response.embeddings.float_
    user_embedding, candidate_embeddings = embeddings[0], embeddings[1:]
    return {
        idea["id"]: max(0.0, cosine_similarity(user_embedding, emb))
        for idea, emb in zip(candidates, candidate_embeddings)
    }


def matched_skills(user_skills: list, idea_skills: list, threshold: float = 0.5) -> list:
    # semantic similarity, not literal text match, so e.g. "SQL" registers as related to "Database design"
    if not user_skills or not idea_skills:
        return []
    co = get_cohere_client()
    embed_response = co.embed(
        texts=user_skills + idea_skills,
        model=COHERE_EMBED_MODEL,
        input_type="classification",
        embedding_types=["float"],
    )
    embeddings = embed_response.embeddings.float_
    user_embeddings, idea_embeddings = embeddings[:len(user_skills)], embeddings[len(user_skills):]
    return [
        idea_skill
        for idea_skill, idea_embedding in zip(idea_skills, idea_embeddings)
        if max(cosine_similarity(idea_embedding, ue) for ue in user_embeddings) >= threshold
    ]


def in_range_traits(idea: dict, big_five_scores: dict) -> list:
    return [trait for trait, bounds in idea["ideal_traits"].items() if bounds[0] <= big_five_scores[trait] <= bounds[1]]


def compute_career_best_fit(idea: dict, structured_profile: dict, big_five_scores: dict, budget_eur: float, time_available_hours_per_week: float, skill_fit_value: float) -> dict:
    b_fit = lower_bound_fit(budget_eur, *idea["budget_range_eur"])
    t_fit = lower_bound_fit(time_available_hours_per_week, *idea["time_range_hours_per_week"])
    trait_fits = [range_fit(big_five_scores[trait], *bounds) for trait, bounds in idea["ideal_traits"].items()]
    tr_fit = sum(trait_fits) / len(trait_fits)
    overall = (
        FIT_WEIGHTS["budget"] * b_fit
        + FIT_WEIGHTS["time"] * t_fit
        + FIT_WEIGHTS["trait"] * tr_fit
        + FIT_WEIGHTS["skill"] * skill_fit_value
    )
    return {
        "id": idea["id"],
        "name": idea["name"],
        "description": idea["description"],
        "budget_range_eur": idea["budget_range_eur"],
        "time_range_hours_per_week": idea["time_range_hours_per_week"],
        "budget_fit": round(b_fit, 2),
        "time_fit": round(t_fit, 2),
        "trait_fit": round(tr_fit, 2),
        "skill_fit": round(skill_fit_value, 2),
        "career_best_fit_percentage": round(overall * 100, 1),
    }


def rank_business_ideas(structured_profile: dict, big_five_scores: dict, budget_eur: float, time_available_hours_per_week: float, top_n: int = 5) -> list:
    candidates = retrieve_candidate_ideas(structured_profile)
    skill_fits = compute_skill_fits(structured_profile["skills"], candidates)
    ranked = sorted(
        (compute_career_best_fit(idea, structured_profile, big_five_scores, budget_eur, time_available_hours_per_week, skill_fits[idea["id"]]) for idea in candidates),
        key=lambda r: r["career_best_fit_percentage"],
        reverse=True,
    )
    top = ranked[:top_n]
    ideas_by_id = {idea["id"]: idea for idea in candidates}
    grounded = []
    for r in top:
        idea = ideas_by_id[r["id"]]
        grounded.append({
            **r,
            "matched_skills": matched_skills(structured_profile["skills"], idea["skills_needed"]),
            "in_range_traits": in_range_traits(idea, big_five_scores),
        })
    return grounded
