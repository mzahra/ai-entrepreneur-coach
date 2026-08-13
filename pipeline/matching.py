import json

from .clients import get_cohere_client, get_pinecone_index, COHERE_EMBED_MODEL
from .api_utils import ExternalAPIError, call_with_retries, RETRYABLE_COHERE_ERRORS

FIT_WEIGHTS = {"budget": 0.2, "time": 0.2, "trait": 0.35, "skill": 0.25}
RETRIEVAL_TOP_K = 30
LOW_CONFIDENCE_THRESHOLD = 65.0
# calibrated against real data: strong matches land ~85-90%, and budget_fit/time_fit/trait_fit are lenient
# enough that even an extreme budget/time mismatch (EUR 1, 0.5h/week) only pushed the composite to ~50%.
# 65 catches genuine budget/time/trait mismatches without false-triggering on ordinary profiles. Note this
# threshold can't detect "vague/generic skills text" specifically, skill_fit's own ~0.4-0.6 noise floor
# (see README known limitations) means it barely moves the composite regardless of match quality.


# --- Step 5/6: retrieve candidates from Pinecone, then rank by computed career best fit ---

def build_profile_query_text(structured_profile: dict, broadened: bool = False) -> str:
    # broadened drops the specific skills/experience text, which helps a sparse profile (few or
    # generic skills) surface industry-relevant ideas instead of narrowly (and weakly) skill-matched ones
    if broadened:
        return f"{structured_profile['industry']} professional."
    return (
        f"{structured_profile['industry']} professional with skills in {', '.join(structured_profile['skills'])}. "
        f"{structured_profile['experience_summary']}"
    )


def retrieve_candidate_ideas(structured_profile: dict, top_k: int = RETRIEVAL_TOP_K, broadened: bool = False) -> list:
    co = get_cohere_client()
    query_text = build_profile_query_text(structured_profile, broadened=broadened)
    embed_response = call_with_retries(
        lambda: co.embed(
            texts=[query_text],
            model=COHERE_EMBED_MODEL,
            input_type="search_query",
            embedding_types=["float"],
        ),
        service="Cohere embed (profile query)",
        retryable_errors=RETRYABLE_COHERE_ERRORS,
    )
    if not embed_response.embeddings.float_ or not embed_response.embeddings.float_[0]:
        raise ExternalAPIError("Cohere returned an empty embedding for the profile query.")
    query_embedding = embed_response.embeddings.float_[0]

    index = get_pinecone_index()
    results = call_with_retries(
        lambda: index.query(vector=query_embedding, top_k=top_k, include_metadata=True),
        service="Pinecone query",
    )
    matches = results.get("matches", [])
    if not matches:
        raise ExternalAPIError(
            "Pinecone returned no candidate business ideas for this profile. "
            "The index may be empty or unreachable, try running build_dataset.py."
        )
    return [json.loads(match["metadata"]["data"]) for match in matches]


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
    embed_response = call_with_retries(
        lambda: co.embed(
            texts=texts,
            model=COHERE_EMBED_MODEL,
            input_type="classification",
            embedding_types=["float"],
        ),
        service="Cohere embed (skill fits)",
        retryable_errors=RETRYABLE_COHERE_ERRORS,
    )
    embeddings = embed_response.embeddings.float_
    if len(embeddings) != len(texts):
        raise ExternalAPIError(
            f"Cohere returned {len(embeddings)} embeddings for {len(texts)} inputs while computing skill fits."
        )
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
    texts = user_skills + idea_skills
    embed_response = call_with_retries(
        lambda: co.embed(
            texts=texts,
            model=COHERE_EMBED_MODEL,
            input_type="classification",
            embedding_types=["float"],
        ),
        service="Cohere embed (matched skills)",
        retryable_errors=RETRYABLE_COHERE_ERRORS,
    )
    embeddings = embed_response.embeddings.float_
    if len(embeddings) != len(texts):
        raise ExternalAPIError(
            f"Cohere returned {len(embeddings)} embeddings for {len(texts)} inputs while matching skills."
        )
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


def rank_candidates(structured_profile: dict, big_five_scores: dict, budget_eur: float, time_available_hours_per_week: float, candidates: list) -> list:
    skill_fits = compute_skill_fits(structured_profile["skills"], candidates)
    return sorted(
        (compute_career_best_fit(idea, structured_profile, big_five_scores, budget_eur, time_available_hours_per_week, skill_fits[idea["id"]]) for idea in candidates),
        key=lambda r: r["career_best_fit_percentage"],
        reverse=True,
    )


def rank_business_ideas(structured_profile: dict, big_five_scores: dict, budget_eur: float, time_available_hours_per_week: float, top_n: int = 5) -> list:
    candidates = retrieve_candidate_ideas(structured_profile)
    ranked = rank_candidates(structured_profile, big_five_scores, budget_eur, time_available_hours_per_week, candidates)

    # sparse/generic profile: the specific-skills query came back weak, retry once with a broadened
    # query (industry only) instead of quietly presenting a low-confidence match as a top pick
    if not ranked or ranked[0]["career_best_fit_percentage"] < LOW_CONFIDENCE_THRESHOLD:
        candidates = retrieve_candidate_ideas(structured_profile, broadened=True)
        ranked = rank_candidates(structured_profile, big_five_scores, budget_eur, time_available_hours_per_week, candidates)

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
