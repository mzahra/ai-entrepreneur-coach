# pipeline/ used to be a single ~930 line pipeline.py, split into one module per stage
# (parsing, extraction, personality, matching/ranking, reporting, LangGraph, full pipeline glue)
# for readability. This file re-exports everything so `import pipeline; pipeline.some_function(...)`
# keeps working exactly as before, callers do not need to know which submodule anything lives in.

from .clients import (
    COHERE_EMBED_MODEL,
    PINECONE_INDEX_NAME,
    get_cohere_client,
    get_pinecone_index,
)

from .profile_parsing import (
    LINKEDIN_SECTIONS,
    GENERIC_SECTION_KEYWORDS,
    PROFILE_LINK_PLATFORMS,
    fix_line_wrapped_hyphens,
    split_linkedin_sections,
    extract_profile_header,
    looks_like_header,
    split_generic_sections,
    canonicalize_sections,
    extract_generic_header,
    parse_profile_pdf,
)

from .profile_extraction import (
    STRUCTURED_PROFILE_SCHEMA,
    extract_structured_profile,
)

from .personality import (
    TIPI_ITEMS,
    score_tipi,
)

from .matching import (
    FIT_WEIGHTS,
    RETRIEVAL_TOP_K,
    build_profile_query_text,
    retrieve_candidate_ideas,
    lower_bound_fit,
    range_fit,
    cosine_similarity,
    compute_skill_fits,
    matched_skills,
    in_range_traits,
    compute_career_best_fit,
    rank_business_ideas,
)

from .reporting import (
    OUTPUT_REPORT_SCHEMA,
    generate_output_report,
    sanitize_for_pdf,
    export_report_pdf,
    export_report_html,
    slugify_filename_part,
    build_report_filename,
)

from .graph import (
    RecommendationState,
    CoachingState,
    build_recommendation_graph,
    build_coaching_graph,
)

from .full_pipeline import run_full_pipeline
