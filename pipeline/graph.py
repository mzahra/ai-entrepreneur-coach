from typing import TypedDict, Optional

from langgraph.graph import StateGraph, END

from .profile_parsing import parse_profile_pdf
from .profile_extraction import extract_structured_profile
from .personality import score_tipi
from .matching import retrieve_candidate_ideas, compute_skill_fits, compute_career_best_fit, matched_skills, in_range_traits
from .reporting import generate_output_report, export_report_pdf, export_report_html, build_report_filename

# --- LangGraph pipeline, wires the functions above into two graphs ---
# Split at the point the user picks which idea to get a roadmap for: the recommendation
# graph ends with the ranked list, the coaching graph starts from the user's pick.
# client is captured via closure (not put in graph state) since it is not serializable data.


class RecommendationState(TypedDict):
    pdf_path: str
    tipi_answers: dict
    budget_eur: float
    time_available_hours_per_week: float
    profile_sections: dict
    profile_header: dict
    structured_profile: dict
    big_five_scores: dict
    candidate_ideas: list
    grounded_top_ideas: list


class CoachingState(TypedDict):
    structured_profile: dict
    big_five_scores: dict
    tipi_answers: dict
    budget_eur: float
    time_available_hours_per_week: float
    grounded_top_ideas: list
    roadmap_idea_id: Optional[str]
    feedback_history: Optional[list]
    report_narrative: dict
    pdf_path: str
    html_path: str


def build_recommendation_graph(client):
    def node_score_personality(state: RecommendationState) -> dict:
        return {"big_five_scores": score_tipi(state["tipi_answers"])}

    def node_parse_pdf(state: RecommendationState) -> dict:
        profile_sections, profile_header = parse_profile_pdf(state["pdf_path"])
        return {"profile_sections": profile_sections, "profile_header": profile_header}

    def node_extract_profile(state: RecommendationState) -> dict:
        structured_profile = extract_structured_profile(client, state["profile_sections"], state["profile_header"])
        return {"structured_profile": structured_profile}

    def node_retrieve_candidates(state: RecommendationState) -> dict:
        return {"candidate_ideas": retrieve_candidate_ideas(state["structured_profile"])}

    def node_rank_by_fit(state: RecommendationState) -> dict:
        skill_fits = compute_skill_fits(state["structured_profile"]["skills"], state["candidate_ideas"])
        ranked = sorted(
            (
                compute_career_best_fit(
                    idea, state["structured_profile"], state["big_five_scores"],
                    state["budget_eur"], state["time_available_hours_per_week"], skill_fits[idea["id"]],
                )
                for idea in state["candidate_ideas"]
            ),
            key=lambda r: r["career_best_fit_percentage"],
            reverse=True,
        )
        top = ranked[:5]
        ideas_by_id = {idea["id"]: idea for idea in state["candidate_ideas"]}
        grounded_top_ideas = [
            {
                **r,
                "matched_skills": matched_skills(state["structured_profile"]["skills"], ideas_by_id[r["id"]]["skills_needed"]),
                "in_range_traits": in_range_traits(ideas_by_id[r["id"]], state["big_five_scores"]),
            }
            for r in top
        ]
        return {"grounded_top_ideas": grounded_top_ideas}

    builder = StateGraph(RecommendationState)
    builder.add_node("score_personality", node_score_personality)
    builder.add_node("parse_pdf", node_parse_pdf)
    builder.add_node("extract_profile", node_extract_profile)
    builder.add_node("retrieve_candidates", node_retrieve_candidates)
    builder.add_node("rank_by_fit", node_rank_by_fit)

    builder.set_entry_point("score_personality")
    builder.add_edge("score_personality", "parse_pdf")
    builder.add_edge("parse_pdf", "extract_profile")
    builder.add_edge("extract_profile", "retrieve_candidates")
    builder.add_edge("retrieve_candidates", "rank_by_fit")
    builder.add_edge("rank_by_fit", END)

    return builder.compile()


def build_coaching_graph(client):
    def node_generate_report(state: CoachingState) -> dict:
        report_narrative = generate_output_report(
            client, state["structured_profile"], state["big_five_scores"],
            state["budget_eur"], state["time_available_hours_per_week"],
            state["grounded_top_ideas"], state.get("roadmap_idea_id"), state.get("feedback_history"),
        )
        return {"report_narrative": report_narrative}

    def node_export_pdf(state: CoachingState) -> dict:
        roadmap_idea = next(
            (r for r in state["grounded_top_ideas"] if r["id"] == state.get("roadmap_idea_id")),
            state["grounded_top_ideas"][0],
        )
        filename = build_report_filename(state["structured_profile"]["name"], roadmap_idea["name"], "pdf")
        pdf_path = export_report_pdf(
            state["structured_profile"], state["report_narrative"], state["grounded_top_ideas"],
            f"output/{filename}", state.get("roadmap_idea_id"), state.get("tipi_answers"),
        )
        return {"pdf_path": pdf_path}

    def node_export_html(state: CoachingState) -> dict:
        roadmap_idea = next(
            (r for r in state["grounded_top_ideas"] if r["id"] == state.get("roadmap_idea_id")),
            state["grounded_top_ideas"][0],
        )
        filename = build_report_filename(state["structured_profile"]["name"], roadmap_idea["name"], "html")
        html_path = export_report_html(
            state["structured_profile"], state["report_narrative"], state["grounded_top_ideas"],
            f"output/{filename}", state.get("roadmap_idea_id"), state.get("tipi_answers"),
        )
        return {"html_path": html_path}

    builder = StateGraph(CoachingState)
    builder.add_node("generate_report", node_generate_report)
    builder.add_node("export_pdf", node_export_pdf)
    builder.add_node("export_html", node_export_html)

    builder.set_entry_point("generate_report")
    builder.add_edge("generate_report", "export_pdf")
    builder.add_edge("export_pdf", "export_html")
    builder.add_edge("export_html", END)

    return builder.compile()
