from .profile_parsing import parse_profile_pdf
from .profile_extraction import extract_structured_profile
from .personality import score_tipi
from .matching import rank_business_ideas
from .reporting import generate_output_report, export_report_pdf, export_report_html

# --- full pipeline, glues every step together ---


def run_full_pipeline(client, pdf_path: str, tipi_answers: dict, budget_eur: float, time_available_hours_per_week: float, output_pdf_path: str = "output/entrepreneur_coach_report.pdf", output_html_path: str = "output/entrepreneur_coach_report.html", roadmap_idea_id: str = None) -> dict:
    profile_sections, profile_header = parse_profile_pdf(pdf_path)
    structured_profile = extract_structured_profile(client, profile_sections, profile_header)
    big_five_scores = score_tipi(tipi_answers)
    grounded_top_ideas = rank_business_ideas(structured_profile, big_five_scores, budget_eur, time_available_hours_per_week)
    report_narrative = generate_output_report(client, structured_profile, big_five_scores, budget_eur, time_available_hours_per_week, grounded_top_ideas, roadmap_idea_id)
    pdf_path_out = export_report_pdf(structured_profile, report_narrative, grounded_top_ideas, output_pdf_path, roadmap_idea_id, tipi_answers)
    html_path_out = export_report_html(structured_profile, report_narrative, grounded_top_ideas, output_html_path, roadmap_idea_id, tipi_answers)
    return {
        "structured_profile": structured_profile,
        "big_five_scores": big_five_scores,
        "grounded_top_ideas": grounded_top_ideas,
        "report_narrative": report_narrative,
        "pdf_path": pdf_path_out,
        "html_path": html_path_out,
    }
