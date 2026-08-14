import os

import gradio as gr
from dotenv import load_dotenv
from openai import OpenAI

import pipeline

load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

recommendation_graph = pipeline.build_recommendation_graph(client)
coaching_graph = pipeline.build_coaching_graph(client)

TURQUOISE = gr.themes.Color(
    name="turquoise",
    c50="#f2fdfc", c100="#e0faf7", c200="#bdf4ef", c300="#91ede4", c400="#65e6d9",
    c500="#3adfce", c600="#20c5b5", c700="#1a9e91", c800="#147b71", c900="#0e5850", c950="#093530",
)


def rank_ideas(pdf_file, budget_eur, time_available_hours_per_week, *tipi_ratings):
    if pdf_file is None:
        return "Please upload a profile PDF first.", gr.update(choices=[], value=None), None

    gr.Info("Reading your profile and ranking business ideas, this takes about 10 to 15 seconds...")

    tipi_answers = {item["id"]: int(rating) for item, rating in zip(pipeline.TIPI_ITEMS, tipi_ratings)}

    try:
        recommendation_result = recommendation_graph.invoke({
            "pdf_path": pdf_file,
            "tipi_answers": tipi_answers,
            "budget_eur": budget_eur,
            "time_available_hours_per_week": time_available_hours_per_week,
        })
    except (pipeline.ProfilePdfError, pipeline.ProfileExtractionError, pipeline.ExternalAPIError) as e:
        return str(e), gr.update(choices=[], value=None), None

    structured_profile = recommendation_result["structured_profile"]
    big_five_scores = recommendation_result["big_five_scores"]
    grounded_top_ideas = recommendation_result["grounded_top_ideas"]

    lines = ["## Ranked business ideas\n"]
    if recommendation_result.get("low_confidence_match"):
        lines.append(
            "> ⚠️ Your profile had limited skills/experience detail, so even after broadening the "
            "search these are the closest matches we could find, not confident top picks. Consider "
            "adding more detail to your CV/LinkedIn export for a stronger match.\n"
        )
    for r in grounded_top_ideas:
        lines.append(f"**{r['name']}**, career best fit {r['career_best_fit_percentage']}%")
        lines.append(r["description"] + "\n")

    idea_choices = [(f"{r['name']} ({r['career_best_fit_percentage']}%)", r["id"]) for r in grounded_top_ideas]

    state = {
        "structured_profile": structured_profile,
        "big_five_scores": big_five_scores,
        "tipi_answers": tipi_answers,
        "grounded_top_ideas": grounded_top_ideas,
        "budget_eur": budget_eur,
        "time_available_hours_per_week": time_available_hours_per_week,
    }

    return "\n".join(lines), gr.update(choices=idea_choices, value=idea_choices[0][1]), state


def build_report_markdown(structured_profile, tipi_answers, grounded_top_ideas, report_narrative, roadmap_idea):
    lines = [f"# AI Entrepreneur Coach report for {structured_profile['name']}\n"]
    lines.append("## Working style summary\n")
    lines.append(report_narrative["working_style_summary"] + "\n")

    lines.append("## Personality quiz (TIPI) answers\n")
    lines.append("| I see myself as... | Trait | Your rating (1-7) |")
    lines.append("|---|---|---|")
    for item in pipeline.TIPI_ITEMS:
        lines.append(f"| {item['text']} | {item['trait'].capitalize()} | {tipi_answers[item['id']]} |")
    lines.append("")

    lines.append("## Ranked business ideas\n")
    lines.append("| Idea | Fit % | Budget (EUR) | Time (h/wk) |")
    lines.append("|---|---|---|---|")
    for r in grounded_top_ideas:
        budget = f"{r['budget_range_eur'][0]:.0f}-{r['budget_range_eur'][1]:.0f}"
        time_range = f"{r['time_range_hours_per_week'][0]:.0f}-{r['time_range_hours_per_week'][1]:.0f}"
        lines.append(f"| {r['name']} | {r['career_best_fit_percentage']}% | {budget} | {time_range} |")
    lines.append("")

    budget_range = f"€{roadmap_idea['budget_range_eur'][0]:.0f}-{roadmap_idea['budget_range_eur'][1]:.0f}"
    time_range = f"{roadmap_idea['time_range_hours_per_week'][0]:.0f}-{roadmap_idea['time_range_hours_per_week'][1]:.0f} hours/week"
    lines.append(f"## 90 day roadmap: {roadmap_idea['name']}\n")
    lines.append(roadmap_idea["description"] + "\n")
    lines.append(f"**Budget:** {budget_range}  |  **Time:** {time_range}\n")

    for key, label in [("days_1_30", "Days 1-30"), ("days_31_60", "Days 31-60"), ("days_61_90", "Days 61-90")]:
        phase = report_narrative["roadmap_90_day"][key]
        lines.append(f"### {label}")
        lines.append(phase["summary"] + "\n")
        lines.append("| Action | Hours | Cost (EUR) |")
        lines.append("|---|---|---|")
        total_hours = 0.0
        total_cost = 0.0
        for item in phase["action_items"]:
            lines.append(f"| {item['action']} | {item['estimated_hours']:.1f} | {item['estimated_cost_eur']:.0f} |")
            total_hours += item["estimated_hours"]
            total_cost += item["estimated_cost_eur"]
        lines.append(f"| **Phase total** | **{total_hours:.1f}** | **{total_cost:.0f}** |")
        lines.append("")

    return "\n".join(lines)


def generate_coaching_report(selected_idea_id, state):
    if state is None:
        return "Please rank your ideas first.", None, None, None

    gr.Info("Writing your coaching report and 90 day roadmap, this takes about 15 to 20 seconds...")

    structured_profile = state["structured_profile"]
    big_five_scores = state["big_five_scores"]
    tipi_answers = state["tipi_answers"]
    grounded_top_ideas = state["grounded_top_ideas"]

    report_state = {
        "structured_profile": structured_profile,
        "big_five_scores": big_five_scores,
        "tipi_answers": tipi_answers,
        "budget_eur": state["budget_eur"],
        "time_available_hours_per_week": state["time_available_hours_per_week"],
        "grounded_top_ideas": grounded_top_ideas,
        "roadmap_idea_id": selected_idea_id,
        "feedback_rounds_used": 0,
        "feedback_history": [],
    }
    try:
        coaching_result = coaching_graph.invoke(report_state)
    except pipeline.ExternalAPIError as e:
        return str(e), None, None, state
    report_narrative = coaching_result["report_narrative"]
    pdf_path = coaching_result["pdf_path"]
    html_path = coaching_result["html_path"]

    roadmap_idea = next((r for r in grounded_top_ideas if r["id"] == selected_idea_id), grounded_top_ideas[0])
    markdown = build_report_markdown(structured_profile, tipi_answers, grounded_top_ideas, report_narrative, roadmap_idea)

    return markdown, pdf_path, html_path, {**report_state, "report_narrative": report_narrative}


MAX_FEEDBACK_ROUNDS = 3
MAX_FEEDBACK_CHARS = 300


def regenerate_with_feedback(feedback_text, report_state):
    # server-side, not just the textbox's max_length, since a direct API call bypasses UI-only limits
    # and every round is a real OpenAI call, so this is a real cost control, not just a UX nicety.
    if report_state is None:
        return "Please generate a report first.", gr.update(), gr.update(), report_state
    if not feedback_text or not feedback_text.strip():
        return "Please write some feedback first, then regenerate.", gr.update(), gr.update(), report_state
    if len(feedback_text) > MAX_FEEDBACK_CHARS:
        return (
            f"Your feedback is {len(feedback_text)} characters, please keep it under {MAX_FEEDBACK_CHARS}.",
            gr.update(), gr.update(), report_state,
        )

    rounds_used = report_state.get("feedback_rounds_used", 0)
    if rounds_used >= MAX_FEEDBACK_ROUNDS:
        return (
            f"You've used all {MAX_FEEDBACK_ROUNDS} feedback regenerations for this report. "
            "Rank your ideas again to start a fresh report.",
            gr.update(), gr.update(), report_state,
        )

    gr.Info(f"Rewriting your report with your feedback ({rounds_used + 1}/{MAX_FEEDBACK_ROUNDS}), this takes about 15 to 20 seconds...")

    # cumulative: every round so far is sent again, not just this one, so earlier feedback (e.g. "don't
    # suggest Wix") isn't silently forgotten once a newer, unrelated feedback round comes in
    feedback_history = report_state.get("feedback_history", []) + [feedback_text]

    try:
        coaching_result = coaching_graph.invoke({**report_state, "feedback_history": feedback_history})
    except pipeline.ExternalAPIError as e:
        return str(e), gr.update(), gr.update(), report_state
    report_narrative = coaching_result["report_narrative"]
    pdf_path = coaching_result["pdf_path"]
    html_path = coaching_result["html_path"]

    grounded_top_ideas = report_state["grounded_top_ideas"]
    roadmap_idea = next(
        (r for r in grounded_top_ideas if r["id"] == report_state["roadmap_idea_id"]), grounded_top_ideas[0]
    )
    markdown = build_report_markdown(
        report_state["structured_profile"], report_state["tipi_answers"], grounded_top_ideas, report_narrative, roadmap_idea
    )

    updated_state = {**report_state, "feedback_rounds_used": rounds_used + 1, "feedback_history": feedback_history, "report_narrative": report_narrative}
    return markdown, pdf_path, html_path, updated_state


TIPI_CSS = """
.gradio-container {
    background-color: #ededed !important;
}
.gradio-container, .gradio-container p, .gradio-container span, .gradio-container label,
.gradio-container li, .gradio-container .prose {
    color: #111111 !important;
    font-size: 1.05rem !important;
}
.gradio-container h1 { font-size: 2.1rem !important; color: #111111 !important; }
.gradio-container h2 { font-size: 1.5rem !important; color: #111111 !important; }
.gradio-container h3 { font-size: 1.2rem !important; color: #111111 !important; }

.primary-action-btn, .primary-action-btn button {
    background: #0e5850 !important;
    border-color: #0e5850 !important;
    color: #ffffff !important;
    font-size: 1.1rem !important;
}
.primary-action-btn:hover, .primary-action-btn button:hover {
    background: #147b71 !important;
}

.tipi-card {
    background: linear-gradient(135deg, #f2fdfc 0%, #e0faf7 100%);
    border-left: 5px solid #20c5b5;
    border-radius: 10px;
    padding: 14px 18px;
    margin-bottom: 14px;
}
.tipi-scale .wrap, .tipi-scale > div { display: flex !important; width: 100%; gap: 8px; }
.tipi-scale label {
    flex: 1 1 0;
    justify-content: center;
    background: #ffffff;
    border: 1px solid #bdf4ef !important;
}
.tipi-scale label.selected {
    background: #20c5b5 !important;
    border-color: #147b71 !important;
    color: #ffffff !important;
}
summary.tipi-info-toggle { cursor: pointer; list-style: none; color: #147b71; font-weight: 600; }
summary.tipi-info-toggle::-webkit-details-marker { display: none; }
"""

with gr.Blocks(title="AI Entrepreneur Coach") as demo:
    gr.Markdown(
        "# AI Entrepreneur Coach\n"
        "Upload your LinkedIn PDF export (or CV), fill in the Big Five quiz, and get business idea "
        "recommendations matched to your traits, skills, budget, and time."
    )

    pdf_input = gr.File(label="Profile PDF (LinkedIn export or CV)", file_types=[".pdf"], type="filepath")

    with gr.Row():
        budget_input = gr.Number(label="Budget (EUR)", value=500)
        time_input = gr.Number(label="Time available (hours/week)", value=10)

    gr.Markdown(
        "## Big Five personality quiz (TIPI)\n"
        "For each statement below, rate how much it describes you, from **1 (disagree strongly)** "
        "to **7 (agree strongly)**, with **4 meaning neutral, neither agree nor disagree**. "
        "\"I see myself as...\"\n\n"
        "Each statement below is actually a pair of traits (for example \"Extraverted, enthusiastic\"). "
        "There are 10 statements in total, 2 for each of the five Big Five traits (Openness, Conscientiousness, "
        "Extraversion, Agreeableness, Neuroticism), this is the real, validated TIPI instrument, not made up questions. "
        "Click the &#9432; next to a statement if you are not sure what it means."
    )
    tipi_sliders = []
    for item in pipeline.TIPI_ITEMS:
        with gr.Group(elem_classes=["tipi-card"]):
            gr.HTML(
                f"<div style='display:flex; align-items:center; gap:8px;'>"
                f"<span>I see myself as: <b>{item['text']}</b></span>"
                f"<details><summary class='tipi-info-toggle'>&#9432;</summary>"
                f"<div style='font-size:0.85rem; color:#666; margin-top:4px;'>{item['info']}</div>"
                f"</details></div>"
            )
            tipi_sliders.append(
                gr.Radio(choices=[1, 2, 3, 4, 5, 6, 7], value=4, label=None, container=False, elem_classes=["tipi-scale"])
            )

    rank_btn = gr.Button("Rank my business ideas", variant="primary", elem_classes=["primary-action-btn"])
    ranked_markdown = gr.Markdown()

    idea_selector = gr.Radio(label="Which idea do you want a 90 day roadmap for?", choices=[])
    coach_state = gr.State()

    coach_btn = gr.Button("Coach me on this one", variant="primary", elem_classes=["primary-action-btn"])

    output_markdown = gr.Markdown()
    with gr.Row():
        output_pdf = gr.File(label="Download PDF report")
        output_html = gr.File(label="Download HTML report")

    report_state = gr.State()
    gr.Markdown(
        "## Not happy with the report? Give feedback and regenerate\n"
        "This rewrites the working style summary, idea rationales, and roadmap text based on your feedback. "
        "It will NOT change the fit percentages, budget, or time ranges above, those come from the actual "
        f"calculation, not from your feedback. Limited to {MAX_FEEDBACK_ROUNDS} regenerations per report and "
        f"{MAX_FEEDBACK_CHARS} characters per feedback."
    )
    feedback_input = gr.Textbox(
        label="Your feedback",
        placeholder="For example: focus more on the low-cost first steps, or make the tone more casual, or I want more freelance platform suggestions.",
        lines=2,
        max_length=MAX_FEEDBACK_CHARS,
    )
    regenerate_btn = gr.Button("Regenerate with my feedback", variant="primary", elem_classes=["primary-action-btn"])

    rank_btn.click(
        rank_ideas,
        inputs=[pdf_input, budget_input, time_input] + tipi_sliders,
        outputs=[ranked_markdown, idea_selector, coach_state],
    )

    coach_btn.click(
        generate_coaching_report,
        inputs=[idea_selector, coach_state],
        outputs=[output_markdown, output_pdf, output_html, report_state],
    )

    regenerate_btn.click(
        regenerate_with_feedback,
        inputs=[feedback_input, report_state],
        outputs=[output_markdown, output_pdf, output_html, report_state],
    )


if __name__ == "__main__":
    demo.queue()
    demo.launch(theme=gr.themes.Default(primary_hue=TURQUOISE), css=TIPI_CSS)
