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

    recommendation_result = recommendation_graph.invoke({
        "pdf_path": pdf_file,
        "tipi_answers": tipi_answers,
        "budget_eur": budget_eur,
        "time_available_hours_per_week": time_available_hours_per_week,
    })
    structured_profile = recommendation_result["structured_profile"]
    big_five_scores = recommendation_result["big_five_scores"]
    grounded_top_ideas = recommendation_result["grounded_top_ideas"]

    lines = ["## Ranked business ideas\n"]
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


def generate_coaching_report(selected_idea_id, state):
    if state is None:
        return "Please rank your ideas first.", None, None

    gr.Info("Writing your coaching report and 90 day roadmap, this takes about 15 to 20 seconds...")

    structured_profile = state["structured_profile"]
    big_five_scores = state["big_five_scores"]
    tipi_answers = state["tipi_answers"]
    grounded_top_ideas = state["grounded_top_ideas"]

    coaching_result = coaching_graph.invoke({
        "structured_profile": structured_profile,
        "big_five_scores": big_five_scores,
        "tipi_answers": tipi_answers,
        "budget_eur": state["budget_eur"],
        "time_available_hours_per_week": state["time_available_hours_per_week"],
        "grounded_top_ideas": grounded_top_ideas,
        "roadmap_idea_id": selected_idea_id,
    })
    report_narrative = coaching_result["report_narrative"]
    pdf_path = coaching_result["pdf_path"]
    html_path = coaching_result["html_path"]

    roadmap_idea = next((r for r in grounded_top_ideas if r["id"] == selected_idea_id), grounded_top_ideas[0])

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

    return "\n".join(lines), pdf_path, html_path


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

    rank_btn.click(
        rank_ideas,
        inputs=[pdf_input, budget_input, time_input] + tipi_sliders,
        outputs=[ranked_markdown, idea_selector, coach_state],
    )

    coach_btn.click(
        generate_coaching_report,
        inputs=[idea_selector, coach_state],
        outputs=[output_markdown, output_pdf, output_html],
    )


if __name__ == "__main__":
    demo.queue()
    demo.launch(theme=gr.themes.Default(primary_hue=TURQUOISE), css=TIPI_CSS)
