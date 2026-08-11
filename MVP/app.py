import os

import gradio as gr
from dotenv import load_dotenv
from openai import OpenAI

import pipeline

load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

TURQUOISE = gr.themes.Color(
    name="turquoise",
    c50="#f2fdfc", c100="#e0faf7", c200="#bdf4ef", c300="#91ede4", c400="#65e6d9",
    c500="#3adfce", c600="#20c5b5", c700="#1a9e91", c800="#147b71", c900="#0e5850", c950="#093530",
)


def rank_ideas(pdf_file, budget_eur, time_available_hours_per_week, *tipi_ratings):
    if pdf_file is None:
        return "Please upload a profile PDF first.", gr.update(choices=[], value=None, visible=False), None

    tipi_answers = {item["id"]: int(rating) for item, rating in zip(pipeline.TIPI_ITEMS, tipi_ratings)}
    big_five_scores = pipeline.score_tipi(tipi_answers)

    profile_sections, profile_header = pipeline.parse_profile_pdf(pdf_file)
    structured_profile = pipeline.extract_structured_profile(client, profile_sections, profile_header)
    grounded_top_ideas = pipeline.rank_business_ideas(
        structured_profile, big_five_scores, budget_eur, time_available_hours_per_week
    )

    lines = ["## Ranked business ideas\n"]
    for r in grounded_top_ideas:
        lines.append(f"**{r['name']}**, career best fit {r['career_best_fit_percentage']}%")
        lines.append(r["description"] + "\n")

    idea_choices = [(f"{r['name']} ({r['career_best_fit_percentage']}%)", r["id"]) for r in grounded_top_ideas]

    state = {
        "structured_profile": structured_profile,
        "big_five_scores": big_five_scores,
        "grounded_top_ideas": grounded_top_ideas,
        "budget_eur": budget_eur,
        "time_available_hours_per_week": time_available_hours_per_week,
    }

    return "\n".join(lines), gr.update(choices=idea_choices, value=idea_choices[0][1], visible=True), state


def generate_coaching_report(selected_idea_id, state):
    if state is None:
        return "Please rank your ideas first.", None

    structured_profile = state["structured_profile"]
    big_five_scores = state["big_five_scores"]
    grounded_top_ideas = state["grounded_top_ideas"]

    report_narrative = pipeline.generate_output_report(
        client,
        structured_profile,
        big_five_scores,
        state["budget_eur"],
        state["time_available_hours_per_week"],
        grounded_top_ideas,
        roadmap_idea_id=selected_idea_id,
    )
    pdf_path = pipeline.export_report_pdf(
        structured_profile,
        report_narrative,
        grounded_top_ideas,
        "output/entrepreneur_coach_report.pdf",
        roadmap_idea_id=selected_idea_id,
    )

    roadmap_idea = next((r for r in grounded_top_ideas if r["id"] == selected_idea_id), grounded_top_ideas[0])
    rationales_by_id = {r["id"]: r["rationale"] for r in report_narrative["idea_rationales"]}

    lines = [f"# AI Entrepreneur Coach report for {structured_profile['name']}\n"]
    lines.append("## Working style summary\n")
    lines.append(report_narrative["working_style_summary"] + "\n")
    lines.append("## Ranked business ideas\n")
    for r in grounded_top_ideas:
        lines.append(f"### {r['name']}, career best fit {r['career_best_fit_percentage']}%")
        lines.append(r["description"])
        lines.append(rationales_by_id.get(r["id"], "") + "\n")
    lines.append(f"## 90 day roadmap: {roadmap_idea['name']}\n")
    lines.append("### Days 1-30")
    lines.append(report_narrative["roadmap_90_day"]["days_1_30"] + "\n")
    lines.append("### Days 31-60")
    lines.append(report_narrative["roadmap_90_day"]["days_31_60"] + "\n")
    lines.append("### Days 61-90")
    lines.append(report_narrative["roadmap_90_day"]["days_61_90"])

    return "\n".join(lines), pdf_path


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
        "Extraversion, Agreeableness, Neuroticism), this is the real, validated TIPI instrument, not made up questions."
    )
    tipi_sliders = [
        gr.Slider(1, 7, value=4, step=1, label=item["text"])
        for item in pipeline.TIPI_ITEMS
    ]

    rank_btn = gr.Button("Rank my business ideas", variant="primary")
    ranked_markdown = gr.Markdown()

    idea_selector = gr.Radio(label="Which idea do you want a 90 day roadmap for?", choices=[], visible=False)
    coach_state = gr.State()

    coach_btn = gr.Button("Coach me on this one", variant="primary")

    output_markdown = gr.Markdown()
    output_pdf = gr.File(label="Download PDF report")

    rank_btn.click(
        rank_ideas,
        inputs=[pdf_input, budget_input, time_input] + tipi_sliders,
        outputs=[ranked_markdown, idea_selector, coach_state],
    )

    coach_btn.click(
        generate_coaching_report,
        inputs=[idea_selector, coach_state],
        outputs=[output_markdown, output_pdf],
    )


if __name__ == "__main__":
    demo.launch(theme=gr.themes.Default(primary_hue=TURQUOISE))
