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


def run_app(pdf_file, budget_eur, time_available_hours_per_week, *tipi_ratings):
    if pdf_file is None:
        return "Please upload a profile PDF first.", None

    tipi_answers = {item["id"]: int(rating) for item, rating in zip(pipeline.TIPI_ITEMS, tipi_ratings)}

    result = pipeline.run_full_pipeline(
        client=client,
        pdf_path=pdf_file,
        tipi_answers=tipi_answers,
        budget_eur=budget_eur,
        time_available_hours_per_week=time_available_hours_per_week,
    )

    structured_profile = result["structured_profile"]
    report_narrative = result["report_narrative"]
    grounded_top_ideas = result["grounded_top_ideas"]
    rationales_by_id = {r["id"]: r["rationale"] for r in report_narrative["idea_rationales"]}

    lines = [f"# AI Entrepreneur Coach report for {structured_profile['name']}\n"]
    lines.append("## Working style summary\n")
    lines.append(report_narrative["working_style_summary"] + "\n")
    lines.append("## Ranked business ideas\n")
    for r in grounded_top_ideas:
        lines.append(f"### {r['name']}, career best fit {r['career_best_fit_percentage']}%")
        lines.append(r["description"])
        lines.append(rationales_by_id.get(r["id"], "") + "\n")
    lines.append(f"## 90 day roadmap: {grounded_top_ideas[0]['name']}\n")
    lines.append("### Days 1-30")
    lines.append(report_narrative["roadmap_90_day"]["days_1_30"] + "\n")
    lines.append("### Days 31-60")
    lines.append(report_narrative["roadmap_90_day"]["days_31_60"] + "\n")
    lines.append("### Days 61-90")
    lines.append(report_narrative["roadmap_90_day"]["days_61_90"])

    return "\n".join(lines), result["pdf_path"]


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

    submit_btn = gr.Button("Get my recommendations", variant="primary")

    output_markdown = gr.Markdown()
    output_pdf = gr.File(label="Download PDF report")

    submit_btn.click(
        run_app,
        inputs=[pdf_input, budget_input, time_input] + tipi_sliders,
        outputs=[output_markdown, output_pdf],
    )


if __name__ == "__main__":
    demo.launch(theme=gr.themes.Default(primary_hue=TURQUOISE))
