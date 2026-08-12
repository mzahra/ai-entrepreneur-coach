import os
import re
import json
import html

from fpdf import FPDF

from .personality import TIPI_ITEMS


def slugify_filename_part(text: str) -> str:
    text = re.sub(r"[^\w\s-]", "", text)
    return re.sub(r"[\s-]+", "_", text.strip())


def build_report_filename(user_name: str, idea_name: str, extension: str) -> str:
    return f"report_{slugify_filename_part(user_name)}_{slugify_filename_part(idea_name)}.{extension}"


def _roadmap_phase_schema() -> dict:
    return {
        "type": "object",
        "properties": {
            "summary": {"type": "string"},
            "action_items": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "action": {"type": "string"},
                        "estimated_hours": {"type": "number"},
                        "estimated_cost_eur": {"type": "number"},
                    },
                    "required": ["action", "estimated_hours", "estimated_cost_eur"],
                    "additionalProperties": False,
                },
            },
        },
        "required": ["summary", "action_items"],
        "additionalProperties": False,
    }


OUTPUT_REPORT_SCHEMA = {
    "type": "object",
    "properties": {
        "working_style_summary": {"type": "string"},
        "idea_rationales": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {"id": {"type": "string"}, "rationale": {"type": "string"}},
                "required": ["id", "rationale"],
                "additionalProperties": False,
            },
        },
        "roadmap_90_day": {
            "type": "object",
            "properties": {
                "days_1_30": _roadmap_phase_schema(),
                "days_31_60": _roadmap_phase_schema(),
                "days_61_90": _roadmap_phase_schema(),
            },
            "required": ["days_1_30", "days_31_60", "days_61_90"],
            "additionalProperties": False,
        },
    },
    "required": ["working_style_summary", "idea_rationales", "roadmap_90_day"],
    "additionalProperties": False,
}


# --- Step 7: output report ---

def generate_output_report(client, structured_profile: dict, big_five_scores: dict, budget_eur: float, time_available_hours_per_week: float, grounded_top_ideas: list, roadmap_idea_id: str = None, feedback_history: list = None) -> dict:
    roadmap_idea = next((r for r in grounded_top_ideas if r["id"] == roadmap_idea_id), grounded_top_ideas[0])

    feedback_block = ""
    if feedback_history:
        numbered_feedback = "\n".join(f"{i + 1}. {fb}" for i, fb in enumerate(feedback_history))
        feedback_block = f"""
The user already saw earlier versions of this report and gave feedback, numbered below in the order they
gave it. ALL of these still apply together, not just the last one, earlier feedback is not replaced by
later feedback unless a later point specifically contradicts an earlier one (in that case, follow the more
recent instruction only for that specific point, everything else from earlier rounds still applies). Use
this to rewrite the report, but only for tone, emphasis, and which points to focus on, do NOT let it change
budget_range_eur, time_range_hours_per_week, or career_best_fit_percentage, those are already computed from
real data, not your judgment.

If the feedback asks for a lower budget or less time than the idea realistically needs: the roadmap must
still include the real, necessary costs implied by this idea's skills_needed and description (for example
materials, tools, software, or certifications a person would actually need to do this work, not just
marketing/setup tasks). Do NOT quietly drop or avoid mentioning a necessary cost just to make the total look
lower, a roadmap that hits a low number by omitting something the person obviously needs is misleading, not
helpful. If, even after minimizing everywhere genuinely possible, the total still cannot reach what the user
asked for, keep the necessary items in and add one clear sentence in the relevant phase summary explaining
that the real minimum is higher and naming the actual cost driver (for example "materials and tools for
furniture making cannot be avoided"). Apply everything else in the feedback fully.

User feedback, in order given:
{numbered_feedback}
"""

    report_input = f"""User profile:
Name: {structured_profile['name']}
Industry: {structured_profile['industry']}
Years of experience: {structured_profile['years_of_experience']}
Skills: {', '.join(structured_profile['skills'])}
Experience summary: {structured_profile['experience_summary']}

Big Five scores (1-7 scale): {json.dumps(big_five_scores)}
Budget: €{budget_eur}
Time available: {time_available_hours_per_week} hours/week

Top ranked business ideas (already ranked and scored, do not change the ranking or invent a different fit number):
{json.dumps(grounded_top_ideas, indent=2)}

Reminder: idea_rationales must contain exactly {len(grounded_top_ideas)} entries, one for EACH of these ids: {[r['id'] for r in grounded_top_ideas]}, do not skip any of them.
The 90 day roadmap (roadmap_90_day) is ONLY for the idea the user picked, id "{roadmap_idea['id']}", name "{roadmap_idea['name']}", "
budget_range_eur {roadmap_idea['budget_range_eur']}, time_range_hours_per_week {roadmap_idea['time_range_hours_per_week']}, not for the others.
{feedback_block}"""
    response = client.responses.create(
        model="gpt-4o-mini",
        input=[
            {
                "role": "system",
                "content": (
                    "You write the narrative parts of a business idea recommendation report. "
                    "Address the person directly as \"you\"/\"your\" throughout, like a coach talking to them, not by their name and not in the third person "
                    "(write \"you should focus on...\", not \"Zahra should focus on...\"). "
                    "Write in plain, simple English throughout, short sentences, everyday words, this report is read by a general audience "
                    "including non-native English speakers, avoid business jargon and complex vocabulary (for example say \"you work well with "
                    "others\" not \"you are an approachable collaborator\", say \"you like trying new things\" not \"you exhibit a penchant for "
                    "novel experiences\"). "
                    "The fit percentages are already computed, do not change or restate them as your own judgment. "
                    "working_style_summary: 2 to 4 short, simple sentences on the person's working style, based on their Big Five scores and "
                    "experience, explain what the trait means in plain terms rather than naming it clinically (for example \"you stay calm under "
                    "pressure\" instead of just \"low neuroticism\"). "
                    "idea_rationales: one entry for EVERY idea in the list, all of them, not just the one the user picked for the roadmap, "
                    "each a 1 to 2 sentence rationale that references at least two concrete things from the data (matched_skills and/or "
                    "in_range_traits, name the trait), do not invent skills or traits not present in the data. "
                    "roadmap_90_day: a 90 day roadmap ONLY for the idea the user picked, identified at the end of the user message by its id and name, "
                    "not necessarily the top ranked one, one entry per phase (days_1_30, days_31_60, days_61_90). "
                    "Each phase needs: summary (1 sentence overview of the phase's goal), and action_items, a list of 3 to 5 concrete, specific "
                    "actions (not vague advice), each with estimated_hours (realistic hours to complete just that action) and estimated_cost_eur "
                    "(realistic cost in EUR for just that action, 0 if free). Name specific real world places or platforms in the action text where "
                    "relevant (for example Upwork, Fiverr, LinkedIn, local meetup or coworking groups, relevant subreddits or Slack/Discord "
                    "communities, industry conferences), matched to the idea's category. "
                    "Across all 3 phases combined, the sum of estimated_hours per week should roughly stay within time_range_hours_per_week, and the "
                    "sum of estimated_cost_eur should roughly stay within budget_range_eur, both given for the idea in the user message, do not wildly "
                    "exceed them without a concrete reason."
                ),
            },
            {"role": "user", "content": report_input},
        ],
        text={"format": {"type": "json_schema", "name": "output_report", "schema": OUTPUT_REPORT_SCHEMA, "strict": True}},
    )
    return json.loads(response.output_text)


# --- Step 8: export the report as a PDF ---

def sanitize_for_pdf(text: str) -> str:
    replacements = {"‘": "'", "’": "'", "“": '"', "”": '"', "–": "-", "—": "-", "…": "..."}
    for bad, good in replacements.items():
        text = text.replace(bad, good)
    return text.encode("latin-1", "replace").decode("latin-1")


def export_report_pdf(structured_profile: dict, report_narrative: dict, grounded_top_ideas: list, output_path: str, roadmap_idea_id: str = None, tipi_answers: dict = None) -> str:
    roadmap_idea = next((r for r in grounded_top_ideas if r["id"] == roadmap_idea_id), grounded_top_ideas[0])

    pdf = FPDF()
    pdf.add_page()
    pdf.set_auto_page_break(auto=True, margin=15)

    def heading(text: str, size: int = 13) -> None:
        pdf.set_font("Helvetica", "B", size)
        pdf.multi_cell(0, 8, sanitize_for_pdf(text))
        pdf.ln(1)

    def body(text: str, size: int = 11) -> None:
        pdf.set_font("Helvetica", "", size)
        pdf.multi_cell(0, 6, sanitize_for_pdf(text))
        pdf.ln(1)

    heading(f"AI Entrepreneur Coach report for {structured_profile['name']}", size=16)
    pdf.ln(2)

    heading("Working style summary")
    body(report_narrative["working_style_summary"])
    pdf.ln(2)

    if tipi_answers:
        heading("Personality quiz (TIPI) answers")
        pdf.set_font("Helvetica", "", 10)
        with pdf.table(col_widths=(90, 30, 20), text_align=("LEFT", "CENTER", "CENTER")) as table:
            header_row = table.row()
            for h in ["I see myself as...", "Trait", "Your rating (1-7)"]:
                header_row.cell(h)
            for item in TIPI_ITEMS:
                row = table.row()
                row.cell(sanitize_for_pdf(item["text"]))
                row.cell(item["trait"].capitalize())
                row.cell(str(tipi_answers[item["id"]]))
        pdf.ln(3)

    heading("Ranked business ideas")
    pdf.set_font("Helvetica", "", 10)
    with pdf.table(col_widths=(70, 15, 30, 25), text_align=("LEFT", "CENTER", "CENTER", "CENTER")) as table:
        header_row = table.row()
        for h in ["Idea", "Fit %", "Budget (EUR)", "Time (h/wk)"]:
            header_row.cell(h)
        for r in grounded_top_ideas:
            row = table.row()
            row.cell(sanitize_for_pdf(r["name"]))
            row.cell(f"{r['career_best_fit_percentage']}%")
            row.cell(f"{r['budget_range_eur'][0]:.0f}-{r['budget_range_eur'][1]:.0f}")
            row.cell(f"{r['time_range_hours_per_week'][0]:.0f}-{r['time_range_hours_per_week'][1]:.0f}")
    pdf.ln(3)

    heading(f"90 day roadmap: {roadmap_idea['name']}")
    body(roadmap_idea["description"])
    body(
        f"Budget: EUR {roadmap_idea['budget_range_eur'][0]:.0f}-{roadmap_idea['budget_range_eur'][1]:.0f}"
        f"  |  Time: {roadmap_idea['time_range_hours_per_week'][0]:.0f}-{roadmap_idea['time_range_hours_per_week'][1]:.0f} hours/week"
    )
    pdf.ln(1)

    for key, label in [("days_1_30", "Days 1-30"), ("days_31_60", "Days 31-60"), ("days_61_90", "Days 61-90")]:
        phase = report_narrative["roadmap_90_day"][key]
        heading(label, size=12)
        body(phase["summary"])

        pdf.set_font("Helvetica", "", 10)
        total_hours = sum(item["estimated_hours"] for item in phase["action_items"])
        total_cost = sum(item["estimated_cost_eur"] for item in phase["action_items"])
        with pdf.table(col_widths=(110, 20, 25), text_align=("LEFT", "CENTER", "CENTER")) as table:
            header_row = table.row()
            for h in ["Action", "Hours", "Cost (EUR)"]:
                header_row.cell(h)
            for item in phase["action_items"]:
                row = table.row()
                row.cell(sanitize_for_pdf(item["action"]))
                row.cell(f"{item['estimated_hours']:.1f}")
                row.cell(f"{item['estimated_cost_eur']:.0f}")
            total_row = table.row()
            total_row.cell("Phase total")
            total_row.cell(f"{total_hours:.1f}")
            total_row.cell(f"{total_cost:.0f}")
        pdf.ln(3)

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    pdf.output(output_path)
    return output_path


# --- Step 8b: export the report as a standalone styled HTML file ---

def export_report_html(structured_profile: dict, report_narrative: dict, grounded_top_ideas: list, output_path: str, roadmap_idea_id: str = None, tipi_answers: dict = None) -> str:
    roadmap_idea = next((r for r in grounded_top_ideas if r["id"] == roadmap_idea_id), grounded_top_ideas[0])

    def esc(value) -> str:
        return html.escape(str(value))

    tipi_section = ""
    if tipi_answers:
        tipi_rows = "\n".join(
            f"<tr><td>{esc(item['text'])}</td><td>{esc(item['trait'].capitalize())}</td>"
            f"<td class='num'>{tipi_answers[item['id']]}</td></tr>"
            for item in TIPI_ITEMS
        )
        tipi_section = f"""<h2>Personality quiz (TIPI) answers</h2>
  <table>
    <thead><tr><th>I see myself as...</th><th>Trait</th><th class="num">Your rating (1-7)</th></tr></thead>
    <tbody>
      {tipi_rows}
    </tbody>
  </table>"""

    idea_rows = "\n".join(
        f"<tr><td>{esc(r['name'])}</td><td class='num'>{r['career_best_fit_percentage']}%</td>"
        f"<td class='num'>&euro;{r['budget_range_eur'][0]:.0f}-{r['budget_range_eur'][1]:.0f}</td>"
        f"<td class='num'>{r['time_range_hours_per_week'][0]:.0f}-{r['time_range_hours_per_week'][1]:.0f}</td></tr>"
        for r in grounded_top_ideas
    )

    def phase_html(key: str, label: str) -> str:
        phase = report_narrative["roadmap_90_day"][key]
        total_hours = sum(item["estimated_hours"] for item in phase["action_items"])
        total_cost = sum(item["estimated_cost_eur"] for item in phase["action_items"])
        rows = "\n".join(
            f"<tr><td>{esc(item['action'])}</td><td class='num'>{item['estimated_hours']:.1f}</td>"
            f"<td class='num'>&euro;{item['estimated_cost_eur']:.0f}</td></tr>"
            for item in phase["action_items"]
        )
        return f"""<div class="phase">
  <h3>{label}</h3>
  <p>{esc(phase['summary'])}</p>
  <table>
    <thead><tr><th>Action</th><th class="num">Hours</th><th class="num">Cost</th></tr></thead>
    <tbody>
      {rows}
      <tr class="total-row"><td>Phase total</td><td class="num">{total_hours:.1f}</td><td class="num">&euro;{total_cost:.0f}</td></tr>
    </tbody>
  </table>
</div>"""

    roadmap_html = "\n".join([
        phase_html("days_1_30", "Days 1-30"),
        phase_html("days_31_60", "Days 31-60"),
        phase_html("days_61_90", "Days 61-90"),
    ])

    html_doc = f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>AI Entrepreneur Coach report for {esc(structured_profile['name'])}</title>
<style>
  :root {{
    --turquoise: #20c5b5;
    --turquoise-dark: #147b71;
    --text: #1f2937;
    --muted: #6b7280;
    --border: #e5e7eb;
    --card-bg: #f9fafb;
  }}
  * {{ box-sizing: border-box; }}
  body {{
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Helvetica, Arial, sans-serif;
    color: var(--text);
    max-width: 800px;
    margin: 40px auto;
    padding: 0 20px 60px 20px;
    line-height: 1.5;
  }}
  h1 {{ font-size: 1.8rem; border-bottom: 3px solid var(--turquoise); padding-bottom: 10px; }}
  h2 {{ font-size: 1.3rem; margin-top: 2.5rem; color: var(--turquoise-dark); }}
  h3 {{ font-size: 1.05rem; margin-bottom: 4px; }}
  .summary-box {{ background: var(--card-bg); border-left: 4px solid var(--turquoise); padding: 16px 20px; border-radius: 6px; }}
  table {{ width: 100%; border-collapse: collapse; margin: 12px 0 24px 0; font-size: 0.95rem; }}
  th, td {{ border: 1px solid var(--border); padding: 8px 10px; text-align: left; }}
  th {{ background: var(--card-bg); }}
  td.num, th.num {{ text-align: center; white-space: nowrap; }}
  tr.total-row {{ font-weight: 600; background: var(--card-bg); }}
  .idea-meta {{ color: var(--muted); font-size: 0.9rem; margin: 4px 0 8px 0; }}
  @media print {{
    body {{ margin: 0; max-width: none; }}
    table {{ break-inside: avoid; }}
  }}
</style>
</head>
<body>
  <h1>AI Entrepreneur Coach report for {esc(structured_profile['name'])}</h1>

  <h2>Working style summary</h2>
  <div class="summary-box"><p>{esc(report_narrative['working_style_summary'])}</p></div>

  {tipi_section}

  <h2>Ranked business ideas</h2>
  <table>
    <thead><tr><th>Idea</th><th class="num">Fit %</th><th class="num">Budget</th><th class="num">Time</th></tr></thead>
    <tbody>
      {idea_rows}
    </tbody>
  </table>

  <h2>90 day roadmap: {esc(roadmap_idea['name'])}</h2>
  <p>{esc(roadmap_idea['description'])}</p>
  <p class="idea-meta">Budget: &euro;{roadmap_idea['budget_range_eur'][0]:.0f}-{roadmap_idea['budget_range_eur'][1]:.0f} &nbsp;|&nbsp; Time: {roadmap_idea['time_range_hours_per_week'][0]:.0f}-{roadmap_idea['time_range_hours_per_week'][1]:.0f} h/wk</p>

  {roadmap_html}
</body>
</html>
"""

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(html_doc)
    return output_path
