import os
import re
import json

from pypdf import PdfReader
from fpdf import FPDF

LINKEDIN_SECTIONS = [
    "Contact", "Top Skills", "Languages", "Certifications",
    "Honors-Awards", "Summary", "Experience", "Education",
]

STRUCTURED_PROFILE_SCHEMA = {
    "type": "object",
    "properties": {
        "skills": {"type": "array", "items": {"type": "string"}},
        "industry": {"type": "string"},
        "years_of_experience": {"type": "number"},
        "experience_summary": {"type": "string"},
        "highest_education": {"type": "string"},
    },
    "required": ["skills", "industry", "years_of_experience", "experience_summary", "highest_education"],
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
                "days_1_30": {"type": "string"},
                "days_31_60": {"type": "string"},
                "days_61_90": {"type": "string"},
            },
            "required": ["days_1_30", "days_31_60", "days_61_90"],
            "additionalProperties": False,
        },
    },
    "required": ["working_style_summary", "idea_rationales", "roadmap_90_day"],
    "additionalProperties": False,
}

TIPI_ITEMS = [
    {"id": 1, "text": "Extraverted, enthusiastic", "trait": "extraversion", "reverse": False},
    {"id": 2, "text": "Critical, quarrelsome", "trait": "agreeableness", "reverse": True},
    {"id": 3, "text": "Dependable, self-disciplined", "trait": "conscientiousness", "reverse": False},
    {"id": 4, "text": "Anxious, easily upset", "trait": "neuroticism", "reverse": False},
    {"id": 5, "text": "Open to new experiences, complex", "trait": "openness", "reverse": False},
    {"id": 6, "text": "Reserved, quiet", "trait": "extraversion", "reverse": True},
    {"id": 7, "text": "Sympathetic, warm", "trait": "agreeableness", "reverse": False},
    {"id": 8, "text": "Disorganized, careless", "trait": "conscientiousness", "reverse": True},
    {"id": 9, "text": "Calm, emotionally stable", "trait": "neuroticism", "reverse": True},
    {"id": 10, "text": "Conventional, uncreative", "trait": "openness", "reverse": True},
]

BUSINESS_IDEAS_PATH = "input/business_ideas.json"

def load_business_ideas(path: str = BUSINESS_IDEAS_PATH) -> list:
    with open(path) as f:
        return json.load(f)

BUSINESS_IDEAS = load_business_ideas()

FIT_WEIGHTS = {"budget": 0.2, "time": 0.2, "trait": 0.35, "skill": 0.25}


# --- Step 2: parse the profile PDF (LinkedIn export layout) ---

def fix_line_wrapped_hyphens(text: str) -> str:
    # only join when hyphen attaches to the word directly (no space before it), so a real " - " separator stays untouched
    lines = text.split("\n")
    fixed = []
    i = 0
    while i < len(lines):
        line = lines[i]
        if line.endswith("-") and not line.endswith(" -") and i + 1 < len(lines):
            fixed.append(line + lines[i + 1])
            i += 2
        else:
            fixed.append(line)
            i += 1
    return "\n".join(fixed)


def split_linkedin_sections(text: str) -> dict:
    lines = text.split("\n")
    sections = {}
    current = "header"
    buffer = []
    for line in lines:
        stripped = line.strip()
        if re.match(r"^Page \d+ of \d+$", stripped):
            continue
        if stripped in LINKEDIN_SECTIONS:
            sections[current] = "\n".join(buffer).strip()
            current = stripped
            buffer = []
        else:
            buffer.append(line)
    sections[current] = "\n".join(buffer).strip()
    return sections


def extract_profile_header(sections: dict) -> dict:
    # LinkedIn's export puts name, headline, location right before "Summary" with no header word of its own,
    # so it lands at the tail of whatever sidebar section came last, pull it back out here
    section_names = list(sections.keys())
    if "Summary" not in section_names:
        return {}
    prev_section = section_names[section_names.index("Summary") - 1]
    lines = [l for l in sections[prev_section].split("\n") if l.strip()]
    if len(lines) < 3:
        return {}
    name, headline, location = lines[-3], lines[-2], lines[-1]
    sections[prev_section] = "\n".join(lines[:-3]).strip()
    return {"name": name, "headline": headline, "location": location}


def parse_profile_pdf(pdf_path: str) -> tuple[dict, dict]:
    reader = PdfReader(pdf_path)
    raw_text = "\n".join(page.extract_text() for page in reader.pages)
    raw_text = fix_line_wrapped_hyphens(raw_text)
    sections = split_linkedin_sections(raw_text)
    header = extract_profile_header(sections)
    return sections, header


# --- Step 3: extract structured profile ---

def extract_structured_profile(client, profile_sections: dict, profile_header: dict) -> dict:
    extraction_input = f"""Headline: {profile_header.get('headline', '')}
Location: {profile_header.get('location', '')}

Top Skills:
{profile_sections.get('Top Skills', '')}

Summary:
{profile_sections.get('Summary', '')}

Experience:
{profile_sections.get('Experience', '')}

Education:
{profile_sections.get('Education', '')}
"""
    response = client.responses.create(
        model="gpt-4o-mini",
        input=[
            {
                "role": "system",
                "content": "Extract a structured profile from the given LinkedIn sections. skills should include both explicitly listed skills and skills clearly implied by the experience/summary text. years_of_experience should be your best estimate total professional years.",
            },
            {"role": "user", "content": extraction_input},
        ],
        text={"format": {"type": "json_schema", "name": "structured_profile", "schema": STRUCTURED_PROFILE_SCHEMA, "strict": True}},
    )
    structured_profile = json.loads(response.output_text)
    structured_profile["name"] = profile_header.get("name", "")
    structured_profile["location"] = profile_header.get("location", "")
    return structured_profile


# --- Step 4: Big Five via TIPI ---

def score_tipi(answers: dict) -> dict:
    trait_scores = {}
    for item in TIPI_ITEMS:
        raw = answers[item["id"]]
        score = 8 - raw if item["reverse"] else raw
        trait_scores.setdefault(item["trait"], []).append(score)
    return {trait: sum(scores) / len(scores) for trait, scores in trait_scores.items()}


# --- Step 6: career best fit ranking ---

def lower_bound_fit(value: float, low: float, high: float) -> float:
    if low == 0 or value >= low:
        return 1.0
    return max(0.0, value / low)


def range_fit(value: float, low: float, high: float, max_scale: float = 6) -> float:
    if low <= value <= high:
        return 1.0
    distance = low - value if value < low else value - high
    return max(0.0, 1 - distance / max_scale)


def skill_fit(user_skills: list, idea_skills: list) -> float:
    if not idea_skills:
        return 1.0
    user_skills_lower = [s.lower() for s in user_skills]
    matches = sum(
        1 for idea_skill in idea_skills
        if any(idea_skill.lower() in us or us in idea_skill.lower() for us in user_skills_lower)
    )
    return matches / len(idea_skills)


def matched_skills(user_skills: list, idea_skills: list) -> list:
    user_skills_lower = [s.lower() for s in user_skills]
    return [
        idea_skill for idea_skill in idea_skills
        if any(idea_skill.lower() in us or us in idea_skill.lower() for us in user_skills_lower)
    ]


def in_range_traits(idea: dict, big_five_scores: dict) -> list:
    return [trait for trait, bounds in idea["ideal_traits"].items() if bounds[0] <= big_five_scores[trait] <= bounds[1]]


def compute_career_best_fit(idea: dict, structured_profile: dict, big_five_scores: dict, budget_eur: float, time_available_hours_per_week: float) -> dict:
    b_fit = lower_bound_fit(budget_eur, *idea["budget_range_eur"])
    t_fit = lower_bound_fit(time_available_hours_per_week, *idea["time_range_hours_per_week"])
    trait_fits = [range_fit(big_five_scores[trait], *bounds) for trait, bounds in idea["ideal_traits"].items()]
    tr_fit = sum(trait_fits) / len(trait_fits)
    s_fit = skill_fit(structured_profile["skills"], idea["skills_needed"])
    overall = (
        FIT_WEIGHTS["budget"] * b_fit
        + FIT_WEIGHTS["time"] * t_fit
        + FIT_WEIGHTS["trait"] * tr_fit
        + FIT_WEIGHTS["skill"] * s_fit
    )
    return {
        "id": idea["id"],
        "name": idea["name"],
        "description": idea["description"],
        "budget_fit": round(b_fit, 2),
        "time_fit": round(t_fit, 2),
        "trait_fit": round(tr_fit, 2),
        "skill_fit": round(s_fit, 2),
        "career_best_fit_percentage": round(overall * 100, 1),
    }


def rank_business_ideas(structured_profile: dict, big_five_scores: dict, budget_eur: float, time_available_hours_per_week: float, top_n: int = 5) -> list:
    ranked = sorted(
        (compute_career_best_fit(idea, structured_profile, big_five_scores, budget_eur, time_available_hours_per_week) for idea in BUSINESS_IDEAS),
        key=lambda r: r["career_best_fit_percentage"],
        reverse=True,
    )
    top = ranked[:top_n]
    ideas_by_id = {idea["id"]: idea for idea in BUSINESS_IDEAS}
    grounded = []
    for r in top:
        idea = ideas_by_id[r["id"]]
        grounded.append({
            **r,
            "matched_skills": matched_skills(structured_profile["skills"], idea["skills_needed"]),
            "in_range_traits": in_range_traits(idea, big_five_scores),
        })
    return grounded


# --- Step 7: output report ---

def generate_output_report(client, structured_profile: dict, big_five_scores: dict, budget_eur: float, time_available_hours_per_week: float, grounded_top_ideas: list) -> dict:
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
"""
    response = client.responses.create(
        model="gpt-4o-mini",
        input=[
            {
                "role": "system",
                "content": (
                    "You write the narrative parts of a business idea recommendation report. "
                    "The fit percentages are already computed, do not change or restate them as your own judgment. "
                    "working_style_summary: 2 to 4 sentences on the person's working style, based on their Big Five scores and experience. "
                    "idea_rationales: for EACH idea given, one entry with a 1 to 2 sentence rationale that references at least two concrete "
                    "things from the data (matched_skills and/or in_range_traits, name the trait), do not invent skills or traits not present in the data. "
                    "roadmap_90_day: a 90 day roadmap ONLY for the first (top ranked) idea in the list, one entry per phase (days_1_30, days_31_60, days_61_90). "
                    "Each phase should be a full paragraph (5 to 8 sentences), not a one-liner: include concrete first actions, and name specific "
                    "real world places or platforms the person could actually use (for example Upwork, Fiverr, LinkedIn, local meetup or coworking groups, "
                    "relevant subreddits or Slack/Discord communities, industry conferences), matched to the idea's category."
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


def export_report_pdf(structured_profile: dict, report_narrative: dict, grounded_top_ideas: list, output_path: str) -> str:
    rationales_by_id = {r["id"]: r["rationale"] for r in report_narrative["idea_rationales"]}

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

    heading("Ranked business ideas")
    for r in grounded_top_ideas:
        heading(f"{r['name']} - career best fit {r['career_best_fit_percentage']}%", size=11)
        body(r["description"])
        body(rationales_by_id.get(r["id"], ""))
        pdf.ln(2)

    heading(f"90 day roadmap: {grounded_top_ideas[0]['name']}")
    heading("Days 1-30", size=11)
    body(report_narrative["roadmap_90_day"]["days_1_30"])
    heading("Days 31-60", size=11)
    body(report_narrative["roadmap_90_day"]["days_31_60"])
    heading("Days 61-90", size=11)
    body(report_narrative["roadmap_90_day"]["days_61_90"])

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    pdf.output(output_path)
    return output_path


# --- full pipeline, glues every step together ---

def run_full_pipeline(client, pdf_path: str, tipi_answers: dict, budget_eur: float, time_available_hours_per_week: float, output_pdf_path: str = "output/entrepreneur_coach_report.pdf") -> dict:
    profile_sections, profile_header = parse_profile_pdf(pdf_path)
    structured_profile = extract_structured_profile(client, profile_sections, profile_header)
    big_five_scores = score_tipi(tipi_answers)
    grounded_top_ideas = rank_business_ideas(structured_profile, big_five_scores, budget_eur, time_available_hours_per_week)
    report_narrative = generate_output_report(client, structured_profile, big_five_scores, budget_eur, time_available_hours_per_week, grounded_top_ideas)
    pdf_path_out = export_report_pdf(structured_profile, report_narrative, grounded_top_ideas, output_pdf_path)
    return {
        "structured_profile": structured_profile,
        "big_five_scores": big_five_scores,
        "grounded_top_ideas": grounded_top_ideas,
        "report_narrative": report_narrative,
        "pdf_path": pdf_path_out,
    }
