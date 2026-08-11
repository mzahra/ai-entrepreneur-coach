import os
import re
import json
import html
from typing import TypedDict, Optional

from pypdf import PdfReader
from fpdf import FPDF
import cohere
from pinecone import Pinecone
from langgraph.graph import StateGraph, END

LINKEDIN_SECTIONS = [
    "Contact", "Top Skills", "Languages", "Certifications",
    "Honors-Awards", "Summary", "Experience", "Education",
]

# maps a detected CV header (matched by keyword, case-insensitive) to the canonical section key
# extract_structured_profile expects, so a CV written with different wording (e.g. "Professional
# Experience" instead of "Experience") still lines up with the same downstream fields.
GENERIC_SECTION_KEYWORDS = {
    "Top Skills": ["skill", "competenc", "expertise"],
    "Summary": ["summary", "profile", "objective", "about"],
    "Experience": ["experience", "employment", "work history", "career history"],
    "Education": ["education", "academic"],
    "Certifications": ["certificat", "licens"],
    "Languages": ["language"],
    "Projects": ["project"],
}

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

TIPI_ITEMS = [
    {"id": 1, "text": "Extraverted, enthusiastic", "trait": "extraversion", "reverse": False,
     "info": "Being outgoing, talkative, and full of energy when you are around other people."},
    {"id": 2, "text": "Critical, quarrelsome", "trait": "agreeableness", "reverse": True,
     "info": "Often finding fault in others and being quick to argue or disagree."},
    {"id": 3, "text": "Dependable, self-disciplined", "trait": "conscientiousness", "reverse": False,
     "info": "Being reliable and organized, and able to control yourself to finish what you start."},
    {"id": 4, "text": "Anxious, easily upset", "trait": "neuroticism", "reverse": False,
     "info": "Feeling nervous or stressed often, and getting upset easily by small problems."},
    {"id": 5, "text": "Open to new experiences, complex", "trait": "openness", "reverse": False,
     "info": "Enjoying new ideas and ways of thinking, even when they are unusual or complicated."},
    {"id": 6, "text": "Reserved, quiet", "trait": "extraversion", "reverse": True,
     "info": "Preferring to stay quiet, keep to yourself, and not talk much in social situations."},
    {"id": 7, "text": "Sympathetic, warm", "trait": "agreeableness", "reverse": False,
     "info": "Caring about other people's feelings, and being kind and friendly towards them."},
    {"id": 8, "text": "Disorganized, careless", "trait": "conscientiousness", "reverse": True,
     "info": "Not planning ahead, losing track of things, and not paying close attention to details."},
    {"id": 9, "text": "Calm, emotionally stable", "trait": "neuroticism", "reverse": True,
     "info": "Staying relaxed and steady, even in stressful or difficult situations."},
    {"id": 10, "text": "Conventional, uncreative", "trait": "openness", "reverse": True,
     "info": "Preferring familiar, traditional ways of doing things over new or unusual ideas."},
]

FIT_WEIGHTS = {"budget": 0.2, "time": 0.2, "trait": 0.35, "skill": 0.25}

COHERE_EMBED_MODEL = "embed-v4.0"
PINECONE_INDEX_NAME = "entrepreneur-coach-ideas"
RETRIEVAL_TOP_K = 30

_cohere_client = None
_pinecone_index = None


def get_cohere_client():
    global _cohere_client
    if _cohere_client is None:
        _cohere_client = cohere.ClientV2(api_key=os.getenv("COHERE_API_KEY"))
    return _cohere_client


def get_pinecone_index():
    global _pinecone_index
    if _pinecone_index is None:
        pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
        _pinecone_index = pc.Index(PINECONE_INDEX_NAME)
    return _pinecone_index


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


# --- Step 2b: fallback parser for CV formats that are not the LinkedIn export layout ---
# Triggered only when the LinkedIn-specific parser above fails to find "Experience" or "Top Skills",
# so the tested LinkedIn path is unaffected, this only kicks in for other CV formats.

def looks_like_header(line: str) -> bool:
    stripped = line.strip()
    if not stripped or len(stripped) > 45:
        return False
    if stripped.endswith((".", ",", ";", ":")):
        return False
    if stripped.startswith(("•", "-", "*", "◦")):
        return False
    words = stripped.split()
    if not (1 <= len(words) <= 5):
        return False
    if not (stripped.isupper() or stripped.istitle()):
        return False
    if len(words) == 1:
        # a single word is the most likely case of a false positive, a PDF-wrapped sentence can leave one
        # word alone on its own line (e.g. "Structured Outputs" wrapping onto two lines leaves "Outputs" by
        # itself), only trust a single word as a real header if it matches a known CV section keyword
        lower = stripped.lower()
        return any(kw in lower for keywords in GENERIC_SECTION_KEYWORDS.values() for kw in keywords)
    return True


def split_generic_sections(text: str) -> dict:
    lines = text.split("\n")
    sections = {}
    current = "header"
    buffer = []
    seen_first_line = False
    for line in lines:
        stripped = line.strip()
        if re.match(r"^Page \d+ of \d+$", stripped):
            continue
        if not stripped:
            buffer.append(line)
            continue
        if not seen_first_line:
            # the very first non-empty line is the person's name, not a section header, even though a
            # short all caps name looks exactly like one structurally
            seen_first_line = True
            buffer.append(line)
            continue
        if looks_like_header(stripped):
            sections[current] = "\n".join(buffer).strip()
            current = stripped
            buffer = []
        else:
            buffer.append(line)
    sections[current] = "\n".join(buffer).strip()
    return sections


def canonicalize_sections(raw_sections: dict) -> dict:
    canonical = {"header": raw_sections.get("header", "")}
    for header_text, content in raw_sections.items():
        if header_text == "header":
            continue
        header_lower = header_text.lower()
        matched_key = next(
            (key for key, keywords in GENERIC_SECTION_KEYWORDS.items() if any(kw in header_lower for kw in keywords)),
            header_text,  # unrecognized headers are kept as-is, still useful context even if not a canonical field
        )
        canonical[matched_key] = (canonical.get(matched_key, "") + "\n" + content).strip()
    return canonical


PROFILE_LINK_PLATFORMS = ["linkedin", "github", "researchgate", "twitter", "portfolio", "gitlab"]


def extract_generic_header(header_text: str) -> dict:
    lines = [l.strip() for l in header_text.split("\n") if l.strip()]
    if not lines:
        return {}
    name = lines[0]
    location = ""
    headline = ""
    for line in lines[1:6]:
        first_segment = line.split("|")[0].strip()
        if not location and "," in first_segment and "@" not in first_segment:
            location = first_segment
        is_just_links = all(
            any(platform in segment.lower() for platform in PROFILE_LINK_PLATFORMS)
            for segment in line.split("|") if segment.strip()
        )
        if not headline and not is_just_links and "@" not in line and "http" not in line.lower() and not re.search(r"\d{4,}", line):
            headline = line
    return {"name": name, "headline": headline, "location": location}


def parse_profile_pdf(pdf_path: str) -> tuple[dict, dict]:
    reader = PdfReader(pdf_path)
    raw_text = "\n".join(page.extract_text() for page in reader.pages)
    raw_text = fix_line_wrapped_hyphens(raw_text)

    sections = split_linkedin_sections(raw_text)
    if "Experience" in sections and "Top Skills" in sections:
        header = extract_profile_header(sections)
        return sections, header

    # not a LinkedIn export, fall back to the generic heuristic parser
    sections = canonicalize_sections(split_generic_sections(raw_text))
    header = extract_generic_header(sections.get("header", ""))
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

Projects:
{profile_sections.get('Projects', '')}

Education:
{profile_sections.get('Education', '')}
"""
    response = client.responses.create(
        model="gpt-4o-mini",
        input=[
            {
                "role": "system",
                "content": (
                    "Extract a structured profile from the given LinkedIn sections. skills should include both "
                    "explicitly listed skills (Top Skills, KEY SKILLS bullets) and skills clearly implied by the "
                    "rest of the text, do not rely only on an explicit skills list if one exists. In particular, look "
                    "at the job titles and descriptions in Experience, not just the Summary: if the person held "
                    "multiple roles like Trainer, Lecturer, Instructor, or Teacher, or their experience mentions "
                    "teaching/training courses, include skills like Teaching, Training, Public Speaking, or "
                    "Curriculum Development, even if those words never appear in an explicit skills list. The same "
                    "goes for other recurring role patterns (management, writing, sales), a title repeated across "
                    "several jobs is a real skill signal, do not ignore it just because it is not in a bullet list. "
                    "years_of_experience should be your best estimate total professional years."
                ),
            },
            {"role": "user", "content": extraction_input},
        ],
        temperature=0,
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


# --- Step 5/6: retrieve candidates from Pinecone, then rank by computed career best fit ---

def build_profile_query_text(structured_profile: dict) -> str:
    return (
        f"{structured_profile['industry']} professional with skills in {', '.join(structured_profile['skills'])}. "
        f"{structured_profile['experience_summary']}"
    )


def retrieve_candidate_ideas(structured_profile: dict, top_k: int = RETRIEVAL_TOP_K) -> list:
    co = get_cohere_client()
    query_text = build_profile_query_text(structured_profile)
    embed_response = co.embed(
        texts=[query_text],
        model=COHERE_EMBED_MODEL,
        input_type="search_query",
        embedding_types=["float"],
    )
    query_embedding = embed_response.embeddings.float_[0]

    index = get_pinecone_index()
    results = index.query(vector=query_embedding, top_k=top_k, include_metadata=True)
    return [json.loads(match["metadata"]["data"]) for match in results["matches"]]


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
    embed_response = co.embed(
        texts=texts,
        model=COHERE_EMBED_MODEL,
        input_type="classification",
        embedding_types=["float"],
    )
    embeddings = embed_response.embeddings.float_
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
    embed_response = co.embed(
        texts=user_skills + idea_skills,
        model=COHERE_EMBED_MODEL,
        input_type="classification",
        embedding_types=["float"],
    )
    embeddings = embed_response.embeddings.float_
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


def rank_business_ideas(structured_profile: dict, big_five_scores: dict, budget_eur: float, time_available_hours_per_week: float, top_n: int = 5) -> list:
    candidates = retrieve_candidate_ideas(structured_profile)
    skill_fits = compute_skill_fits(structured_profile["skills"], candidates)
    ranked = sorted(
        (compute_career_best_fit(idea, structured_profile, big_five_scores, budget_eur, time_available_hours_per_week, skill_fits[idea["id"]]) for idea in candidates),
        key=lambda r: r["career_best_fit_percentage"],
        reverse=True,
    )
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


# --- Step 7: output report ---

def generate_output_report(client, structured_profile: dict, big_five_scores: dict, budget_eur: float, time_available_hours_per_week: float, grounded_top_ideas: list, roadmap_idea_id: str = None) -> dict:
    roadmap_idea = next((r for r in grounded_top_ideas if r["id"] == roadmap_idea_id), grounded_top_ideas[0])

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
"""
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
            state["grounded_top_ideas"], state.get("roadmap_idea_id"),
        )
        return {"report_narrative": report_narrative}

    def node_export_pdf(state: CoachingState) -> dict:
        pdf_path = export_report_pdf(
            state["structured_profile"], state["report_narrative"], state["grounded_top_ideas"],
            "output/entrepreneur_coach_report.pdf", state.get("roadmap_idea_id"), state.get("tipi_answers"),
        )
        return {"pdf_path": pdf_path}

    def node_export_html(state: CoachingState) -> dict:
        html_path = export_report_html(
            state["structured_profile"], state["report_narrative"], state["grounded_top_ideas"],
            "output/entrepreneur_coach_report.html", state.get("roadmap_idea_id"), state.get("tipi_answers"),
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
