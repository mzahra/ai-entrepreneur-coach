import json

CONTENT_SECTIONS = ["Top Skills", "Summary", "Experience", "Projects", "Education"]


class ProfileExtractionError(Exception):
    """Raised when the parsed PDF has no real profile content to extract from, so the model
    would otherwise be asked to write a structured profile from nothing, and would fabricate one."""


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


# --- Step 3: extract structured profile ---

def extract_structured_profile(client, profile_sections: dict, profile_header: dict) -> dict:
    has_content = any(profile_sections.get(key, "").strip() for key in CONTENT_SECTIONS)
    if not has_content and not profile_header:
        raise ProfileExtractionError(
            "Could not find any real profile content in this PDF (no skills, summary, experience, "
            "or education text found). Please upload a PDF with actual resume/profile content."
        )

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
