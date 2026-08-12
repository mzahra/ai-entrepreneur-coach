import re

from pypdf import PdfReader

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

PROFILE_LINK_PLATFORMS = ["linkedin", "github", "researchgate", "twitter", "portfolio", "gitlab"]


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


class ProfilePdfError(Exception):
    """Raised when the uploaded file cannot be read as a PDF at all."""


def parse_profile_pdf(pdf_path: str) -> tuple[dict, dict]:
    try:
        reader = PdfReader(pdf_path)
        raw_text = "\n".join(page.extract_text() for page in reader.pages)
    except Exception as e:
        raise ProfilePdfError(
            "Could not read this file as a PDF. Please upload a valid PDF (LinkedIn export or CV)."
        ) from e
    raw_text = fix_line_wrapped_hyphens(raw_text)

    sections = split_linkedin_sections(raw_text)
    if "Experience" in sections and "Top Skills" in sections:
        header = extract_profile_header(sections)
        return sections, header

    # not a LinkedIn export, fall back to the generic heuristic parser
    sections = canonicalize_sections(split_generic_sections(raw_text))
    header = extract_generic_header(sections.get("header", ""))
    return sections, header
