# --- Step 4: Big Five via TIPI ---

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


def score_tipi(answers: dict) -> dict:
    trait_scores = {}
    for item in TIPI_ITEMS:
        raw = answers[item["id"]]
        score = 8 - raw if item["reverse"] else raw
        trait_scores.setdefault(item["trait"], []).append(score)
    return {trait: sum(scores) / len(scores) for trait, scores in trait_scores.items()}
