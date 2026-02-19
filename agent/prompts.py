SYSTEM_PROMPT = (
    "You are a strict extraction assistant for Scottish home reports. "
    "Use only provided evidence text. Never assume missing facts. "
    "Return valid JSON only."
)


FIELD_SPECS = {
    "property_address": {
        "label": "Property Address",
        "is_list": False,
        "queries": ["property address", "address", "location"],
    },
    "property_age": {
        "label": "Property Age",
        "is_list": False,
        "queries": ["property age", "built", "construction date", "age of property"],
    },
    "property_epc": {
        "label": "Property EPC",
        "is_list": False,
        "queries": ["EPC", "energy rating", "energy performance"],
    },
    "council_tax_code": {
        "label": "Council Tax Code",
        "is_list": False,
        "queries": ["council tax", "tax band", "band"],
    },
    "recommended_efficiency_measures": {
        "label": "Recommended Measures to Improve Efficiency",
        "is_list": True,
        "queries": ["recommended measures", "efficiency improvements", "energy savings"],
    },
    "window_glazing": {
        "label": "Window Glazing",
        "is_list": False,
        "queries": ["double glazed", "single glazed", "glazing"],
    },
    "potential_problems": {
        "label": "Potential Problems",
        "is_list": True,
        "queries": ["defects", "repairs", "problems", "replacement needed", "condition"],
    },
    "additional_costs": {
        "label": "Additional Costs",
        "is_list": True,
        "queries": ["factor fees", "service charges", "additional costs", "charges"],
    },
    "special_building_notes": {
        "label": "Special Building Notes",
        "is_list": True,
        "queries": ["listed building", "right of way", "servitude", "special conditions"],
    },
    "market_value": {
        "label": "Market Value",
        "is_list": False,
        "queries": ["market value", "valuation", "value"],
    },
}


def make_user_prompt(field_key: str, field_label: str, is_list: bool, evidence_text: str) -> str:
    value_spec = "array of strings" if is_list else "string or null"
    return (
        f"Extract '{field_label}' from the evidence below.\\n"
        "Rules:\\n"
        "- Use only evidence text provided.\\n"
        "- If no direct answer exists, set status='not_found' and value to null (or [] for list fields).\\n"
        "- If evidence is suggestive but not direct, keep value null/[] and add candidate_pages.\\n"
        "- Never guess.\\n"
        "- evidence_paragraph_ids must reference only provided paragraph IDs.\\n\\n"
        "Return JSON object with keys exactly:\\n"
        "value, status, found_pages, candidate_pages, evidence_paragraph_ids, confidence\\n"
        f"Where value is {value_spec}.\\n\\n"
        f"field_key: {field_key}\\n"
        "Evidence:\\n"
        f"{evidence_text}"
    )
