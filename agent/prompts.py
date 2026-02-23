SYSTEM_PROMPT = (
    "You are a strict extraction assistant for Scottish home reports. "
    "Use only provided evidence text. Never assume missing facts. "
    "Return valid JSON only."
)


SECTION_ROUTER_SYSTEM_PROMPT = (
    "You are selecting relevant document sections for information extraction. "
    "Use only the provided section titles and summaries. "
    "Return valid JSON only."
)


FIELD_SPECS = {
    "property_address": {
        "label": "Property Address",
        "is_list": False,
        "queries": ["property address", "address", "location"],
        "extra_prompt": None,
    },
    "property_age": {
        "label": "Property Age",
        "is_list": False,
        "queries": ["property age", "built", "construction date", "age of property"],
        "extra_prompt": None,
    },
    "property_area": {
        "label": "Property Area",
        "is_list": False,
        "queries": ["area", "floor area", "sq m", "square metres", "m2", "ft2"],
        "extra_prompt": (
            "Extract property area exactly as stated, including units used in the report "
            "(e.g., sq m, m2, ft2)."
        ),
    },
    "property_epc": {
        "label": "Property EPC",
        "is_list": False,
        "queries": ["EPC", "energy rating", "energy performance"],
        "extra_prompt": None,
    },
    "council_tax_code": {
        "label": "Council Tax Code",
        "is_list": False,
        "queries": ["council tax", "tax band", "band"],
        "extra_prompt": None,
    },
    "recommended_efficiency_measures": {
        "label": "Recommended Measures to Improve Efficiency",
        "is_list": True,
        "queries": ["recommended measures", "efficiency improvements", "energy savings"],
        "extra_prompt": (
            "Only include concrete recommended measures/actions for the property. "
            "Do not include generic narrative text."
        ),
    },
    "window_glazing": {
        "label": "Window Glazing",
        "is_list": False,
        "queries": ["double glazed", "single glazed", "glazing"],
        "extra_prompt": (
            "Extract only glazing type explicitly stated (e.g., double glazed/single glazed/mixed). "
            "If not explicit, return not_found."
        ),
    },
    "external_walls_material": {
        "label": "External Walls Material",
        "is_list": False,
        "queries": ["external walls", "wall construction", "outer walls", "masonry", "brick", "stone"],
        "extra_prompt": (
            "Extract only what external walls are made of, if explicitly stated."
        ),
    },
    "internal_walls_material": {
        "label": "Internal Walls Material",
        "is_list": False,
        "queries": ["internal walls", "partition walls", "wall lining", "plasterboard"],
        "extra_prompt": (
            "Extract only what internal walls are made of, if explicitly stated."
        ),
    },
    "gas_and_boiler_notes": {
        "label": "Gas and Boiler Notes",
        "is_list": False,
        "queries": ["gas", "boiler", "gas supply", "heating system", "gas safety"],
        "extra_prompt": (
            "Summarize only explicit notes about gas supply/system and boiler condition/type/safety."
        ),
    },
    "electricity_notes": {
        "label": "Electricity Notes",
        "is_list": False,
        "queries": ["electricity", "electrical", "consumer unit", "wiring", "electrical safety"],
        "extra_prompt": (
            "Summarize only explicit notes about electrical installation/supply/condition/safety."
        ),
    },
    "potential_problems": {
        "label": "Potential Problems",
        "is_list": True,
        "queries": ["defects", "repairs", "problems", "replacement needed", "condition", "leak",
        "flooding", "damp", "rot", "structural issues"],
        "extra_prompt": (
            "Include only concrete property issues/defects/repair needs. "
            "Avoid generic disclaimers or process-related text."
        ),
    },
    "additional_costs": {
        "label": "Additional Costs",
        "is_list": True,
        "queries": ["factor fees", "service charges", "additional costs", "charges"],
        "extra_prompt": (
            "Include only concrete property-related costs payable by owner/occupier (e.g., factor fees, "
            "service/maintenance charges, communal charges) when explicitly stated. "
            "Exclude survey/report/conveyancing/inspection/cancellation/administrative fees and contractual boilerplate. "
            "If no property-related amount or fee is explicitly stated, return not_found."
        ),
    },
    "special_building_notes": {
        "label": "Special Building Notes",
        "is_list": True,
        "queries": ["listed building", "right of way", "servitude", "special conditions", "flood",
        "damp", "rot", "structural issues", "general remarks", "mine"],
        "extra_prompt": (
            "Identify any explicit out-of-ordinary property notes that could materially affect ownership, use, risk, value, insurance, or resale. "
            "Include legal, environmental, historical, structural, or access-related flags"
            " (for example: rights of way/servitudes, listed or protected status, conservation constraints, former mining land, flood/ground-risk references,"
            " contamination, statutory notices, unusual title burdens, shared/private access restrictions). "
            "Do not list normal property features or generic boilerplate. "
            "Use only facts explicitly stated in the evidence; if nothing unusual is explicitly stated, return not_found."
        ),
    },
    "market_value": {
        "label": "Market Value",
        "is_list": False,
        "queries": ["market value", "valuation", "value"],
        "extra_prompt": (
            "Extract the market value/valuation amount only if explicitly stated for the property."
        ),
    },
}


def make_user_prompt(field_key: str, field_label: str, is_list: bool, evidence_text: str) -> str:
    value_spec = "array of strings" if is_list else "string or null"
    extra_prompt = FIELD_SPECS.get(field_key, {}).get("extra_prompt")
    extra_prompt_block = ""
    if extra_prompt:
        extra_prompt_block = f"Field-specific instruction:\\n- {extra_prompt}\\n\\n"
    return (
        f"Extract '{field_label}' from the evidence below.\\n"
        "Rules:\\n"
        "- Use only evidence text provided.\\n"
        "- If no direct answer exists, set status='not_found' and value to null (or [] for list fields).\\n"
        "- If evidence is suggestive but not direct, keep value null/[] and add candidate_pages.\\n"
        "- Never guess.\\n"
        "- evidence_paragraph_ids must reference only provided paragraph IDs.\\n\\n"
        f"{extra_prompt_block}"
        "Return JSON object with keys exactly:\\n"
        "value, status, found_pages, candidate_pages, evidence_paragraph_ids, confidence\\n"
        f"Where value is {value_spec}.\\n\\n"
        f"field_key: {field_key}\\n"
        "Evidence:\\n"
        f"{evidence_text}"
    )


def make_section_router_prompt(
    field_key: str,
    field_label: str,
    sections_text: str
) -> str:
    extra_prompt = FIELD_SPECS.get(field_key, {}).get("extra_prompt")
    extra_prompt_block = ""
    if extra_prompt:
        extra_prompt_block = f"Field-specific instruction:\n- {extra_prompt}\n\n"
    return (
        f"Select the most relevant section IDs for extracting '{field_label}'.\n"
        "Rules:\n"
        "- Use only provided section titles/summaries.\n"
        "- Select between 0 and 5 section IDs.\n"
        "- If uncertain, return an empty list.\n"
        "- Never invent section IDs.\n\n"
        f"{extra_prompt_block}"
        "Return JSON object with keys exactly:\n"
        "section_ids\n\n"
        "field_key: "
        f"{field_key}\n"
        "Available sections:\n"
        f"{sections_text}"
    )

"""
I need a better approach for special_building_notes property. For this one I don't know every possible note I want to take about the property in advance. Agent should be adaptable. However current approach chooses set of paragraphs based on set number of keywords
"""
