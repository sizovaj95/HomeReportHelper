from enum import StrEnum


class FieldGroup(StrEnum):
    PROPERTY_OVERVIEW = "Property Overview"
    ENERGY_UTILITIES = "Energy & Utilities"
    BUILDING_FABRIC = "Building Fabric"
    CONDITION_WORKS = "Condition & Works"
    COSTS = "Costs"
    TITLE_SITE_CONSTRAINTS = "Title / Site Constraints"


FIELD_GROUP_ORDER: tuple[FieldGroup, ...] = (
    FieldGroup.PROPERTY_OVERVIEW,
    FieldGroup.ENERGY_UTILITIES,
    FieldGroup.BUILDING_FABRIC,
    FieldGroup.CONDITION_WORKS,
    FieldGroup.COSTS,
    FieldGroup.TITLE_SITE_CONSTRAINTS,
)


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
        "ui_group": FieldGroup.PROPERTY_OVERVIEW,
        "queries": ["property address", "address", "location"],
        "extra_prompt": None,
    },
    "property_age": {
        "label": "Property Age",
        "is_list": False,
        "ui_group": FieldGroup.PROPERTY_OVERVIEW,
        "queries": ["property age", "built", "construction date", "age of property"],
        "extra_prompt": (
            "Extract either the property's age in years or the year/date of construction, whichever is explicitly stated. "
            "Prefer an explicit age value if given as age; otherwise return the stated construction year/date. "
            "Do not infer age from the current year unless the report explicitly states it."
        ),
    },
    "property_area": {
        "label": "Property Area",
        "is_list": False,
        "ui_group": FieldGroup.PROPERTY_OVERVIEW,
        "queries": ["area", "floor area", "sq m", "square metres", "m2", "ft2"],
        "extra_prompt": (
            "Extract property area exactly as stated, including units used in the report "
            "(e.g., sq m, m2, ft2)."
        ),
    },
    "property_epc": {
        "label": "Property EPC",
        "is_list": False,
        "ui_group": FieldGroup.PROPERTY_OVERVIEW,
        "queries": ["EPC", "energy rating", "energy performance"],
        "extra_prompt": (
            "Extract the property's EPC rating result/band, not the heading/title of the EPC section. "
            "Prefer the explicit band letter and score if shown (for example: 'Band C (76)' or 'C (76)'). "
            "Do not return generic labels like 'Energy Performance Certificate'. "
            "If no explicit EPC band/result for the property is stated in the evidence, return not_found."
        ),
    },
    "council_tax_code": {
        "label": "Council Tax Code",
        "is_list": False,
        "ui_group": FieldGroup.PROPERTY_OVERVIEW,
        "queries": ["council tax", "tax band", "band"],
        "extra_prompt": None,
    },
    "construction_firm": {
        "label": "Construction Firm",
        "is_list": False,
        "ui_group": FieldGroup.PROPERTY_OVERVIEW,
        "queries": ["builder", "developer", "constructed by", "construction firm", "built by"],
        "extra_prompt": (
            "Extract the builder/developer/construction firm name only if explicitly stated. "
            "Do not infer from estate/development name or address."
        ),
    },
    "recommended_efficiency_measures": {
        "label": "Recommended Measures to Improve Efficiency",
        "is_list": True,
        "ui_group": FieldGroup.ENERGY_UTILITIES,
        "queries": ["recommended measures", "efficiency improvements", "energy savings"],
        "extra_prompt": (
            "Only include concrete recommended measures/actions for the property. "
            "Do not include generic narrative text."
        ),
    },
    "window_glazing": {
        "label": "Window Glazing",
        "is_list": False,
        "ui_group": FieldGroup.BUILDING_FABRIC,
        "queries": ["double glazed", "single glazed", "glazing"],
        "extra_prompt": (
            "Extract only glazing type explicitly stated (e.g., double glazed/single glazed/mixed). "
            "If not explicit, return not_found."
        ),
    },
    "external_walls_material": {
        "label": "External Walls Material",
        "is_list": False,
        "ui_group": FieldGroup.BUILDING_FABRIC,
        "queries": ["external walls", "wall construction", "outer walls", "masonry", "brick", "stone"],
        "extra_prompt": (
            "Extract only what external walls are made of, if explicitly stated."
        ),
    },
    "internal_walls_material": {
        "label": "Internal Walls Material",
        "is_list": False,
        "ui_group": FieldGroup.BUILDING_FABRIC,
        "queries": ["internal walls", "partition walls", "wall lining", "plasterboard"],
        "extra_prompt": (
            "Extract only what internal walls are made of, if explicitly stated."
        ),
    },
    "roof_material_and_condition": {
        "label": "Roof Material and Condition",
        "is_list": False,
        "ui_group": FieldGroup.BUILDING_FABRIC,
        "queries": ["roof", "roof covering", "slates", "tiles", "flat roof", "roof condition"],
        "extra_prompt": (
            "Extract roof material/covering and condition only if explicitly stated. "
            "If both are stated, combine them concisely. If only one is stated, return only that."
        ),
    },
    "gas_and_boiler_notes": {
        "label": "Gas and Boiler Notes",
        "is_list": False,
        "ui_group": FieldGroup.ENERGY_UTILITIES,
        "queries": ["gas", "boiler", "gas supply", "heating system", "gas safety"],
        "extra_prompt": (
            "Extract only property-specific gas/heating facts explicitly stated in evidence "
            "(for example: boiler brand/model, boiler type/fuel, boiler location, stated age/installation year, "
            "explicit property-specific defects or compliance issues). "
            "Exclude generic disclaimers and boilerplate such as: visual inspection limitations, assumptions, "
            "recommendations for regular servicing/testing, and general regulatory guidance not specific to this property."
        ),
    },
    "electricity_notes": {
        "label": "Electricity Notes",
        "is_list": False,
        "ui_group": FieldGroup.ENERGY_UTILITIES,
        "queries": ["electricity", "electrical", "consumer unit", "wiring", "electrical safety"],
        "extra_prompt": (
            "Extract only property-specific electrical facts explicitly stated in evidence "
            "(for example: consumer unit type/material/location, visible wiring type, mains supply notes, "
            "explicit property-specific defects or compliance concerns). "
            "Exclude generic recommendations and boilerplate such as periodic test advice, change-of-ownership guidance, "
            "inspection limitations, assumptions, and general standards commentary not specific to this property."
        ),
    },
    "potential_problems": {
        "label": "Potential Problems",
        "is_list": True,
        "ui_group": FieldGroup.CONDITION_WORKS,
        "queries": ["defects", "repairs", "problems", "replacement needed", "condition", "leak",
        "flooding", "damp", "rot", "structural issues", "termites"],
        "extra_prompt": (
            "Include only concrete property issues/defects/repair needs. "
            "Avoid generic disclaimers or process-related text."
        ),
    },
    "renovations_done_and_when": {
        "label": "Renovations Done and When",
        "is_list": True,
        "ui_group": FieldGroup.CONDITION_WORKS,
        "queries": ["renovation", "improvement", "replacement", "refurbishment", "installed", "upgraded", "year"],
        "extra_prompt": (
            "Include only completed renovation/improvement/replacement works explicitly stated. "
            "Include timing (date/year) only when explicitly stated. "
            "Exclude recommended or planned works unless clearly described as already completed."
        ),
    },
    "additional_costs": {
        "label": "Additional Costs",
        "is_list": True,
        "ui_group": FieldGroup.COSTS,
        "queries": ["factor fees", "service charges", "additional costs", "charges"],
        "extra_prompt": (
            "Include only concrete property-related costs payable by owner/occupier (e.g., factor fees, "
            "service/maintenance charges, communal charges) when explicitly stated. "
            "If a property-related charge/fee is explicitly mentioned but no amount is given, still include it (for example, 'factor fees mentioned, amount not stated'). "
            "Exclude survey/report/conveyancing/inspection/cancellation/administrative fees and contractual boilerplate. "
            "Return not_found only if no property-related charge/fee is explicitly mentioned at all."
        ),
    },
    "former_mining_site": {
        "label": "Former Mining Site",
        "is_list": False,
        "ui_group": FieldGroup.TITLE_SITE_CONSTRAINTS,
        "queries": ["former mine", "mining", "coal mining", "mining risk", "mine workings"],
        "extra_prompt": (
            "Determine only whether the property/site is explicitly stated to be on former mining land, affected by mining, "
            "or subject to mining-related risk/search notes. "
            "If explicitly present, return a concise value like 'Yes - <explicit note>'. "
            "If explicitly stated absent/negative, return 'No' or 'No - <explicit note>'. "
            "If not explicitly stated, return not_found. Never infer from silence."
        ),
    },
    "right_of_way_or_servitude": {
        "label": "Right of Way or Servitude",
        "is_list": False,
        "ui_group": FieldGroup.TITLE_SITE_CONSTRAINTS,
        "queries": ["right of way", "servitude", "servitudes", "access rights", "wayleave"],
        "extra_prompt": (
            "Determine only whether any explicit right of way, servitude, or similar access/title burden/right is stated. "
            "Exclude notes that refer only to communal/shared area access and do not create a property-specific right or burden. "
            "If explicitly present, return a concise value like 'Yes - <explicit note>'. "
            "If explicitly stated absent/negative, return 'No' or 'No - <explicit note>'. "
            "If not explicitly stated, return not_found. Never infer from silence."
        ),
    },
    "listed_or_protected_building": {
        "label": "Listed or Protected Building",
        "is_list": False,
        "ui_group": FieldGroup.TITLE_SITE_CONSTRAINTS,
        "queries": ["listed building", "protected building", "conservation area", "historic designation"],
        "extra_prompt": (
            "Determine only whether the property/building is explicitly stated to be listed, protected, or under conservation/historic designation constraints. "
            "If explicitly present, return a concise value like 'Yes - <explicit note>'. "
            "If explicitly stated absent/negative, return 'No' or 'No - <explicit note>'. "
            "If not explicitly stated, return not_found. Never infer from silence."
        ),
    },
    "market_value": {
        "label": "Market Value",
        "is_list": False,
        "ui_group": FieldGroup.PROPERTY_OVERVIEW,
        "queries": ["market value", "valuation", "value"],
        "extra_prompt": (
            "Extract the market value/valuation amount only if explicitly stated for the property."
        ),
    },
}


def _infer_ui_order_from_group(field_specs: dict) -> None:
    """Assign deterministic ui_order per group using declaration order."""
    counters: dict[str, int] = {}
    for _, spec in field_specs.items():
        group = str(spec.get("ui_group", FieldGroup.PROPERTY_OVERVIEW))
        counters[group] = counters.get(group, 0) + 10
        spec["ui_order"] = counters[group]


_infer_ui_order_from_group(FIELD_SPECS)


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
    sections_text: str,
    query_hints: list[str] | None = None,
) -> str:
    extra_prompt = FIELD_SPECS.get(field_key, {}).get("extra_prompt")
    extra_prompt_block = ""
    if extra_prompt:
        extra_prompt_block = f"Field-specific instruction:\n- {extra_prompt}\n\n"
    query_hints_block = ""
    if query_hints:
        rendered_hints = "\n".join(f"- {hint}" for hint in query_hints if str(hint).strip())
        if rendered_hints:
            query_hints_block = f"Search query hints:\n{rendered_hints}\n\n"
    return (
        f"Select the most relevant section IDs for extracting '{field_label}'.\n"
        "Rules:\n"
        "- Use only provided section titles/summaries.\n"
        "- Select between 0 and 5 section IDs.\n"
        "- If uncertain, return an empty list.\n"
        "- Never invent section IDs.\n\n"
        f"{extra_prompt_block}"
        f"{query_hints_block}"
        "Return JSON object with keys exactly:\n"
        "section_ids\n\n"
        "field_key: "
        f"{field_key}\n"
        "Available sections:\n"
        f"{sections_text}"
    )
