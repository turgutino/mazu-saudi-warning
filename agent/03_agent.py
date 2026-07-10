# =============================================================================
# MAZU — Layer 4: Agent orchestration (DeepSeek function calling)
#
# The LLM receives a natural-language question, decides which of the 6
# verified tools to call (forecast_tool, causal_kg_tool, conditions_tool,
# similar_events_tool, region_risk_tool, cap_alert_tool), and composes a
# final answer grounded in their real outputs. The model is instructed to
# NEVER invent a probability or a citation — every number and every
# mechanism/citation in its answer must come from a tool call.
# =============================================================================

import os
import json
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import tools
from openai import OpenAI

HERE = os.path.dirname(os.path.abspath(__file__))
KEY_FILE = os.path.join(HERE, "..", "kg", "causal", ".deepseek_key")

SYSTEM_PROMPT = """You are the MAZU early-warning assistant for Saudi Arabia extreme weather.

You answer questions about flash-flood, heatwave, and dust-storm risk using SIX tools:
  - forecast_tool(city, target_date, hazard): risk probability ON target_date,
    from a trained, verified model (ROC-AUC reported with each call). It
    internally uses the PREVIOUS day's indicators -- always pass the exact
    date the user is asking about as target_date, do NOT shift it yourself.
    It also returns elevation_m and, for mountain cities (Abha, Taif), a
    terrain_note flagging lower confidence in steep terrain -- if terrain_note
    is present, mention that caveat in your answer. It also returns
    impact_context (city population, 2022 census) -- use this ONLY to help the
    user understand the scale of the city at risk (e.g. "Riyadh, pop. ~9.06M"),
    NEVER state or imply a specific number of people who would be affected --
    this system does not model exposure, so that would be fabrication. It also
    returns reflexive_check, an INDEPENDENT cross-check comparing the model's
    probability against a separate rule-based physical detection engine on the
    same day's raw indicators. If consistency is "model_higher_than_detection"
    or "detection_higher_than_model", briefly mention this disagreement and
    its note in your answer -- it is a genuine caveat about confidence, not
    noise to hide.
  - causal_kg_tool(hazard): the physical mechanisms driving a hazard, with
    literature citations where available.
  - conditions_tool(city, date): the actual observed indicator values on that date.
  - similar_events_tool(city, date, hazard): compares that city/date's actual
    indicators against the KG's 6 known real 2025 extreme events (3
    flash_flood, 2 heatwave, 1 dust_storm), returning a
    similarity_pct per event (NOT a probability -- purely descriptive). Use
    this when the user asks "does this look like a known event" or when
    adding historical context would help. Each event's own coordinates are
    its grid-cell MAXIMUM (the storm/heat centroid), which can be tens of km
    from a same-named city's center (event_distance_from_city_km shows this)
    -- a same-city, same-day query can legitimately score LOW similarity to
    its own same-named event if the extreme was hyperlocal; state this
    plainly if it comes up, do not treat it as an error.
  - region_risk_tool(city, date=optional): which hazards a CITY is exposed to
    (city-first, unlike the hazard-first tools above) via the KG's
    at_risk_of/exposed_to edges, e.g. "what should I worry about in Jizan".
    Some hazards a city is at_risk_of (e.g. "coastal") have no trained
    forecast model -- has_forecast_model will be False for those; do not call
    forecast_tool for them. If a hazard's mechanisms_affecting_this_city list
    is empty, say the KG doesn't have a specific city-mechanism link for that
    combination rather than inventing one -- this is a known, disclosed gap
    for a few city/hazard pairs, not an error to hide or guess around.
  - cap_alert_tool(city, target_date, hazard): generates a CAP 1.2 (Common
    Alerting Protocol, the OASIS international standard MAZU's own national
    framework is built on) XML alert message, ready to plug into real
    broadcast/siren/SMS warning infrastructure. Internally calls
    forecast_tool, so only call this when the user specifically wants a
    formal/machine-readable alert message, not for ordinary risk questions.
    If alert_warranted is False, no alert was issued (probability too low for
    this hazard's own threshold) -- say so plainly, do not fabricate an XML
    message. The <status> field is always "Exercise", never "Actual" --
    mention this if the user asks whether the alert is "real" or "live": this
    is a historical-dataset demo, not a live feed.

Rules (strict):
1. NEVER state a probability, indicator value, mechanism name, or citation that
   did not come from a tool result. If you don't have a tool result for something,
   say you don't have that information rather than guessing.
2. For any risk question, call forecast_tool first, then causal_kg_tool to explain
   WHY, and cite the literature source if the mechanism is literature-grounded.
3. If a tool returns an "error" field, relay the error honestly to the user;
   do not silently substitute a guess.
4. Known cities: Jeddah, Mecca, Riyadh, Jizan, Dammam, Taif, Medina, Abha.
   Data only covers 2025-01-01 to 2025-12-31; forecast_tool needs the previous
   day's data so the earliest usable target_date is 2025-01-02. This is a
   historical-data research demo, not a live/real-time system -- disclose this
   if asked about "today".
5. Be concise. State the probability, the model's verified ROC-AUC, the driving
   mechanism(s), and (if available) the specific literature citation with a
   short quote. End with a brief, sensible recommendation."""

TOOL_SCHEMAS = [
    {
        "type": "function",
        "function": {
            "name": "forecast_tool",
            "description": "Predict hazard risk probability ON target_date for a city (internally uses the previous day's indicators).",
            "parameters": {
                "type": "object",
                "properties": {
                    "city": {"type": "string", "enum": list(tools.CITIES.keys())},
                    "target_date": {"type": "string", "description": "YYYY-MM-DD, the exact date whose risk is being forecast (2025 only)"},
                    "hazard": {"type": "string", "enum": ["heatwave", "flash_flood", "dust_storm"]},
                },
                "required": ["city", "target_date", "hazard"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "causal_kg_tool",
            "description": "Get the physical mechanisms driving a hazard and their literature citations.",
            "parameters": {
                "type": "object",
                "properties": {"hazard": {"type": "string", "enum": ["heatwave", "flash_flood", "dust_storm"]}},
                "required": ["hazard"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "conditions_tool",
            "description": "Get today's actual observed raw indicator values for a city.",
            "parameters": {
                "type": "object",
                "properties": {
                    "city": {"type": "string", "enum": list(tools.CITIES.keys())},
                    "date": {"type": "string", "description": "YYYY-MM-DD (2025 only)"},
                },
                "required": ["city", "date"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "similar_events_tool",
            "description": "Compare a city/date's indicators against the KG's 5 known real 2025 extreme events, ranked by descriptive similarity (not a probability).",
            "parameters": {
                "type": "object",
                "properties": {
                    "city": {"type": "string", "enum": list(tools.CITIES.keys())},
                    "date": {"type": "string", "description": "YYYY-MM-DD (2025 only)"},
                    "hazard": {"type": "string", "enum": ["heatwave", "flash_flood", "dust_storm"]},
                },
                "required": ["city", "date", "hazard"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "region_risk_tool",
            "description": "Which hazards a city is exposed to and their driving mechanisms (city-first query, via the KG); optionally attaches live forecast probabilities if a date is given.",
            "parameters": {
                "type": "object",
                "properties": {
                    "city": {"type": "string", "enum": list(tools.CITIES.keys())},
                    "date": {"type": "string", "description": "optional YYYY-MM-DD (2025 only) -- if given, attaches forecast probabilities"},
                },
                "required": ["city"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "cap_alert_tool",
            "description": "Generate a formal CAP 1.2 (Common Alerting Protocol) XML alert message for a hazard forecast, suitable for real broadcast/siren/SMS infrastructure. Only call when the user explicitly wants a formal/machine-readable alert, not for ordinary risk questions.",
            "parameters": {
                "type": "object",
                "properties": {
                    "city": {"type": "string", "enum": list(tools.CITIES.keys())},
                    "target_date": {"type": "string", "description": "YYYY-MM-DD, the exact date whose risk is being forecast (2025 only)"},
                    "hazard": {"type": "string", "enum": ["heatwave", "flash_flood", "dust_storm"]},
                },
                "required": ["city", "target_date", "hazard"],
            },
        },
    },
]

TOOL_FUNCS = {
    "forecast_tool": tools.forecast_tool,
    "causal_kg_tool": tools.causal_kg_tool,
    "conditions_tool": tools.conditions_tool,
    "similar_events_tool": tools.similar_events_tool,
    "region_risk_tool": tools.region_risk_tool,
    "cap_alert_tool": tools.cap_alert_tool,
}


def get_client():
    with open(KEY_FILE) as f:
        key = f.read().strip()
    return OpenAI(api_key=key, base_url="https://api.deepseek.com")


def ask(question, client=None, max_turns=6, verbose=True,
        system_prompt=None, tool_schemas=None, tool_funcs=None):
    """Run the agent loop for one question. Returns (final_answer, trace).

    system_prompt/tool_schemas/tool_funcs are overridable (default to the
    module-level versions) so that ablation experiments (e.g. running the
    agent with causal_kg_tool removed) can reuse this same loop rather than
    duplicating it -- see 05_ablation_test.py.
    """
    client = client or get_client()
    system_prompt = system_prompt if system_prompt is not None else SYSTEM_PROMPT
    tool_schemas = tool_schemas if tool_schemas is not None else TOOL_SCHEMAS
    tool_funcs = tool_funcs if tool_funcs is not None else TOOL_FUNCS
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": question},
    ]
    trace = []

    for turn in range(max_turns):
        resp = client.chat.completions.create(
            model="deepseek-chat", messages=messages,
            tools=tool_schemas, tool_choice="auto", temperature=0.0,
        )
        msg = resp.choices[0].message
        messages.append(msg.model_dump(exclude_none=True))

        if not msg.tool_calls:
            if verbose:
                print(f"[final answer after {turn} tool call round(s)]")
            return msg.content, trace

        for tc in msg.tool_calls:
            fname = tc.function.name
            try:
                fargs = json.loads(tc.function.arguments)
            except Exception as e:
                fargs = {}
            if verbose:
                print(f"  [tool call] {fname}({fargs})")
            result = tool_funcs[fname](**fargs)
            trace.append({"tool": fname, "args": fargs, "result": result})
            messages.append({
                "role": "tool", "tool_call_id": tc.id,
                "content": json.dumps(result, default=str),
            })

    return "[agent exceeded max tool-call turns without a final answer]", trace


if __name__ == "__main__":
    q = sys.argv[1] if len(sys.argv) > 1 else \
        "What is the flash-flood risk in Jizan for 2025-08-23, and why?"
    print(f"Q: {q}\n")
    answer, trace = ask(q)
    print(f"\nA: {answer}")
