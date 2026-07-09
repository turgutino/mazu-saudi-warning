# =============================================================================
# MAZU — Layer 4: Agent orchestration (DeepSeek function calling)
#
# The LLM receives a natural-language question, decides which of the 3
# verified tools to call (forecast_tool, causal_kg_tool, conditions_tool),
# and composes a final answer grounded in their real outputs. The model is
# instructed to NEVER invent a probability or a citation — every number and
# every mechanism/citation in its answer must come from a tool call.
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

You answer questions about flash-flood and heatwave risk using THREE tools:
  - forecast_tool(city, target_date, hazard): risk probability ON target_date,
    from a trained, verified model (ROC-AUC reported with each call). It
    internally uses the PREVIOUS day's indicators -- always pass the exact
    date the user is asking about as target_date, do NOT shift it yourself.
  - causal_kg_tool(hazard): the physical mechanisms driving a hazard, with
    literature citations where available.
  - conditions_tool(city, date): the actual observed indicator values on that date.

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
                    "hazard": {"type": "string", "enum": ["heatwave", "flash_flood"]},
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
                "properties": {"hazard": {"type": "string", "enum": ["heatwave", "flash_flood"]}},
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
]

TOOL_FUNCS = {
    "forecast_tool": tools.forecast_tool,
    "causal_kg_tool": tools.causal_kg_tool,
    "conditions_tool": tools.conditions_tool,
}


def get_client():
    with open(KEY_FILE) as f:
        key = f.read().strip()
    return OpenAI(api_key=key, base_url="https://api.deepseek.com")


def ask(question, client=None, max_turns=6, verbose=True):
    """Run the agent loop for one question. Returns (final_answer, trace)."""
    client = client or get_client()
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": question},
    ]
    trace = []

    for turn in range(max_turns):
        resp = client.chat.completions.create(
            model="deepseek-chat", messages=messages,
            tools=TOOL_SCHEMAS, tool_choice="auto", temperature=0.0,
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
            result = TOOL_FUNCS[fname](**fargs)
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
