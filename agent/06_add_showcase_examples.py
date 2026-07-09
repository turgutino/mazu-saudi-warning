# =============================================================================
# MAZU — Extension: add real, freshly-run agent transcripts to the static
# showcase page (example_transcripts.json), covering today's new tools
# (terrain, reflexive_check, similar_events, dust_storm, region_risk) that
# the original 3 Layer-4 examples predate. Each call is a REAL, live
# DeepSeek run -- not scripted -- captured in full including tool traces.
# =============================================================================
import os
import json
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import importlib
agent_mod = importlib.import_module("03_agent")

HERE = os.path.dirname(os.path.abspath(__file__))
TRANSCRIPTS = os.path.join(HERE, "example_transcripts.json")

NEW_QUESTIONS = [
    "I'm planning a trip to Abha on 2025-08-31. What weather risks should I know about?",
    "Tell me about the extreme rain event in Jizan on 2025-08-23 - how much rain fell?",
    "What is the dust storm risk in Dammam for 2025-07-06, and what physically causes dust storms in Saudi Arabia?",
    "Compare the dust storm risk between Medina and Riyadh for 2025-06-20",
]


def main():
    with open(TRANSCRIPTS, encoding="utf-8") as f:
        examples = json.load(f)
    existing_questions = {e["question"] for e in examples}

    client = agent_mod.get_client()
    added = 0
    for q in NEW_QUESTIONS:
        if q in existing_questions:
            print(f"skip (already present): {q}")
            continue
        print(f"running: {q}")
        answer, trace = agent_mod.ask(q, client=client, verbose=True)
        examples.append({"question": q, "answer": answer, "trace": trace})
        added += 1

    with open(TRANSCRIPTS, "w", encoding="utf-8") as f:
        json.dump(examples, f, indent=2, default=str, ensure_ascii=False)
    print(f"\n[SAVED] {TRANSCRIPTS} -- {len(examples)} total examples ({added} newly added)")


if __name__ == "__main__":
    main()
