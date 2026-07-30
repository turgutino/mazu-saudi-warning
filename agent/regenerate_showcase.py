# =============================================================================
# MAZU — one-off: regenerate ALL example_transcripts.json entries fresh
# against the current system (post-2026-07-30 knowledge-graph update), since
# similar_events_tool's rankings changed once the 12 site-verified events
# were added -- one published example (Abha) was found to now give a
# different "closest match" than what was on the live page. Rather than
# patch just that one, every example is re-run live so the whole showcase
# page is consistent with today's actual tool outputs, same questions and
# order as before.
# =============================================================================
import os
import json
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import importlib
agent_mod = importlib.import_module("03_agent")

HERE = os.path.dirname(os.path.abspath(__file__))
TRANSCRIPTS = os.path.join(HERE, "example_transcripts.json")

with open(TRANSCRIPTS, encoding="utf-8") as f:
    existing = json.load(f)
questions = [e["question"] for e in existing]

client = agent_mod.get_client()
new_examples = []
for q in questions:
    print(f"running: {q}")
    answer, trace = agent_mod.ask(q, client=client, verbose=True)
    new_examples.append({"question": q, "answer": answer, "trace": trace})
    print(f"  -> {len(trace)} tool call(s)")

with open(TRANSCRIPTS, "w", encoding="utf-8") as f:
    json.dump(new_examples, f, indent=2, default=str, ensure_ascii=False)
print(f"\n[SAVED] {TRANSCRIPTS} -- {len(new_examples)} examples, all freshly re-run")
