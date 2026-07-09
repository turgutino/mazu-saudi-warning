# =============================================================================
# MAZU — Layer 4: build a static showcase page of real agent transcripts
#
# Why static, not a live chat: GitHub Pages serves static files only. A live
# chat would require either exposing the DeepSeek API key in client-side JS
# (a real security risk -- anyone could steal and abuse it) or standing up a
# hosted backend server (out of scope here). Instead, this page shows REAL,
# verified transcripts from 03_agent.py -- the exact tool calls and results
# are included so a reader can check the answer traces back to real tool
# output, not a scripted fake.
# =============================================================================
import os
import json
import html as htmllib

HERE = os.path.dirname(os.path.abspath(__file__))
TRANSCRIPTS = os.path.join(HERE, "example_transcripts.json")
OUT_HTML = os.path.join(HERE, "..", "dashboard", "agent_view.html")

with open(TRANSCRIPTS, encoding="utf-8") as f:
    examples = json.load(f)


def render_trace(trace):
    parts = []
    for t in trace:
        args = ", ".join(f"{k}={v!r}" for k, v in t["args"].items())
        result_json = json.dumps(t["result"], indent=2, default=str, ensure_ascii=False)
        parts.append(f'''<details class="toolcall">
  <summary><code>{htmllib.escape(t["tool"])}({htmllib.escape(args)})</code></summary>
  <pre>{htmllib.escape(result_json)}</pre>
</details>''')
    return "\n".join(parts)


def render_answer(answer):
    # minimal markdown-ish rendering: bold, headers, line breaks
    import re
    a = htmllib.escape(answer)
    a = re.sub(r"\*\*(.+?)\*\*", r"<b>\1</b>", a)
    a = re.sub(r"^### (.+)$", r"<h4>\1</h4>", a, flags=re.M)
    a = re.sub(r"^## (.+)$", r"<h3>\1</h3>", a, flags=re.M)
    a = re.sub(r"^---$", r"<hr>", a, flags=re.M)
    a = re.sub(r"^\d+\.\s+(.+)$", r"<li>\1</li>", a, flags=re.M)
    a = a.replace("\n\n", "</p><p>")
    return f"<p>{a}</p>"


cards = []
for e in examples:
    cards.append(f'''<div class="card">
  <div class="q">{htmllib.escape(e["question"])}</div>
  <div class="tools">{render_trace(e["trace"])}</div>
  <div class="answer">{render_answer(e["answer"])}</div>
</div>''')

html_out = f"""<!doctype html>
<html lang="en"><head>
<meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>MAZU · Agent — worked examples</title>
<style>
  :root{{--bg:#0E1B2A;--panel:#16283C;--line:#243B54;--txt:#E6EEF6;--mut:#8AA0B4;--cyan:#2BC8E2;--green:#5EE8A5}}
  *{{box-sizing:border-box;margin:0;padding:0}}
  body{{font-family:Segoe UI,Arial,sans-serif;background:var(--bg);color:var(--txt);line-height:1.6}}
  .wrap{{max-width:900px;margin:0 auto;padding:40px 24px}}
  header{{margin-bottom:30px}}
  header .kicker{{color:var(--cyan);letter-spacing:.15em;font-size:12px;font-weight:700;text-transform:uppercase}}
  header h1{{font-size:28px;margin:8px 0}}
  header p{{color:var(--mut);font-size:14px;max-width:700px}}
  header p b{{color:var(--txt)}}
  a.back{{color:var(--cyan);text-decoration:none;font-size:13px}}
  .card{{background:var(--panel);border:1px solid var(--line);border-radius:14px;padding:22px;margin-bottom:24px}}
  .q{{font-size:17px;font-weight:700;color:var(--cyan);margin-bottom:14px}}
  .q:before{{content:"Q  ";color:var(--mut);font-weight:400}}
  .tools{{margin-bottom:16px}}
  details.toolcall{{background:var(--bg);border:1px solid var(--line);border-radius:8px;padding:8px 12px;margin-bottom:6px}}
  details.toolcall summary{{cursor:pointer;font-size:12px;color:var(--green);font-family:Consolas,monospace}}
  details.toolcall pre{{font-size:11px;color:var(--mut);white-space:pre-wrap;margin-top:8px;font-family:Consolas,monospace}}
  .answer{{font-size:14px;color:var(--txt);border-top:1px solid var(--line);padding-top:14px}}
  .answer h3,.answer h4{{color:var(--cyan);margin:12px 0 6px}}
  .answer hr{{border:none;border-top:1px solid var(--line);margin:12px 0}}
  .answer li{{margin-left:20px;color:var(--txt)}}
  .answer b{{color:var(--green)}}
  .note{{background:rgba(94,232,165,0.08);border-left:2px solid var(--green);border-radius:6px;padding:12px 16px;font-size:13px;color:var(--mut);margin-bottom:26px}}
</style></head>
<body><div class="wrap">
<header>
  <a class="back" href="index.html">&larr; back</a>
  <div class="kicker">Layer 4 &middot; Agent</div>
  <h1>MAZU Agent — worked examples</h1>
  <p>Real transcripts from the DeepSeek function-calling agent (<code>agent/03_agent.py</code>). Every number and citation below came from an actual tool call — click a tool-call line to see its exact input and output.</p>
</header>
<div class="note">Static page, not a live chat: GitHub Pages cannot run the agent's Python backend, and embedding the API key in client-side JavaScript would let anyone steal and abuse it. These are real, verified runs of the agent shown as evidence it works — see <code>agent/LAYER4_REPORT.md</code> for the full test suite (27 tool tests + 4 end-to-end scenarios, all passing).</div>
{''.join(cards)}
</div></body></html>"""

os.makedirs(os.path.dirname(OUT_HTML), exist_ok=True)
with open(OUT_HTML, "w", encoding="utf-8") as f:
    f.write(html_out)
print(f"[SAVED] {OUT_HTML}  ({len(examples)} examples, {round(len(html_out)/1024,1)} KB)")
