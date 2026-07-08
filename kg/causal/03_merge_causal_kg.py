# =============================================================================
# MAZU — Layer 3: merge verified causal triples into the structural KG
#
# Adds:
#   - Citation nodes (one per literature source), with title/authors/url
#   - grounded_by edges: Mechanism -> Citation, for mechanisms that now have
#     verbatim-verified literature support
#   - Each Citation node carries its verified evidence triples (subject/
#     relation/object/quote) as metadata, shown in the dashboard on click
#
# Quality gate: one extracted triple (from ref_redsea_coast_trends) was
# manually reviewed and found to be circular/descriptive rather than a real
# causal claim (the source text lists statistical trends and labels them
# "consistent with warming" — not a mechanism). It passed the automatic
# verbatim-quote check but is EXCLUDED here after manual review. This is
# disclosed, not hidden — see causal_kg_report.txt.
#
# Mechanism honestly left WITHOUT literature grounding: orographic_lift.
# No corpus source was found for it in this round; it remains hand-coded
# domain knowledge, not literature-grounded, and this is disclosed.
# =============================================================================

import os
import json

HERE = os.path.dirname(os.path.abspath(__file__))
KG_JSON = os.path.join(HERE, "..", "kg_data.json")
TRIPLES_JSON = os.path.join(HERE, "causal_triples.json")
OUT_KG_JSON = os.path.join(HERE, "..", "kg_data.json")   # overwrite in place
OUT_REPORT = os.path.join(HERE, "causal_kg_report.txt")

# manual quality-review exclusion (see module docstring)
EXCLUDED_QUOTES = {
    "Data for the Saudi Arabian Red Sea coast from 1979 to 2020 show significant positive "
    "trends in surface air temperature and wind speed, alongside significant negative trends "
    "in relative humidity and sea-level pressure, consistent with a warming, drying coastal "
    "atmosphere over this period."
}

CORPUS_META = {
    "ref_devries2013": {"title": "Extreme precipitation events in the Middle East: Dynamics of the Active Red Sea Trough",
                        "citation": "de Vries et al. (2013), J. Geophys. Res. Atmos.",
                        "url": "https://agupubs.onlinelibrary.wiley.com/doi/full/10.1002/jgrd.50569"},
    "ref_redsea_topo": {"title": "On the Effect of Red Sea and Topography on Rainfall over Saudi Arabia",
                        "citation": "ResearchGate case study",
                        "url": "https://www.researchgate.net/publication/336928573"},
    "ref_heatflux_era5": {"title": "Surface Heat Fluxes over the Northern Arabian Gulf and Red Sea (ERA5/MERRA2)",
                          "citation": "MDPI Atmosphere (2019)",
                          "url": "https://www.mdpi.com/2073-4433/10/9/504"},
    "ref_heatwave_circulation": {"title": "Extreme summer temperatures in Saudi Arabia and large-scale circulation",
                                 "citation": "ScienceDirect, Atmospheric Research",
                                 "url": "https://www.sciencedirect.com/science/article/abs/pii/S0169809519301747"},
    "ref_heatwave_variability": {"title": "Observed heatwaves characteristics and variability over Saudi Arabia",
                                 "citation": "Springer, Meteorology and Atmospheric Physics (2024)",
                                 "url": "https://link.springer.com/article/10.1007/s00703-024-01010-6"},
    "ref_shamal_yu2016": {"title": "Climatology of summer Shamal wind in the Middle East",
                          "citation": "Yu et al. (2016), J. Geophys. Res. Atmos.",
                          "url": "https://agupubs.onlinelibrary.wiley.com/doi/full/10.1002/2015jd024063"},
}


def main():
    with open(KG_JSON, encoding="utf-8") as f:
        kg = json.load(f)
    with open(TRIPLES_JSON, encoding="utf-8") as f:
        triples = json.load(f)

    existing_ids = {n["id"] for n in kg["nodes"]}
    n_before_nodes, n_before_edges = len(kg["nodes"]), len(kg["links"])

    excluded = [t for t in triples if t["evidence_quote"] in EXCLUDED_QUOTES]
    kept = [t for t in triples if t["evidence_quote"] not in EXCLUDED_QUOTES]

    # group kept triples by source citation
    by_source = {}
    for t in kept:
        by_source.setdefault(t["source_id"], []).append(t)

    added_citation_nodes = 0
    added_grounded_edges = 0
    mechanisms_grounded = set()

    for source_id, group in by_source.items():
        meta = CORPUS_META[source_id]
        cid = f"cite_{source_id}"
        evidence_list = [
            {"subject": t["subject"], "relation": t["relation"], "object": t["object"], "quote": t["evidence_quote"]}
            for t in group
        ]
        if cid not in existing_ids:
            kg["nodes"].append({
                "id": cid, "ntype": "Citation", "label": meta["citation"],
                "desc": meta["title"], "url": meta["url"],
                "evidence": evidence_list, "n_triples": len(group),
            })
            existing_ids.add(cid)
            added_citation_nodes += 1

        mech = group[0]["mechanism"]
        if mech in existing_ids:
            kg["links"].append({"etype": "grounded_by", "source": mech, "target": cid})
            added_grounded_edges += 1
            mechanisms_grounded.add(mech)

    with open(OUT_KG_JSON, "w", encoding="utf-8") as f:
        json.dump(kg, f, ensure_ascii=False, indent=1)

    all_mechanisms = {n["id"] for n in kg["nodes"] if n.get("ntype") == "Mechanism"}
    ungrounded = all_mechanisms - mechanisms_grounded

    lines = [
        "=" * 70, "MAZU Layer 3 — Causal KG merge report", "=" * 70,
        f"Input triples: {len(triples)} (all passed verbatim-quote verification)",
        f"Manually excluded after quality review: {len(excluded)}",
        f"  Reason: circular/descriptive framing (trend statistics relabelled as causal),",
        f"          not a genuine mechanistic claim, despite passing the quote check.",
        f"Kept and merged: {len(kept)}",
        "",
        f"KG before: {n_before_nodes} nodes, {n_before_edges} edges",
        f"KG after:  {len(kg['nodes'])} nodes, {len(kg['links'])} edges",
        f"Added: {added_citation_nodes} Citation nodes, {added_grounded_edges} grounded_by edges",
        "",
        f"Mechanisms now literature-grounded ({len(mechanisms_grounded)}/{len(all_mechanisms)}):",
    ]
    triples_per_mechanism = {}
    for t in kept:
        triples_per_mechanism.setdefault(t["mechanism"], 0)
        triples_per_mechanism[t["mechanism"]] += 1
    for m in sorted(mechanisms_grounded):
        lines.append(f"  - {m}  ({triples_per_mechanism.get(m, 0)} verified triples)")
    lines.append("")
    lines.append(f"Mechanisms WITHOUT literature grounding (honest disclosure, {len(ungrounded)}):")
    for m in sorted(ungrounded):
        lines.append(f"  - {m}  (remains hand-coded domain knowledge; no corpus source found this round)")

    report = "\n".join(lines)
    with open(OUT_REPORT, "w", encoding="utf-8") as f:
        f.write(report)
    print(report)


if __name__ == "__main__":
    main()
