import sys, os
sys.path.insert(0, r"C:\Users\Turqut\Desktop\Competation\mazu-system\agent")
from render_multiday import render_multiday
from render_gif import build_gif
import tools

HAIL = (27.52, 41.70)
BURAIDAH = (26.33, 43.97)
HAIL_BURAIDAH_MARKERS = {"Hail": HAIL, "Buraidah": BURAIDAH}

OUT_DIR = "regenerated"
os.makedirs(OUT_DIR, exist_ok=True)

# (out_name, dates(4), hazard, title, highlight, extra_markers)
STATIC = [
    ("map_dammam_dust", ["2025-05-16", "2025-05-17", "2025-05-18", "2025-05-19"], "dust_storm",
     "Dammam dust storm, 16-19 May 2025 -- national event, city-level peak on 17th", "Dammam", None),
    ("map_dammam_heat", ["2025-06-11", "2025-06-12", "2025-06-13", "2025-06-14"], "heatwave",
     "Dammam heatwave, 11-14 June 2025 -- model crosses threshold, rule-engine ground-truth stays below 45C locally", "Dammam", None),
    ("map_haboob_dust", ["2025-05-03", "2025-05-04", "2025-05-05", "2025-05-06"], "dust_storm",
     "Haboob dust storm, 4-5 May 2025 -- national event, outside 8-city coverage", None, None),
    ("map_junjul_dust", ["2025-06-30", "2025-07-01", "2025-07-02", "2025-07-03", "2025-07-04", "2025-07-05"], "dust_storm",
     "Dust storm, 30 Jun-5 Jul 2025 -- strong hit at Riyadh and Dammam", ["Riyadh", "Dammam"], None),
    ("map_heatwave2_miss", ["2025-06-29", "2025-06-30", "2025-07-01", "2025-07-02"], "heatwave",
     "Heatwave 2nd wave, 28 Jun-5 Jul 2025 -- model miss at Dammam", "Dammam", None),
    ("map_heatwave3", ["2025-07-21", "2025-07-22", "2025-07-23", "2025-07-24"], "heatwave",
     "Heatwave 3rd wave, 20-27 Jul 2025 -- named cities miss, independent hit at Mecca", ["Dammam", "Riyadh", "Mecca"], None),
    ("map_heatwave4", ["2025-08-02", "2025-08-03", "2025-08-04", "2025-08-05"], "heatwave",
     "Heatwave 4th wave, 29 Jul-5 Aug 2025 -- partial hit, Mecca independent hit", ["Dammam", "Mecca"], None),
    ("map_flood_jan", ["2025-01-05", "2025-01-06", "2025-01-07", "2025-01-08"], "flash_flood",
     "Flood, 6-7 Jan 2025 -- inconclusive, data gap + within training period", "Mecca", None),
    ("map_flood_mar", ["2025-03-05", "2025-03-06", "2025-03-07", "2025-03-08"], "flash_flood",
     "Flood, 6-7 Mar 2025 -- real score stays 0.0 at Hail/Buraidah, but model reaches 0.94-0.98 there (not caught in the written report)", None, HAIL_BURAIDAH_MARKERS),
    ("map_flood_taif", ["2025-08-12", "2025-08-13", "2025-08-14", "2025-08-15"], "flash_flood",
     "Flood, 14 Aug 2025 (Taif) -- model/real disagreement", "Taif", None),
    ("map_flood_aug2728", ["2025-08-26", "2025-08-27", "2025-08-28", "2025-08-29"], "flash_flood",
     "Flood, 27-28 Aug 2025 (Asir/Jizan) -- mixed result", ["Abha", "Jizan"], None),
]

# (out_name, dates(6), hazard, label, highlight, extra_markers)
GIFS = [
    ("dammam_dust", ["2025-05-15", "2025-05-16", "2025-05-17", "2025-05-18", "2025-05-19", "2025-05-20"],
     "dust_storm", "Dammam dust storm", "Dammam", None),
    ("dammam_heat", ["2025-06-10", "2025-06-11", "2025-06-12", "2025-06-13", "2025-06-14", "2025-06-15"],
     "heatwave", "Dammam heatwave", "Dammam", None),
    ("jeddah_flood", ["2025-12-07", "2025-12-08", "2025-12-09", "2025-12-10", "2025-12-11", "2025-12-12"],
     "flash_flood", "Jeddah flood", "Jeddah", None),
    ("haboob_dust", ["2025-05-02", "2025-05-03", "2025-05-04", "2025-05-05", "2025-05-06", "2025-05-07"],
     "dust_storm", "Haboob dust storm", None, None),
    ("junjul_dust", ["2025-06-30", "2025-07-01", "2025-07-02", "2025-07-03", "2025-07-04", "2025-07-05"],
     "dust_storm", "Dust storm (Riyadh/Dammam)", ["Riyadh", "Dammam"], None),
    ("heatwave2_miss", ["2025-06-28", "2025-06-29", "2025-06-30", "2025-07-01", "2025-07-02", "2025-07-03"],
     "heatwave", "Heatwave 2nd wave", "Dammam", None),
    ("heatwave3", ["2025-07-20", "2025-07-21", "2025-07-22", "2025-07-23", "2025-07-24", "2025-07-25"],
     "heatwave", "Heatwave 3rd wave", ["Dammam", "Riyadh", "Mecca"], None),
    ("heatwave4", ["2025-08-01", "2025-08-02", "2025-08-03", "2025-08-04", "2025-08-05", "2025-08-06"],
     "heatwave", "Heatwave 4th wave", ["Dammam", "Mecca"], None),
    ("flood_jan", ["2025-01-04", "2025-01-05", "2025-01-06", "2025-01-07", "2025-01-08", "2025-01-09"],
     "flash_flood", "Flood (Mecca)", "Mecca", None),
    ("flood_mar", ["2025-03-04", "2025-03-05", "2025-03-06", "2025-03-07", "2025-03-08", "2025-03-09"],
     "flash_flood", "Flood (Hail/Buraidah)", None, HAIL_BURAIDAH_MARKERS),
    ("flood_taif", ["2025-08-11", "2025-08-12", "2025-08-13", "2025-08-14", "2025-08-15", "2025-08-16"],
     "flash_flood", "Flood (Taif)", "Taif", None),
    ("flood_aug2728", ["2025-08-25", "2025-08-26", "2025-08-27", "2025-08-28", "2025-08-29", "2025-08-30"],
     "flash_flood", "Flood (Asir/Jizan)", ["Abha", "Jizan"], None),
]


def main():
    import sys as _sys
    skip_static = "--gifs-only" in _sys.argv
    skip_gifs = "--static-only" in _sys.argv
    only_from = None
    for a in _sys.argv:
        if a.startswith("--from="):
            only_from = a.split("=", 1)[1]

    if not skip_static:
        print("=== STATIC MULTI-DAY MAPS ===")
        for name, dates, hazard, title, hl, extra in STATIC:
            out = os.path.join(OUT_DIR, f"{name}.png")
            render_multiday(dates, hazard, out, title, highlight_city=hl, extra_markers=extra)

    if skip_gifs:
        print("\nDONE (static only):", len(STATIC), "static maps ->", OUT_DIR)
        return

    print("\n=== GIFS ===")
    started = only_from is None
    for name, dates, hazard, label, hl, extra in GIFS:
        if not started:
            if name == only_from:
                started = True
            else:
                continue
        out = os.path.join(OUT_DIR, f"{name}.gif")
        build_gif(dates, hazard, label, hl, out, extra_markers=extra)

    print("\nDONE:", len(STATIC), "static maps +", len(GIFS), "gifs ->", OUT_DIR)


if __name__ == "__main__":
    main()
