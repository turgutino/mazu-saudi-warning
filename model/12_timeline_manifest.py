# =============================================================================
# MAZU — Priority-4 item 2: build the JSON manifest the interactive timeline
# page (deploy/interactive_timeline.html) reads. Reuses the already-rendered
# gif_frames/*.png (one real 3-panel map per date, already produced while
# building the site's GIFs -- no new rendering needed for daily-scale
# playback across all 12 events) plus the extended 28-day "hero" frame set
# for Dammam dust storm (see 11_timeline_hero_frames.py) for genuine
# weekly/pentad-scale demonstration.
#
# For "定格灾害事件...底部标注对应真实灾害记录" (freeze on hazard events,
# bottom-labeled with real hazard records), each date is checked against
# the same rule-based DetectionEngine used everywhere else on the site --
# real event flags, not hardcoded guesses.
# =============================================================================
import os, sys, json, shutil, importlib.util

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, r"C:\Users\Turqut\Desktop\Competation\mazu-system\agent")

spec = importlib.util.spec_from_file_location("de", os.path.join(HERE, "01_detection_engine.py"))
de = importlib.util.module_from_spec(spec)
spec.loader.exec_module(de)

GIF_FRAME_DIR = os.path.join(HERE, "gif_frames")
HERO_FRAME_DIR = os.path.join(HERE, "timeline_frames", "dammam_dust_extended")
OUT_IMG_DIR = os.path.join(HERE, "..", "img", "timeline")
os.makedirs(OUT_IMG_DIR, exist_ok=True)

HAIL = (27.52, 41.70)
BURAIDAH = (26.33, 43.97)

CITIES = {"Jeddah": (21.5, 39.2), "Mecca": (21.4, 39.8), "Riyadh": (24.7, 46.7),
          "Jizan": (16.9, 42.6), "Dammam": (26.4, 50.1), "Taif": (21.3, 40.4),
          "Medina": (24.5, 39.6), "Abha": (18.2, 42.5)}

# (key, dates, hazard, label, highlight_coords) -- highlight_coords is a list
# of (lat, lon) points the event is centered on, or None for a genuinely
# national event with no single focal point. Used to scope "real event"
# flagging to THIS event's location rather than any dust/heat cluster
# anywhere in the country (see bug note below).
EVENTS = [
    ("dammam_dust", ["2025-05-15", "2025-05-16", "2025-05-17", "2025-05-18", "2025-05-19", "2025-05-20"],
     "dust_storm", "Dammam dust storm", [CITIES["Dammam"]]),
    ("dammam_heat", ["2025-06-10", "2025-06-11", "2025-06-12", "2025-06-13", "2025-06-14", "2025-06-15"],
     "heatwave", "Dammam heatwave", [CITIES["Dammam"]]),
    ("jeddah_flood", ["2025-12-07", "2025-12-08", "2025-12-09", "2025-12-10", "2025-12-11", "2025-12-12"],
     "flash_flood", "Jeddah flood", [CITIES["Jeddah"]]),
    ("haboob_dust", ["2025-05-02", "2025-05-03", "2025-05-04", "2025-05-05", "2025-05-06", "2025-05-07"],
     "dust_storm", "Haboob dust storm", None),
    ("junjul_dust", ["2025-06-30", "2025-07-01", "2025-07-02", "2025-07-03", "2025-07-04", "2025-07-05"],
     "dust_storm", "Dust storm (Riyadh/Dammam)", [CITIES["Riyadh"], CITIES["Dammam"]]),
    ("heatwave2_miss", ["2025-06-28", "2025-06-29", "2025-06-30", "2025-07-01", "2025-07-02", "2025-07-03"],
     "heatwave", "Heatwave 2nd wave", [CITIES["Dammam"]]),
    ("heatwave3", ["2025-07-20", "2025-07-21", "2025-07-22", "2025-07-23", "2025-07-24", "2025-07-25"],
     "heatwave", "Heatwave 3rd wave", [CITIES["Dammam"], CITIES["Riyadh"], CITIES["Mecca"]]),
    ("heatwave4", ["2025-08-01", "2025-08-02", "2025-08-03", "2025-08-04", "2025-08-05", "2025-08-06"],
     "heatwave", "Heatwave 4th wave", [CITIES["Dammam"], CITIES["Mecca"]]),
    ("flood_jan", ["2025-01-04", "2025-01-05", "2025-01-06", "2025-01-07", "2025-01-08", "2025-01-09"],
     "flash_flood", "Flood (Mecca)", [CITIES["Mecca"]]),
    ("flood_mar", ["2025-03-04", "2025-03-05", "2025-03-06", "2025-03-07", "2025-03-08", "2025-03-09"],
     "flash_flood", "Flood (Hail/Buraidah)", [HAIL, BURAIDAH]),
    ("flood_taif", ["2025-08-11", "2025-08-12", "2025-08-13", "2025-08-14", "2025-08-15", "2025-08-16"],
     "flash_flood", "Flood (Taif)", [CITIES["Taif"]]),
    ("flood_aug2728", ["2025-08-25", "2025-08-26", "2025-08-27", "2025-08-28", "2025-08-29", "2025-08-30"],
     "flash_flood", "Flood (Asir/Jizan)", [CITIES["Abha"], CITIES["Jizan"]]),
]

HERO_KEY = "dammam_dust"
HERO_START = "2025-05-05"
HERO_END = "2025-06-01"


def safe_label(label):
    return label.replace(" ", "_").replace("/", "-").replace("\\", "-")


NEAR_DEG = 2.0  # degrees lat/lon -- matches the ~2.2 deg zoom radius used elsewhere


def real_event_dates(eng, hazard, dates, focus_points):
    """A date is flagged 'real event' only if a detected cluster's peak
    falls near THIS event's own location(s) (focus_points), not just
    anywhere in the country. Bug found while testing the timeline: without
    this filter, dust_storm/heatwave cluster somewhere nationally on
    nearly every day (confirmed: 25/28 days for the unfiltered Dammam
    dust-storm window), which made "freeze on hazard event" nearly
    meaningless -- almost every frame would freeze. focus_points=None
    (genuinely national events, e.g. Haboob) keeps the unfiltered
    national-level check, since there's no single location to scope to."""
    flagged = {}
    for d in dates:
        try:
            events = eng.detect(d, hazard)
        except (KeyError, IndexError, ValueError):
            events = []
        if focus_points is None:
            flagged[d] = len(events) > 0
        else:
            flagged[d] = any(
                min(((e["lat"] - fla) ** 2 + (e["lon"] - flo) ** 2) ** 0.5 for fla, flo in focus_points) <= NEAR_DEG
                for e in events
            )
    return flagged


def daterange(start, end):
    import datetime
    s = datetime.date.fromisoformat(start)
    e = datetime.date.fromisoformat(end)
    out = []
    d = s
    while d <= e:
        out.append(d.isoformat())
        d += datetime.timedelta(days=1)
    return out


def main():
    eng = de.DetectionEngine()
    manifest = {"events": []}

    for key, dates, hazard, label, focus_points in EVENTS:
        is_hero = key == HERO_KEY
        frame_dates = daterange(HERO_START, HERO_END) if is_hero else dates
        src_dir = HERO_FRAME_DIR if is_hero else GIF_FRAME_DIR
        sl = safe_label(label)

        out_dir = os.path.join(OUT_IMG_DIR, key)
        os.makedirs(out_dir, exist_ok=True)

        flags = real_event_dates(eng, hazard, frame_dates, focus_points)
        frames = []
        for d in frame_dates:
            if is_hero:
                src = os.path.join(src_dir, f"{d}.png")
            else:
                src = os.path.join(src_dir, f"{hazard}_{sl}_{d}.png")
            if not os.path.exists(src):
                print(f"  WARNING missing frame: {src}")
                continue
            dst_name = f"{d}.png"
            dst = os.path.join(out_dir, dst_name)
            shutil.copy2(src, dst)
            frames.append({"date": d, "img": f"img/timeline/{key}/{dst_name}", "real_event": flags[d]})

        manifest["events"].append({
            "key": key, "label": label, "hazard": hazard,
            "extended": is_hero, "frames": frames,
        })
        print(f"{key}: {len(frames)} frames" + (" (extended, daily/weekly/pentad enabled)" if is_hero else ""))

    manifest_path = os.path.join(HERE, "..", "timeline_manifest.json")
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
    print("saved", manifest_path)
    eng.close()


if __name__ == "__main__":
    main()
