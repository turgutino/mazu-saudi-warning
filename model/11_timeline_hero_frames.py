# =============================================================================
# MAZU — Priority-4 item 2 support script: render an extended ~28-day frame
# set for one "hero" event (Dammam dust storm) so the interactive timeline's
# daily/weekly/pentad scale-switching has genuinely enough real data to
# demonstrate all three scales meaningfully (the site's other 11 events
# only have their original 4-6 day GIF windows, which is too short for a
# real weekly/pentad view -- see PRIORITY_FIXES_LOG.md for the scope
# reasoning: full per-frame render cost (~87s, due to the bias-panel
# ensemble computation) makes extending all 12 events impractical in one
# session).
# =============================================================================
import sys, os, datetime
sys.path.insert(0, r"C:\Users\Turqut\Desktop\Competation\mazu-system\agent")
from render_map import render_pair

HERE = os.path.dirname(os.path.abspath(__file__))
OUT_DIR = os.path.join(HERE, "timeline_frames", "dammam_dust_extended")
os.makedirs(OUT_DIR, exist_ok=True)

START = datetime.date(2025, 5, 5)
END = datetime.date(2025, 6, 1)  # 28 days inclusive


def main():
    d = START
    dates = []
    while d <= END:
        dates.append(d.isoformat())
        d += datetime.timedelta(days=1)
    print(f"Rendering {len(dates)} frames ({START} to {END})...")
    for ds in dates:
        out = os.path.join(OUT_DIR, f"{ds}.png")
        if os.path.exists(out):
            print("skip (exists):", out)
            continue
        render_pair(ds, "dust_storm", out, title_prefix="Dammam dust storm -- ", highlight_city="Dammam")
    print("DONE:", len(dates), "frames ->", OUT_DIR)


if __name__ == "__main__":
    main()
