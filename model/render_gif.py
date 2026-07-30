import sys, os, glob
sys.path.insert(0, r"C:\Users\Turqut\Desktop\Competation\mazu-system\agent")
from render_map import render_pair
from PIL import Image

FRAME_DIR = "gif_frames"
os.makedirs(FRAME_DIR, exist_ok=True)


def build_gif(dates, hazard, event_label, highlight_city, out_gif, fps_ms=900, extra_markers=None):
    frame_paths = []
    for d in dates:
        safe_label = event_label.replace(' ', '_').replace('/', '-').replace('\\', '-')
        fp = os.path.join(FRAME_DIR, f"{hazard}_{safe_label}_{d}.png")
        render_pair(d, hazard, fp, title_prefix=f"{event_label} -- ", highlight_city=highlight_city, extra_markers=extra_markers)
        frame_paths.append(fp)
    frames = [Image.open(f).convert("RGB") for f in frame_paths]

    # Priority-4 item 3 added a 3rd (bias, blue-white-red) panel, pushing
    # each frame's distinct-color count higher; GIF's 256-color palette is
    # shared across the whole saved file, and PIL's default save_all
    # quantizes off a single frame -- colors that only appear in OTHER
    # frames (e.g. the bias panel's blue) can get crowded out and drift
    # toward a nearby palette entry (observed: blue -> teal). Build one
    # master palette from ALL frames combined so every frame's real colors
    # are represented, then quantize every frame against it with dithering.
    combo = Image.new("RGB", (sum(f.width for f in frames), frames[0].height))
    x = 0
    for f in frames:
        combo.paste(f, (x, 0))
        x += f.width
    master_palette = combo.quantize(colors=256, method=Image.Quantize.MAXCOVERAGE)
    frames_p = [f.quantize(palette=master_palette, dither=Image.Dither.FLOYDSTEINBERG) for f in frames]

    frames_p[0].save(out_gif, save_all=True, append_images=frames_p[1:],
                      duration=fps_ms, loop=0)
    print("saved GIF:", out_gif, f"({len(frames)} frames)")


if __name__ == "__main__":
    build_gif(
        ["2025-05-15", "2025-05-16", "2025-05-17", "2025-05-18", "2025-05-19", "2025-05-20"],
        "dust_storm", "Dammam dust storm", "Dammam", "dammam_dust.gif",
    )
    build_gif(
        ["2025-06-10", "2025-06-11", "2025-06-12", "2025-06-13", "2025-06-14", "2025-06-15"],
        "heatwave", "Dammam heatwave", "Dammam", "dammam_heat.gif",
    )
    build_gif(
        ["2025-12-07", "2025-12-08", "2025-12-09", "2025-12-10", "2025-12-11", "2025-12-12"],
        "flash_flood", "Jeddah flood", "Jeddah", "jeddah_flood.gif",
    )
