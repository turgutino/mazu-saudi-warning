# MAZU Website Map Fixes — Priority Log

Teacher's critique document: `参赛作品网页图的问题.docx`, section "三、分维度改进方案（按优先级）".
This log tracks, per priority, exactly what the document required (literal quote), what was
changed, in which files, and how it was verified. Updated as each priority is completed.

Live repo: `deploy/` (git remote `mazu-saudi-warning`). Deployed images: `deploy/img/` (29 files
listed below, plus 4 more under `img/layers/` from Priority 3 item 3, listed separately).

### Full deployed file list (updated 2026-07-29, includes all Priority 1-3 additions)

**Static event maps (12):**
`map_dammam_dust.png`, `map_dammam_heat.png`, `map_flood_aug2728.png`, `map_flood_jan.png`,
`map_flood_mar.png`, `map_flood_taif.png`, `map_haboob_dust.png`, `map_heatwave2_miss.png`,
`map_heatwave3.png`, `map_heatwave4.png`, `map_jeddah_flood.png`, `map_junjul_dust.png`

**Event GIFs (12):**
`dammam_dust.gif`, `dammam_heat.gif`, `flood_aug2728.gif`, `flood_jan.gif`, `flood_mar.gif`,
`flood_taif.gif`, `haboob_dust.gif`, `heatwave2_miss.gif`, `heatwave3.gif`, `heatwave4.gif`,
`jeddah_flood.gif`, `junjul_dust.gif`

**General figures (5):**
`risk_known_events.png`, `risk_annual_hotspots.png`, `forecast_vs_actual.png`,
`compound_hazard_climatology.png`, `compound_hazard_event.png` (last 2 added by Priority 3 item 2)

All 27 Priority-1/2 filenames confirmed 1:1 against `index.html`'s `<img>`/`<source>` references
(no missing/extra/renamed file) — see verification note under Priority 2 below. The 2 new
Priority-3 files confirmed loading with `200 OK` via a live local server + browser network check
(see Priority 3 below).

**Priority 3 additions (not part of the 29-file `img/` count above, listed separately):**
- `compound_layers.html` (deploy root) — new interactive page, item 3
- `img/layers/base_risk_flash_flood.png`, `img/layers/slope.png`, `img/layers/wadi.png`,
  `img/layers/population.png` — the 4 toggleable layers item 3 produces
- `model/hydro_data/HydroRIVERS_v10_eu_shp/` — downloaded source data, gitignored, not deployed
  as-is (only used to generate `wadi.png`)

---

## Priority 1 — 修正基础地理制图错误 (Fix basic geographic mapping errors)

**Status: ✅ COMPLETE (2026-07-29)**

### Sənədin tələbi (literal text)

> 优先级 1：修正基础地理制图错误
> 1) 替换标准 GIS 底图：叠加沙特全国省界、完整国境线、主要城市点位、红海海岸线；补充地形分层设色底图（区分山地 / 高原 / 平原 / 海岸），网格图层半透明叠加在地形底图之上。
> 2) 补齐北部边境网格覆盖：扩大模型绘图范围至沙特全境 + 周边邻国缓冲带（科威特、约旦、伊拉克、也门北部），消除北部空白截断问题；空白无数据区域增加灰色遮罩 + 图例标注"模型未覆盖区域"。
> 3) 地理投影校正：改用等面积地理投影绘图，消除南北网格视觉面积畸变，保证风险区域大小视觉真实。

### Nə dəyişdi

| Sub-item | Implementation |
|---|---|
| Province borders, national borders, city markers, Red Sea coastline | `cfeature.BORDERS`, `admin_1_states_provinces_lines` (10m), `cfeature.COASTLINE` (10m), city dot markers — all at 10m NaturalEarth scale |
| Terrain-tiered basemap (mountain/plateau/plain/coast) | Real elevation hillshade built from the dataset's own `orography` variable (`LightSource.shade()`), 6-band `TERRAIN_CMAP`, cached via `_terrain_rgb()` |
| Grid layer semi-transparent over terrain | `zorder`: terrain=0.3, risk color layer=1.5 (terrain drawn first, risk layer on top at base_alpha 0.65–0.7) |
| Expand to full Saudi Arabia + neighbor buffer (Kuwait, Jordan, Iraq, north Yemen) | `PLOT_EXTENT=[32.0,58.0,14.0,34.0]` (wider than `DATA_EXTENT=[34.0,55.9,16.0,31.9]`); `NEIGHBOR_LABELS` covers JORDAN, IRAQ, KUWAIT, YEMEN (+ OMAN, UAE) |
| Gray no-data mask + legend label | `PathPatch` gray frame (outer=PLOT_EXTENT minus inner=DATA_EXTENT) + `fig.legend(...)` patch labeled "Outside model coverage (no data)" |
| Equal-area projection | `PROJ = ccrs.AlbersEqualArea(central_longitude=45, central_latitude=23, standard_parallels=(18,30))`, used everywhere instead of `PlateCarree` for plotting |

### Fayllar

- `deploy/model/render_map.py` — shared base constants/functions (`PROJ`, `PC`, `DATA_EXTENT`, `NEIGHBOR_LABELS`, `_terrain_rgb`, `_draw_base`, `_draw_cities`)
- `deploy/model/render_multiday.py` — static multi-day panels (imports shared functions)
- `deploy/model/render_gif.py` — GIF frames (calls `render_map.render_pair`, inherits automatically)
- `deploy/model/02_risk_maps.py` — `risk_known_events.png`, `risk_annual_hotspots.png` (own `draw_base()`, mirrors the shared logic)
- `deploy/model/04_forecast_maps.py` — `forecast_vs_actual.png` (own `_draw_base()`, mirrors the shared logic)

### Necə təsdiq edildi

- `grep "subplot_kw"` across all 5 files confirmed `projection: PROJ` (equal-area) used consistently — no leftover `PlateCarree`.
- Confirmed (grep) `admin_1_states_provinces_lines`, `BORDERS`, `_terrain_rgb`, `COASTLINE`, no-data `PathPatch` present in all 3 base-drawing files.
- **Bug found & fixed (2026-07-29):** `02_risk_maps.py` drew the gray no-data mask patch but had no `fig.legend(...)` explaining it (unlike `render_map.py`/`04_forecast_maps.py`, which did). Fixed by adding `coverage_patch` + `fig.legend(...)` to both of its figures, matching the exact color (`#444444`, alpha 0.55) already used for the mask patch itself. Re-tested, re-copied to `deploy/img/`, verified directly from the deployed file.
- All city/label positions and neighbor-country buffer visually confirmed against rendered output.

---

## Priority 2 — 重构色标、分级体系 (Rebuild color scale / tiering system)

**Status: ✅ COMPLETE (2026-07-29)**

### Sənədin tələbi (literal text)

> 优先级 2：重构色标、分级体系（解决判读混淆问题）
> 1) 分灾种独立色阶阈值：山洪：0–0.1 低 / 0.1–0.3 中 / 0.3–0.6 高 / ＞0.6 极端；热浪、沙尘暴：0–0.2 低 / 0.2–0.4 中 / 0.4–0.7 高 / ＞0.7 极端；三类灾害分开配置图例、色条。
> 2) 非线性分级配色：贴合沙特气象预警四级分级标准，拉大极端风险色块色差。
> 3) 增加双层空间绘图：主图层：灾害风险概率；辅助半透明图层：模型置信度（灰度遮罩），低置信网格自动淡化。

### Nə dəyişdi

| Sub-item | Implementation |
|---|---|
| Per-hazard independent thresholds | `TIER_BOUNDS = {"flash_flood":[0,0.1,0.3,0.6,1.0], "heatwave":[0,0.2,0.4,0.7,1.0], "dust_storm":[0,0.2,0.4,0.7,1.0]}` — exact numeric match to the document |
| Separate legend/colorbar per hazard | Every map's colorbar labeled `f"Risk tier ({hazard})"`, ticked at the tier bounds |
| Non-linear, high-contrast tiering | `mcolors.BoundaryNorm` + 4 discrete flat colors (`ListedColormap`) — green/yellow/orange/red, not a continuous gradient |
| Two-layer spatial rendering: main risk layer + auxiliary semi-transparent gray confidence mask | **Two separate `imshow` calls**: main tiered-risk RGBA at `zorder=1.5`, then a second gray-mask RGBA (`confidence_mask_rgba`, color `#888888`, alpha scales 0→`_MASK_MAX_ALPHA` with ensemble std) at `zorder=1.6` on top |
| Confidence source | 5-member ensembles: production ensemble (`agent/saved_models/ensemble/*.joblib`) for the 24 event maps/GIFs; a fresh 5-seed bootstrap ensemble of the same `HistGradientBoostingClassifier` architecture for `forecast_vs_actual.png` (different feature set than the production ensemble, so it isn't reusable there) |
| Confidence applies only to model/forecast panels, never to rule-based/ground-truth panels | Confirmed in all 3 files: `real_grid`/"REAL" panels never call `confidence_mask_rgba`; `02_risk_maps.py` (rule-based `DetectionEngine.risk_field`, no ML model) never uses it at all |

### Fayllar

- `deploy/model/render_map.py` — `TIER_BOUNDS`, `tiered_norm_cmap`, `tiered_rgba`, `confidence_mask_alpha`, `confidence_mask_rgba` (shared core)
- `deploy/model/render_multiday.py` — static multi-day panels, predicted row gets the mask overlay
- `deploy/model/render_gif.py` — inherits via `render_map.render_pair`
- `deploy/model/04_forecast_maps.py` — forecast probability panel gets tiered colors + its own bootstrap-ensemble mask overlay; actual-outcome panel (binary) unchanged
- `deploy/model/02_risk_maps.py` — `risk_known_events.png`'s risk-probability figure gets tiered colors (no confidence layer: rule-based, not ML); `risk_annual_hotspots.png` intentionally NOT tiered (it's an extreme-tier-day COUNT, not a 0–1 probability — out of this priority's literal scope)

### Necə təsdiq edildi

1. **Rəqəmsal**: `TIER_BOUNDS` re-read character by character against the source doc text (extracted verbatim from the teacher's docx) — exact match.
2. **İlkin səhv tapıldı və düzəldildi**: first implementation modulated alpha within a single RGBA array (visually equivalent but architecturally NOT the "two separate layers" the document literally specifies). Rebuilt as two genuine `imshow` calls per panel.
3. **Sintetik test**: a synthetic std gradient confirmed the mask mechanism fades top→bottom correctly (`mask_synthetic_test.png`, isolated from real data).
4. **Real-file görüntü təsdiqi**: found the real deployed event with the highest ensemble disagreement (`map_junjul_dust.png`, 2025-07-05, dust_storm, ~10% of cells above `STD_HIGH`), zoomed 3× into the deployed PNG, and visually confirmed the gray mask desaturating that region compared to neighboring dates — not just a code-level claim.
5. **Orphan-code check**: `grep confidence_alpha` across `deploy/model/` returned zero matches — no leftover reference to the old single-layer alpha-fade function.
6. **Missed-file sweep**: grepped the whole `mazu-system` tree for `risk_field|predicted_grid|real_grid` to check for any other image-producing script; found `07_dust_storm_forecast.py` (produces only a text report, no images — out of scope) and a non-deployed duplicate `mazu-system/model/` folder (not part of the `deploy/` git repo, doesn't serve the live site — out of scope).
7. **HTML cross-check**: extracted every `img/*.png`/`img/*.gif` reference from `index.html` (27 filenames) and confirmed 1:1 against the regenerated file set — no missing/extra/renamed file.
8. **Timestamp audit**: confirmed all 27 deployed files in `deploy/img/` postdate every relevant code file's final edit (no stale copies slipped through after the two-layer rebuild).

---

## Priority 3 — 增加对比辅助图层 (Add comparison auxiliary layers)

**Status: ✅ COMPLETE — all 3 items done (2026-07-29)**

### Sənədin tələbi (literal text)

> 优先级 3：增加对比辅助图层，提升科研与业务价值
> 1) 叠加实况验证点位：每张空间图叠加当日真实灾害标记点（红点 = 山洪、橙点 = 热浪、黄点 = 沙尘），直观展示漏报、空报空间分布。
> 2) 新增复合灾害叠加图层：开发多灾种联合风险填色图，用混合色标注热浪 + 沙尘、暴雨 + 高温重叠高风险带。
> 3) 下垫面 / 脆弱性叠加可选开关：支持一键切换图层：地形坡度、水系流域、城镇人口密度，实现"风险 + 暴露度"一体化空间展示。

### Item 1 — 叠加实况验证点位 (real-event marker overlay) — ✅ DONE

| Sub-item | Implementation |
|---|---|
| Marker on every spatial map, colored by hazard (red=flash_flood, orange=heatwave, yellow=dust_storm) | `EVENT_MARKER_COLORS = {"flash_flood":"#e03131","heatwave":"#f76707","dust_storm":"#fcc419"}` in `render_map.py`; new `_draw_event_markers(ax, date, hazard)` plots each `DetectionEngine.detect(date, hazard)` cluster's peak lat/lon as a filled dot with black edge |
| Data source = real ground truth (not the model) | Reuses `tools._get_detection_engine()` — the exact same rule-based engine `real_grid()` already draws its color field from, so markers are consistent with the "Real" panel by construction |
| Shown on both Real and Predicted panels | Per doc's "每张空间图" (every spatial map) wording — lets a viewer see missed-detection (marker present, predicted panel shows low risk there) or false-alarm (predicted panel shows high risk, no marker) at a glance |
| Legend entry per map | Dynamic `plt.Line2D` legend handle added, labeled with the hazard name and matching color |
| `02_risk_maps.py`'s pre-existing markers | This file already had its own marker system, colored by **severity** (red=extreme, yellow=other) rather than by hazard type — a decision predating this priority. Asked the user; confirmed answer: **replace** with the document's hazard-type color scheme for consistency with the rest of the site. Both `risk_known_events.png` subplots (flash_flood + heatwave in one figure) now show both marker colors with a joint legend. |
| `04_forecast_maps.py` (`forecast_vs_actual.png`) | **Initially missed** — caught during a follow-up "did we really finish item 1?" check (grepped `deploy/model/` for `_draw_event_markers`/`EVENT_MARKER_COLORS` and found this file absent). Added its own `_draw_event_markers()` closure (same logic, using `tools._get_detection_engine()` directly since this file doesn't import `render_build_grid`), wired into both the forecast-probability row AND the actual-outcome row (both represent date `d_to`), joint red/orange legend added (this figure mixes flash_flood + heatwave columns like `02_risk_maps.py` does). |

**Fayllar**: `deploy/model/render_map.py` (`EVENT_MARKER_COLORS`, `_draw_event_markers`, wired into `render_pair`), `deploy/model/render_multiday.py` (wired into both rows), `deploy/model/render_gif.py` (inherits via `render_pair`), `deploy/model/02_risk_maps.py` (`plot_event()` marker color swapped, legend updated), `deploy/model/04_forecast_maps.py` (own `_draw_event_markers()` closure, wired into both rows)

**Necə təsdiq edildi**: isolated test of `render_map.py` (visually confirmed a false-alarm case near Medina — predicted panel showed high risk with no real-event dot there); isolated test of `render_multiday.py`; isolated test of `02_risk_maps.py` (both flash_flood-red and heatwave-orange dots visible with correct joint legend); full batch regeneration of all 23 event files + `map_jeddah_flood.png` + `risk_known_events.png`; copied to `deploy/img/`; spot-checked 2 files directly from the deployed folder (`map_flood_taif.png` static + a `flood_taif.gif` frame) — both show a clear real/predicted disagreement case (SW-mountain extreme risk in "Real" not matched by "Predicted"), exactly the diagnostic purpose the document states. **Coverage sweep**: grepped all of `deploy/model/` for `EVENT_MARKER_COLORS` after the `04_forecast_maps.py` fix — confirmed present in all 4 marker-owning files (`render_map.py`, `render_multiday.py`, `02_risk_maps.py`, `04_forecast_maps.py`), which together cover all 27 deployed files. `04_forecast_maps.py` re-tested (retrains its 10-model bootstrap ensemble, ~3 min), output copied to `deploy/img/forecast_vs_actual.png`, verified directly from the deployed file.

### Item 2 — 新增复合灾害叠加图层 (compound-hazard overlay map) — ✅ DONE (2026-07-29)

Literal text: "开发多灾种联合风险填色图，用混合色标注热浪 + 沙尘、暴雨 + 高温重叠高风险带" — a
multi-hazard joint risk map, using **mixed colors** to mark heatwave+dust_storm and
flash_flood+heatwave overlapping high-risk zones. Only these 2 named pairs are in scope (not
flash_flood+dust_storm, which the document doesn't mention).

User's instruction: "however the Word doc says it, do it at the highest level, don't skip
anything" — interpreted as: build both a climatological summary AND a real single-day example
(the doc doesn't specify which, so both formats cover every reasonable reading), as a new
standalone script (user's explicit choice over extending `02_risk_maps.py`).

| Sub-item | Implementation |
|---|---|
| New standalone script | `deploy/model/08_compound_hazard.py` (new file, not modifying any existing script) |
| Both hazard pairs, ground truth from the same rule-based engine as everywhere else | `DetectionEngine.risk_field(date, hazard)`, thresholded at each hazard's own "extreme" tier lower bound reused from `render_map.TIER_BOUNDS` (flash_flood ≥0.6, heatwave/dust_storm ≥0.7) — not a new/invented definition of "extreme" |
| Figure 1 — climatology | `compound_hazard_climatology.png`: per-cell count of 2025 days where both hazards in a pair were extreme simultaneously, one panel per pair, mirrors `risk_annual_hotspots.png`'s existing style/basemap |
| Figure 2 — real single-day example, genuine mixed color | `compound_hazard_event.png`: for each pair, the *actual* date in 2025 with the most simultaneous extreme-tier overlap cells (found by scanning all 365 days, not picked arbitrarily) — heatwave+dust_storm → 2025-07-20 (722 overlap cells), flash_flood+heatwave → 2025-08-17 (34 overlap cells). Genuine bivariate color mixing: red = hazard A extreme only, blue = hazard B extreme only, **purple = both at once** (the actual "混合色" the document names) — per-panel legend since which hazard is "A"/"B" differs between the two pairs |
| Added to the live site | Two new `<div class="fig">` blocks in `index.html`'s "01 · Detection" section, right after `risk_known_events.png` |

**Əlavə tapılan boşluq (bu addımın gedişində)**: `index.html`'s existing caption for
`risk_known_events.png` still said "red rings mark extreme-severity clusters" — stale text left
over from Priority-3 item 1's marker-color change (severity-colored rings → hazard-colored
dots). Fixed the caption text to match the actual current image.

**Fayllar**: `deploy/model/08_compound_hazard.py` (new), `deploy/index.html` (2 new `<div class="fig">` blocks + 1 stale-caption fix)

**Necə təsdiq edildi**: ran the script standalone (scans all 365 days for both pairs, ~a few minutes), visually inspected both output figures (climatology shows a physically sensible pattern — heatwave+dust concentrated in the north near Iraq/Kuwait, flood+heat compounding rare and confined to the SW mountains; event figure shows a clear purple overlap zone on 2025-07-20 near Kuwait and a small one near Jizan on 2025-08-17); copied both PNGs to `deploy/img/`; started a local HTTP server (`mazu-deploy` launch config) and used the browser tools to confirm both new images plus every other image already on the page return `200 OK` with zero 404s — not just "the file exists," but "the live page actually loads it."

### Item 3 — 下垫面 / 脆弱性叠加可选开关 (optional toggle layers) — ✅ DONE (2026-07-29)

Literal text: "支持一键切换图层：地形坡度、水系流域、城镇人口密度，实现'风险 + 暴露度'一体化空间展示" —
support one-click layer switching: terrain slope, watershed/river system, town population
density, achieving an integrated "risk + exposure" spatial display. Unlike items 1-2, "一键切换"
(one-click switching) requires genuine client-side interactivity — static PNG/GIF cannot do this
— so this item produced a new interactive HTML page, not another matplotlib figure.

**Data-source investigation (each layer checked for real data before building anything):**

| Layer | What was checked | Finding | Decision |
|---|---|---|---|
| Terrain slope | Project's own `orography` variable (10km grid) | Real data, but coarse — max observed slope only ~0.7% at this resolution | Compute via `np.gradient`, normalize against the *actual* observed range (not an arbitrary steep-terrain percentage) — first attempt used a 15% scale and rendered nearly invisible; caught and fixed by checking the real data range first |
| Watershed / river network | Cartopy's built-in Natural Earth `rivers_lake_centerlines` | Rendered **zero** rivers inside Saudi Arabia's own borders (confirmed visually) — all visible rivers were in neighboring Iraq/Yemen | Rejected as unusable for this purpose. Asked the user whether to (a) approximate from the project's own coarse elevation data or (b) fetch a real hydrography dataset. User asked "why not find better data" — searched, found HydroSHEDS/HydroRIVERS (Lehner & Grill 2013), a real, citable, freely available global river-network product |
| **First HydroRIVERS download attempt** | Downloaded the "Asia" regional shapefile (91MB, per web search) | **Wrong file** — that region's actual coverage starts at 57.6°E, entirely east of Saudi Arabia (34-55.9°E); 0 of 1.4M records intersected the Saudi extent, confirmed by testing a known-river bounding box (Euphrates) which also returned 0 | Caught by testing against a bounding box known to have rivers instead of trusting the 0-result on Saudi alone. Deleted the wrong file, re-checked HydroSHEDS's actual regional breakdown, found "Europe and Middle East" (68MB) is the correct region for the Arabian Peninsula |
| Town population | Searched the whole project for population/density files | Found `agent/city_population.json` — real, GASTAT (Saudi Census 2022) sourced city-proper totals for the 8 modeled cities. No gridded density surface exists anywhere in the project | Used as-is: proportionally-sized city markers (sqrt-scaled so *area*, not radius, tracks population), explicitly labeled in the UI as point totals, not a density surface — no fabrication of a raster that doesn't exist |

**Downloads (each confirmed with the user first, per standing policy):**
1. `HydroRIVERS_v10_as_shp.zip` (91MB, Asia region) — downloaded, found to be the wrong region, deleted
2. `HydroRIVERS_v10_eu_shp.zip` (68MB, Europe+Middle East region) — downloaded, confirmed 170,369 segments intersect the Saudi extent; kept in `deploy/model/hydro_data/` (gitignored, matches the existing `*.nc`/`data/` pattern for large files)

**What was built:**

| Component | File | Description |
|---|---|---|
| Layer generator script | `deploy/model/09_interactive_layers.py` | Produces 4 transparent-background PNGs in plain PlateCarree (not the site's usual AlbersEqualArea — Leaflet needs simple lat/lon image bounds), all pixel-identical in extent (1500×1089, verified to match `DATA_EXTENT`'s aspect ratio exactly: 21.9°/15.9° = 1.377 = 1500/1089) so every layer aligns without manual offset |
| Base risk layer | `img/layers/base_risk_flash_flood.png` | `DetectionEngine.risk_field()` for 2025-08-23 (the real Jizan 254.9mm event), same tiered colors as every other map |
| Slope layer | `img/layers/slope.png` | Real elevation-derived gradient, normalized to the actual observed range |
| Wadi layer | `img/layers/wadi.png` | Real HydroRIVERS segments, Strahler stream order ≥4 (13,376 segments within `DATA_EXTENT`, vs 170,369 in the wider buffer zone) |
| Population layer | `img/layers/population.png` | Real GASTAT 2022 city totals as sized circles |
| Interactive page | `deploy/compound_layers.html` | Leaflet.js map (dark CARTO basemap), 4 real checkboxes wired to `L.imageOverlay` add/remove — genuine one-click toggle, not a static mockup. Data-honesty note in the UI itself disclosing each layer's real source and limitations |
| Site integration | `deploy/index.html` | New sentence + link ("Explore the interactive risk + exposure layers →") at the end of the "01 · Detection" section, pointing to `compound_layers.html` |

**Necə təsdiq edildi**: started the `mazu-deploy` local server and used real browser interaction (not just file existence checks) — clicked each of the 4 checkboxes via the browser tool's ref-based click (first two attempts landed on stale coordinates and silently failed, caught by screenshotting after each click rather than assuming success; switched to fresh `read_page` refs, which worked), screenshotted after each toggle to confirm: risk-only, risk+slope, risk+slope+wadi, and slope+wadi+population-with-risk-off all render correctly and align precisely with the real Leaflet basemap's own city labels (population circles visibly centered on "Medina" and "Jeddah" labels). Confirmed the `index.html` link's `href` resolves to the correct URL via direct DOM inspection. Confirmed zero console errors throughout.

---

## Priority 4 — 优化时序交互与多图布局 (Optimize time-series interaction and multi-map layout)

**Status: ✅ COMPLETE — all 3 items done (2026-07-29)**

### Sənədin tələbi (literal text)

> 优先级 4：优化时序交互与多图布局
> 1) 新增一张沙特全域总览图，同步展示三类灾害全国分布，再搭配分城市细节放大图，兼顾全局与局地。
> 2) 升级动态时序可视化：替换自动 GIF，改为可交互时间轴：支持手动拖动日期、定格灾害事件，底部标注对应真实灾害记录；增加单日、逐周、逐候三种时间尺度切换。
> 3) 分窗口并列对比图：增加"预测风险场 vs 规则基准真值场"双栏并排绘图，方便直接对比模型偏差的空间分布规律。

### Item 1 — 新增一张沙特全域总览图 (national overview map) — ✅ DONE (2026-07-29)

Literal text: "新增一张沙特全域总览图，同步展示三类灾害全国分布，再搭配分城市细节放大图，兼顾全局与局地" —
add ONE national overview map SIMULTANEOUSLY showing all 3 hazards' national distribution, paired
with per-city detail zoom-ins, covering both the global and local view.

**Design decision**: "simultaneously" (同步) on "one" (一张) map, for 3 hazards, means a genuine
single-map composite rather than 3 side-by-side panels (which would really be 3 maps). Built a
"dominant hazard" map: for a real 365-day scan (same rule-based `DetectionEngine`, same "extreme"
tier thresholds as every other map on the site), each grid cell is colored by whichever hazard
reached extreme-tier on the most days there, using the already-established `EVENT_MARKER_COLORS`
(red=flash_flood, orange=heatwave, yellow=dust_storm) for site-wide color consistency, with alpha
scaled by how dominant that hazard is (normalized per-hazard by its own max, so a naturally rarer
hazard like flash_flood isn't visually erased by heatwave's higher raw day-counts).

| Sub-item | Implementation |
|---|---|
| New standalone script | `deploy/model/10_national_overview.py` |
| National composite map | `_extreme_day_counts()` (365-day × 3-hazard scan) → `_dominant_hazard_rgba()` |
| 8 per-city detail zoom-ins | Same dominant-hazard composite, cropped to ±2.2°/±2.0° around each of the 8 modeled cities, laid out 2 rows × 4 columns under the national map in one combined figure |
| Added to the live site | New `<div class="fig">` in `index.html`'s "01 · Detection" section, between `risk_known_events.png` and the compound-hazard figures |

**Bug found & fixed during testing**: the first render showed city name labels scattered across
the *entire* figure's blank margins, nowhere near their actual map panels. Root cause: cartopy
`GeoAxes` does not clip `ax.text()`/`ax.plot()` to the visible extent by default — on the tight
per-city zoom panels, the other 7 cities' labels (whose true geographic position is far outside
that panel's small extent) were still being drawn at their real coordinates, landing far outside
the small subplot's visible box. Fixed by filtering `_draw_cities()` to only draw cities within
the current axes' actual extent (`ax.get_extent()`, with a small margin) before plotting, plus
`clip_on=True` as a second safety net. Re-rendered and re-inspected before accepting.

**Fayllar**: `deploy/model/10_national_overview.py` (new), `deploy/index.html` (1 new `<div class="fig">`)

**Necə təsdiq edildi**: ran standalone (365×3 real scan, a few minutes), caught and fixed the
label-clipping bug on the first render (did not accept the buggy version), re-rendered and
visually confirmed all 8 city panels now show only their own relevant label(s), copied to
`deploy/img/national_overview.png`, started the `mazu-deploy` local server and confirmed the new
image loads `200 OK` with zero console errors.

### Item 2 — 升级动态时序可视化 (interactive timeline, replacing auto-GIF) — ✅ DONE (2026-07-29)

Literal text: "替换自动 GIF，改为可交互时间轴：支持手动拖动日期、定格灾害事件，底部标注对应真实灾害记录；增加单日、
逐周、逐候三种时间尺度切换" — replace the auto-playing GIF with an interactive timeline: manual
date dragging, freeze-frame on hazard events, bottom-labeled with real hazard records; add
daily/weekly/pentad(5-day) scale switching.

**Scope decision (made explicitly, user delegated "whatever is best")**: timed a single frame
render at ~87s (the bias-panel ensemble computation added in item 3 makes this much slower than
a plain risk map). Extending all 12 events to a real multi-week window (originally considered:
21 days × 12 events = 252 frames ≈ 6 hours) was not feasible in one session. Decision: give all
12 events genuine interactive daily-scale playback (reusing the already-rendered `gif_frames/*.png`
— zero extra render cost), and extend exactly **one** representative "hero" event (Dammam dust
storm) to a real 28-day window (5 May–1 Jun 2025, ~41 min to render) so daily/weekly/pentad
scale-switching has enough real data to be genuinely meaningful, rather than faked. Documented
in the page itself (a visible "Scope note") rather than hidden.

| Sub-item | Implementation |
|---|---|
| Extended hero frame set | `deploy/model/11_timeline_hero_frames.py` — 28 real `render_pair()` calls, Dammam dust storm, 5 May–1 Jun 2025 |
| Manifest (dates, frame paths, real-event flags) for all 12 events | `deploy/model/12_timeline_manifest.py` → `deploy/timeline_manifest.json` |
| Interactive page | `deploy/interactive_timeline.html` — event dropdown (12), image display, native range-slider (manual drag), Play/Pause with auto-advance, Daily/Weekly/Pentad scale buttons (Weekly/Pentad disabled for the 11 non-extended events, since a 6-day window can't support a real weekly step), bottom color-coded strip (red = real detected event that day, click to jump) |
| "定格灾害事件" (freeze on hazard event) | Auto-play pauses longer (1800ms vs 700ms) on frames flagged `real_event` |
| "底部标注对应真实灾害记录" (bottom labeled with real records) | The bottom strip's red cells, sourced from the same rule-based `DetectionEngine` as every other map |

**Bug found & fixed during testing**: the first version flagged "real event" using an *unfiltered*
national-level `eng.detect(date, hazard)` check — i.e. "did this hazard cluster ANYWHERE in Saudi
Arabia that day," not "did it happen near the city this timeline is actually about." Caught by
inspecting the numbers: 25 of 28 days in the Dammam dust-storm window were flagged real, which
made the freeze-on-event feature nearly meaningless (almost every frame would freeze). Root cause,
confirmed by direct inspection: dust_storm/heatwave clusters are detected somewhere in the country
on nearly every day (Northern Border, Empty Quarter, Red Sea, etc.), unrelated to the specific
city each timeline event is centered on. Fixed by adding per-event `focus_points` (the event's own
city/cities) and scoping the flag to detected-cluster peaks within 2° of those points; genuinely
national events (Haboob dust storm) keep the unfiltered check since they have no single focal
point by design. After the fix: Dammam dust storm dropped to 6/28 flagged days — spot-checked one
still-high case (`flood_aug2728`, 6/6 days) directly against the underlying detected clusters
before accepting it, confirmed genuine (that specific SW-mountain window really does cluster every
day, consistent with an existing site finding elsewhere), not a residual bug.

**Fayllar**: `deploy/model/11_timeline_hero_frames.py` (new), `deploy/model/12_timeline_manifest.py`
(new), `deploy/interactive_timeline.html` (new), `deploy/timeline_manifest.json` (generated),
`deploy/img/timeline/*` (94 frames, generated), `deploy/index.html` (1 new list item)

**Necə təsdiq edildi**: real browser interaction (not just file checks) via the `mazu-deploy`
local server — verified via direct DOM/state inspection after each action: (1) event dropdown
switching updates the image, date, dot count, and correctly disables Weekly/Pentad for
non-extended events; (2) Weekly scale on the hero event produces exactly 5 steps (7-day stride),
Pentad produces 7 steps (5-day stride), both mathematically verified against the 28-frame set;
(3) manual slider drag jumps to the correct date; (4) Play/Pause toggles and auto-advances;
(5) clicking a bottom-strip dot jumps to that exact date; (6) zero console errors throughout;
(7) `index.html`'s new link resolves to the correct URL via direct DOM inspection. The real-event
flagging bug above was caught by inspecting the actual flagged-day counts, not assumed correct
from the code looking reasonable.

### Item 3 — 分窗口并列对比图 (side-by-side predicted-vs-real comparison) — ✅ DONE

**Pre-check finding**: the literal 2-column "predicted vs rule-based ground-truth" side-by-side
format this item asks for was **already the site's core structure** — `render_pair()` and
`render_multiday.py` (used by all 24 event maps/GIFs) and `04_forecast_maps.py` already produce
exactly a "Real | Predicted" comparison, predating this priority-fix work entirely. Flagged this
to the user rather than assuming it needed rebuilding from scratch.

**User's decision**: go further than the literal 2-column ask — add a 3rd **bias/difference**
panel (Predicted − Real), since the doc's stated *purpose* ("方便直接对比模型偏差的空间分布规律" —
directly compare the spatial pattern of model bias) is served more literally by an explicit
difference map than by asking a viewer to eyeball two side-by-side panels.

| Sub-item | Implementation |
|---|---|
| Grid alignment (real_grid is native/fine resolution, predicted_grid is strided/coarse — NOT pixel-aligned) | New `_nearest_indices`/`resample_nearest` in `render_map.py`, resamples the real grid onto the predicted grid's coordinates via nearest-neighbor lookup before subtracting. Verified correct against 20 manually-computed nearest-neighbor lookups at nonzero-value points (0 mismatches) before integrating |
| Bias color scale | `BIAS_CMAP` (blue→white→red diverging, `-1` to `+1`), `bias_rgba()` — blue = model under-predicts, red = over-predicts |
| 3rd panel added to every event map/GIF | `render_map.py`'s `render_pair()` (2→3 panels), `render_multiday.py` (2→3 rows) |
| 3rd row added to the forecast-verification figure | `04_forecast_maps.py` — `proba` and `actual` are already on the identical stride grid here (no resampling needed, simpler case than `render_pair`) |
| Site captions updated to describe the new 3rd panel | `deploy/index.html`: "04 · Spatiotemporal Validation" section's lead paragraph (2 layers → 3 layers) and the `forecast_vs_actual.png` caption (2 rows → 3 rows) both rewritten |

**Bug found & fixed mid-implementation**: after adding the bias panel, GIF playback showed the
bias colorbar's blue drifting toward teal (confirmed via direct pixel inspection: GIF is a
256-color-indexed format, `mode='P'`; PIL's default `save_all` quantizes off a single frame's
color distribution, and the extra bias-panel colors pushed rarer hues like pure blue out of the
shared palette). Static PNGs were unaffected (24-bit, no quantization). Fixed in
`render_gif.py`: build one master 256-color palette from **all frames combined** (not just the
first), then quantize every frame against it with Floyd-Steinberg dithering. Verified the fix
in isolation (single test GIF) before regenerating all 12 production GIFs.

**Fayllar**: `deploy/model/render_map.py` (`_nearest_indices`, `resample_nearest`, `BIAS_CMAP`,
`BIAS_NORM`, `bias_rgba`, `render_pair` restructured to 3 panels), `deploy/model/render_multiday.py`
(3 rows), `deploy/model/04_forecast_maps.py` (3rd bias row), `deploy/model/render_gif.py`
(master-palette GIF quantization fix), `deploy/index.html` (2 caption updates)

**Necə təsdiq edildi**: isolated numeric test of `resample_nearest` (exact match against manual
nearest-neighbor lookup, including at nonzero-value points, not just a trivially-matching zero);
isolated visual test of `render_pair()`'s 3-panel output (bias panel showed a physically sensible
red zone over Kuwait matching the model's known over-prediction there); isolated visual test of
`render_multiday.py`'s 3-row output; isolated visual test of `04_forecast_maps.py`'s bias row
(blue under-prediction for flash_flood near Jizan, red over-prediction for heatwave — both
consistent with findings already documented elsewhere on the site); full batch regeneration of
all 11 static maps + `map_jeddah_flood.png` + `forecast_vs_actual.png` (bias panels), then a
second full regeneration of all 12 GIFs (palette fix) after the GIF color bug was caught and
fixed; copied everything to `deploy/img/`; spot-checked 2 static files and 2 GIF frames directly
from the deployed folder; re-verified the GIF color fix specifically by re-extracting a frame
from the deployed `dammam_dust.gif` and confirming the Dammam marker and bias colorbar render
pure blue, not teal; started the `mazu-deploy` local server and confirmed all images (including
the 2 newly-touched captions' images) return `200 OK` with zero 404s.

---

## Priority 5 — 细节规范优化 (Detail standardization)

**Status: ✅ COMPLETE — all 3 items done (2026-07-29)**

### Sənədin tələbi (literal text)

> 优先级 5：细节规范优化
> 1) 每张图统一标准化图注：包含预报时效、网格分辨率、绘图投影、分级阈值、数据年份、覆盖范围说明；
> 2) 网格平滑插值算法优化：降低平滑强度，避免跨地形虚假连片高风险，保留局地小尺度灾害高值细节；
> 3) 统一比例尺、坐标经纬度网格，所有分布图增加经纬刻度，方便定位灾害经纬度。

### Item 1 — 统一标准化图注 (standardized caption on every map) — ✅ DONE

Literal text requires 6 metadata fields on every map: forecast lead time, grid resolution, map
projection, tier thresholds, data year, coverage range.

**Implementation**: one shared `standard_caption()`/`add_standard_caption()` helper in
`render_map.py` so wording/format is identical everywhere rather than re-typed per script.
Hazard-aware (tier thresholds differ by hazard, so the caption lists whichever hazard(s) that
figure covers) and context-aware (`lead_time` parameter: `"t-1->t"` for panels showing the ML
model's forecast, omitted for rule-based/ground-truth-only or climatology figures). Two figures
(`risk_annual_hotspots.png`, `compound_hazard_climatology.png`) show a day-COUNT metric rather
than a 0-1 risk score, so their caption explicitly says the standard tier thresholds don't
directly apply there, rather than silently reusing thresholds that would be misleading for that
figure.

**Fayllar**: `render_map.py` (helper + wired into `render_pair`), `render_multiday.py`,
`02_risk_maps.py` (both figures), `04_forecast_maps.py`, `08_compound_hazard.py` (both figures),
`10_national_overview.py` — 7 files, every production figure now carries the caption.

### Item 2 — 网格平滑插值算法优化 (reduce grid-smoothing) — ✅ DONE

**Investigation**: grepped the whole rendering pipeline for `gaussian_filter`, `smooth`,
`interpolat`, `griddata`, and `shading=`/`interpolation=` parameters. Found zero actual smoothing
algorithms anywhere in the current code — the only `shading=` usage is `pcolormesh(...,
shading="auto")` in 3 files, which is a blocky/flat render mode, not interpolation. The
"cross-terrain false contiguous high-risk zones" the reviewer's original document describes was a
property of the **pre-fix** website (which this whole multi-priority effort already replaced);
nothing in the rebuilt `tiered_rgba`/`imshow` pipeline applies smoothing today.

**Made it explicit rather than relying on defaults**: matplotlib's `imshow()` default
interpolation isn't `"nearest"` in all versions/configs, so rather than assume the visual result
was already unsmoothed, added `interpolation="nearest"` explicitly to all 14 `imshow()` calls that
render real risk/bias/confidence-mask/dominant-hazard data (terrain hillshade and slope-layer
`imshow` calls were deliberately left alone — those are genuine continuous physical fields, not
discrete risk tiers, so smooth rendering is correct there, not a violation of this item).

**Fayllar**: `render_map.py` (2), `render_multiday.py` (4), `04_forecast_maps.py` (3),
`02_risk_maps.py` (1), `08_compound_hazard.py` (1), `10_national_overview.py` (2),
`09_interactive_layers.py` (1) — 14 total `interpolation="nearest"` additions.

### Item 3 — 统一比例尺、坐标经纬度网格 (unified scale/coordinate grid) — ✅ ALREADY SATISFIED (verified)

Audit (not new work): `grep`-checked every rendering script for `gridlines` / inheritance of the
shared `_draw_base()` (which calls `ax.gridlines(draw_labels=True, ...)`). Confirmed present in
`render_map.py`, `02_risk_maps.py`, `04_forecast_maps.py`, `08_compound_hazard.py`,
`10_national_overview.py` directly, and in `render_multiday.py` via its imported `_draw_base()`
call (no separate `gridlines` string match there, but the function itself draws them — traced the
import chain to confirm, not assumed). The `09_interactive_layers.py` Leaflet overlay PNGs
intentionally have **no** gridlines (`ax.set_axis_off()`) — a deliberate exception: those images
are transparent overlays meant to sit on Leaflet's own basemap, which already provides the
coordinate reference in its own UI; drawing a second, redundant lat/lon grid onto a transparent
overlay would look wrong and add visual noise, not clarity.

### Necə təsdiq edildi (full priority)

1. Isolated test of each of the 6 modified rendering scripts individually before any batch
   regeneration — confirmed the caption renders without overlapping the legend/colorbar, and that
   `interpolation="nearest"` produces visibly sharp/blocky pixels (compared directly against a
   pre-fix render) rather than a blurred gradient.
2. **Incident during batch regeneration, disclosed here rather than smoothed over**: the first
   full-batch regeneration command failed silently (platform-level "usage limit" interruption,
   visible to the user but not immediately to me) while I had already moved on and reported
   "starting the batch" as if it were running. Caught this **only because the user asked me to
   re-verify** — checked the `regenerated/` output folder directly and found it genuinely empty
   (0 files), confirming the claim of "started" had been wrong. Restarted it correctly. While
   re-verifying, discovered the *original* first attempt had actually been running successfully
   the whole time in the background (I had checked it one time-window too early, at the exact
   moment the output folder was created but before the first file existed, and wrongly concluded
   it had failed) — meaning **two copies of the same regeneration script were running
   concurrently**, both writing into the same output folder. Killed the redundant duplicate
   process (`Stop-Process`) as soon as this was discovered, then verified file-by-file integrity
   of all 23 outputs (`PIL.Image.load()` on every file — 0 corrupted) to confirm the concurrent
   write window hadn't damaged anything.
3. Full batch regeneration (11 static + 12 GIFs) + separate `map_jeddah_flood.png` regeneration +
   `09_interactive_layers.py` (4 Leaflet layer PNGs) + `11_timeline_hero_frames.py` (28 hero
   timeline frames, ~41 min, since `render_pair()` changed) + `12_timeline_manifest.py` refresh —
   all re-run against the final Priority-5 code, not the stale pre-fix versions.
4. Copied all outputs to `deploy/img/` (24 event files, 6 general figures, 4 interactive layers,
   94 timeline frames = 128 files total this pass).
5. Spot-checked 2 files directly from the deployed `img/` folder (not `regenerated/` or
   `outputs/`) — confirmed caption + sharp pixels present in the actually-deployed copy.
6. Started the `mazu-deploy` local server and ran real browser checks on all 3 HTML pages:
   `index.html` (42/42 links+images `200 OK`), `interactive_timeline.html` (94/94 frames
   `200 OK`, plus a real click-test confirming Weekly scale still produces exactly 5 steps against
   the freshly-regenerated 28-frame set), `compound_layers.html` (4/4 layer images `200 OK`).
   Zero console errors on any page.

---

## Post-push follow-up: Knowledge Graph updated to reflect current site state (2026-07-30)

**Trigger:** after the first push (all 5 Priorities), user asked whether `kg_data.json`/`kg_view.html`
actually reflected the site's current findings. Investigation found it did not — its 6 "Event" nodes
were auto-detected annual-extreme values, unrelated to the 12 real, individually-verified events in
`index.html`'s map-verification section, and none of this session's findings (Jan-Mar data-gap, the
flood_mar model/rule disagreement, compound-hazard co-occurrence) were represented anywhere in the KG.
Also confirmed the KG is a real production dependency, not just a viewable diagram: `agent/tools.py`'s
`causal_kg_tool` and `similar_events_tool` query `kg_data.json` directly, and `FULL_SYSTEM_AUDIT.py` /
`agent/02_test_tools.py` assert its exact structure (node/edge counts, per-hazard event counts).

**Change made (`kg/01_build_structural_kg.py`):** added all 12 site-verified events as new `Event`
nodes (id prefix `EV_`, distinct from the original `E_`-prefixed auto-detected events, so nothing
existing was renamed or removed — purely additive). Each new event carries its real date, region(s),
exact verdict text copied from its `index.html` caption, and real driver-indicator values freshly read
from `mazu_dataset.nc` at that fixed date/location (never copied from caption prose). Added 2 new
`Region` nodes (Hail, Buraidah) for the `flood_mar` event, since neither is one of the original 8
modeled cities.

**Precision work on the event dates themselves** (not just accepting the caption's date range at face
value):
- `dammam_heat`'s caption gives a "11-14 June" range and a "44.2°C" peak but names no single day.
  Independently re-derived from `mazu_dataset.nc`: the actual peak is **16 June** (44.22°C) — one day
  *after* both the caption's stated window and the rendered map/GIF's 10-15 June frame range. Flagged
  as a real, disclosed range/peak mismatch in the KG node's own description, not silently corrected
  elsewhere.
- `dammam_dust`'s date (17 May) was cross-checked against `wind10_speed`/`wind850_speed` at Dammam and
  confirmed to be the window's actual peak (8.5 m/s / 13.9 m/s), not just the caption's stated day.
- `haboob_dust` (a genuinely national event, no city-level signal) was deliberately given **no**
  Region-node location rather than forced onto the nearest city — matches
  `model/12_timeline_manifest.py`'s own `focus_points=None` treatment of this event.

**Bugs this surfaced and fixed (in order found):**
1. `agent/tools.py`'s `_parse_event_location` had no error handling and crashed (`IndexError`) the
   instant `similar_events_tool` was called for ANY hazard, because it iterates over every Event node
   of that hazard with no fallback — the new events' location strings didn't match the old
   "Name (LATN,LONE)" pattern it expects. Fixed at both ends: the new events' `location` field was
   made to follow that exact pattern (multi-city events like `flood_mar` keep only the primary
   region in `location`, with the rest described in the node's own `desc` text instead of appended
   into a string the parser would choke on); `_parse_event_location` was also made to return `None`
   on a genuinely non-point location (e.g. `haboob_dust`) instead of crashing, so the tool excludes
   it with a clear reason instead of failing the whole call.
2. `similar_events_tool` also crashed with `KeyError: 'value'` — the new events had no `value` field
   (the auto-detected events' `"peak_var peak_val unit"` format doesn't fit a verified-event's real
   story). Fixed by giving every verified event a `value` field of `"verdict: {verdict text}"`, and
   made the tool's own access defensive (`.get()`) so a future Event node missing this field degrades
   gracefully instead of crashing the whole call.
3. **Pre-existing, unrelated drift discovered while regenerating:** the *committed* `kg_data.json`'s
   `dust_storm` Hazard node had `label="Dust Storm"` and a custom `desc`, but the current
   `01_build_structural_kg.py` script's `HAZARDS` dict would generate `label="Dust Storm / Strong
   Wind"` instead — meaning the script, if run as-is, could NOT reproduce what was actually shipped.
   `agent/02_test_tools.py`'s `cap_alert_tool` XML check depends on the exact `"Dust Storm"` label and
   would have silently broken on any future regeneration. Fixed by aligning the script
   (`HAZARDS["dust_storm"] = "Dust Storm"` + a `HAZARD_DESC_OVERRIDE` dict preserving the richer desc)
   to match the previously-shipped, tested value — not by weakening the test.
4. `kg/02_make_dashboard.py`'s own `OUT_HTML` path pointed at `deploy/dashboard/kg_view.html`, a
   directory that has never existed — the live `deploy/kg_view.html` was not actually reproducible
   from this script as committed. Fixed the path to write directly to `deploy/kg_view.html`, matching
   where `index.html` actually links it.
5. `kg/02_make_dashboard.py`'s `NODE_COLORS`/`NODE_SHAPES`/`EDGE_COLORS` dicts were missing entries
   for `Citation` nodes and `grounded_by` edges (6 of each) — both existed in the data and rendered as
   generic grey fallback, invisible in the dashboard's own legend. Fixed by adding both.

**Verification (independent, not just re-running the same code):**
- `agent/02_test_tools.py`: updated 3 hardcoded per-hazard event-count assertions to match the new
  totals (flash_flood 3→8, heatwave 2→6, dust_storm 1→4 — old auto-detected counts plus the newly
  added site-verified events per hazard) — **169/169 pass**, including the pre-existing self-match,
  hyperlocal-distance, and Mecca cross-hazard-profile checks, all still passing unchanged since no
  existing event was touched.
- `FULL_SYSTEM_AUDIT.py`: updated the hardcoded `60 nodes / 183 edges` → `74 nodes / 264 edges`
  assertion (Section J), fixed Section C's event-count check to separately count the 6 auto-detected
  vs. 12 site-verified events (18 total) instead of crashing on the new value format, and **added a
  new Section C2** that independently re-derives every new event's `observed_value` driver readings
  directly from the RAW per-day source files (`E:\Data\New data\indicators\saudi_indicators_*.nc`) at
  that event's fixed date/region coordinates — the same independence standard already used for the 6
  auto-detected events, extended to cover the new content rather than left unchecked. **163/165 pass**;
  the only 2 failures are Section F's remote-vs-local GitHub sync check, which compares against the
  *currently pushed* `kg_data.json` (60/183) — expected to fail until this update is pushed, not a
  real defect.
- `kg_view.html` (the live interactive dashboard) and `img/kg_view_preview.png` (its static preview on
  `index.html`) regenerated via Playwright (headless Chromium screenshot, since the Browser pane
  wasn't displayed to take an interactive screenshot) and visually confirmed: all 12 new event stars
  present (including Hail/Buraidah as new triangle Region nodes), Citation/grounded_by now visible in
  the legend, header stat reads "Nodes 74  Edges 264" matching every other count above.
- `index.html`'s "60 nodes, 183 edges" text (2 places: image alt text + caption) updated to
  "74 nodes, 264 edges", with the caption also now explaining what the 18 Event nodes cover.

**Still pending:** commit + push (this update was NOT pushed automatically — same explicit-approval
policy as the first push; Section F's 2 audit failures will resolve on push, not before).

---

## Pending / not yet started

- Git commit/push for the Knowledge Graph update above — awaiting explicit user go-ahead.
- Sharing results with the teacher — deferred, decision pending
