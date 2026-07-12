# =============================================================================
# MAZU — Layer 3: Literature corpus for causal KG extraction
#
# Every entry below is a concise, faithful paraphrase of material found via
# live web search of real, citable publications (see `citation`/`url`).
# Nothing here is invented — the DeepSeek extraction step (02_extract_causal.py)
# is instructed to extract triples ONLY from this exact text, and every
# extracted triple is verified to trace back to a specific entry before it
# enters the knowledge graph.
# =============================================================================

CORPUS = [
    {
        "id": "ref_devries2013",
        "citation": "de Vries et al. (2013), Journal of Geophysical Research: Atmospheres",
        "title": "Extreme precipitation events in the Middle East: Dynamics of the Active Red Sea Trough",
        "url": "https://agupubs.onlinelibrary.wiley.com/doi/full/10.1002/jgrd.50569",
        "mechanism": "ARST",
        "text": (
            "The Active Red Sea Trough (ARST) is associated with extreme precipitation, "
            "flash floods, and severe societal impacts in the Middle East. The ARST "
            "involves six dynamical factors: a low-level trough (the Red Sea Trough), "
            "an anticyclone over the Arabian Peninsula (the Arabian Anticyclone), a "
            "transient midlatitude upper trough, an intensified subtropical jet stream, "
            "moisture transport pathways, and strong ascent resulting from tropospheric "
            "instability and synoptic-scale dynamical forcing. In the 25 November 2009 "
            "Jeddah flood, an eastward-migrating Mediterranean cyclone joined an extension "
            "of the Sudan low-pressure zone (the Red Sea Trough); a stationary anticyclone "
            "over the southeastern Arabian Peninsula (the Arabian Anticyclone) produced a "
            "clockwise flow that supplied moisture from the Arabian Sea and the Red Sea; "
            "upper-level tropospheric instability combined with this moisture supply to "
            "produce deep moist convection, resulting in a quasi-stationary mesoscale "
            "convective system and heavy rainfall over Jeddah. The Red Sea Trough enhances "
            "low-level moisture transport and convergence, often acting as a precursor to "
            "deep convection and mesoscale convective system development."
        ),
    },
    {
        "id": "ref_redsea_topo",
        "citation": "ResearchGate (peer-reviewed case study)",
        "title": "On the Effect of Red Sea and Topography on Rainfall over Saudi Arabia: Case Study",
        "url": "https://www.researchgate.net/publication/336928573",
        "mechanism": "moisture_transport",
        "text": (
            "Surface fluxes over the Red Sea play an important role in regulating "
            "low-level moist-air convergence prior to convection initiation and "
            "development; Red Sea water is estimated to affect regional rainfall by "
            "about 30 to 40 percent. In the absence of blocking topography, strong "
            "convergence areas form over the Red Sea that enhance uplift motion, "
            "strengthen low-level jets, and create a conditionally unstable atmosphere "
            "that favours cyclonic system development."
        ),
    },
    {
        "id": "ref_heatflux_era5",
        "citation": "MDPI Atmosphere (2019)",
        "title": "Surface Heat Fluxes over the Northern Arabian Gulf and the Northern Red Sea: "
                  "Evaluation of ECMWF-ERA5 and NASA-MERRA2 Reanalyses",
        "url": "https://www.mdpi.com/2073-4433/10/9/504",
        "mechanism": "moisture_transport",
        "text": (
            "Over the Red Sea and Arabian Gulf, stronger winds combined with drier air "
            "produce latent heat losses of up to 410 watts per square metre, indicating "
            "substantial moisture-related energy transfer from the sea surface to the "
            "atmosphere above these semi-enclosed seas."
        ),
    },
    {
        "id": "ref_heatwave_circulation",
        "citation": "ScienceDirect (peer-reviewed)",
        "title": "Analysis of extreme summer temperatures in Saudi Arabia and the association "
                  "with large-scale atmospheric circulation",
        "url": "https://www.sciencedirect.com/science/article/abs/pii/S0169809519301747",
        "mechanism": "subtropical_high",
        "text": (
            "Saudi Arabia experiences powerful solar radiation and persistent "
            "high-pressure systems that contribute to intense surface heating, "
            "especially during summer months. The influence of a subtropical "
            "high-pressure system over the Arabian Peninsula is associated with "
            "extreme summer temperature variation. This subtropical high-pressure "
            "system, combined with the region's geographic location and solar "
            "radiation exposure, is identified as a primary driver of extreme "
            "summer heat in Saudi Arabia."
        ),
    },
    {
        "id": "ref_heatwave_variability",
        "citation": "Springer, Meteorology and Atmospheric Physics (2024)",
        "title": "Observed heatwaves characteristics and variability over Saudi Arabia",
        "url": "https://link.springer.com/article/10.1007/s00703-024-01010-6",
        "mechanism": "subtropical_high",
        "text": (
            "Extreme heatwaves in Saudi Arabia are dominant in the northwestern "
            "region, while moderate and severe heatwaves occur less frequently "
            "along the Red Sea and Arabian Gulf coastal regions, where the "
            "adjacent sea moderates temperature extremes compared to the interior."
        ),
    },
    {
        "id": "ref_shamal_yu2016",
        "citation": "Yu et al. (2016), Journal of Geophysical Research: Atmospheres",
        "title": "Climatology of summer Shamal wind in the Middle East",
        "url": "https://agupubs.onlinelibrary.wiley.com/doi/full/10.1002/2015jd024063",
        "mechanism": "thermal_low",
        "text": (
            "The summer Shamal, occurring from May to August, is a persistent "
            "north-northwesterly wind regime driven by high pressure over eastern "
            "Africa and thermal low-pressure centres over Iran and Pakistan. It "
            "forms due to the steep pressure gradient between the heat low over "
            "Pakistan, Afghanistan and Iran and the high-pressure system over the "
            "eastern Mediterranean and northern Saudi Arabia. The summer Shamal is "
            "the major driver of dust storm activity across the Arabian Peninsula, "
            "capable of lifting dust and transporting it to the Persian Gulf and "
            "the Arabian Peninsula, with peak wind speeds typically occurring "
            "around midday."
        ),
    },
    {
        "id": "ref_redsea_coast_trends",
        "citation": "Nature Scientific Reports (2021)",
        "title": "Recent atmospheric changes and future projections along the Saudi Arabian "
                  "Red Sea Coast",
        "url": "https://www.nature.com/articles/s41598-021-04200-z",
        "mechanism": "moisture_transport",
        "text": (
            "Data for the Saudi Arabian Red Sea coast from 1979 to 2020 show "
            "significant positive trends in surface air temperature and wind "
            "speed, alongside significant negative trends in relative humidity "
            "and sea-level pressure, consistent with a warming, drying coastal "
            "atmosphere over this period."
        ),
    },
    # -------------------------------------------------------------------------
    # Entries below (2026-07-13) were added for the literature_evidence_tool
    # extension. Same methodology as above: each text is a faithful paraphrase
    # of a real, open-access paper, fetched and read directly (not from
    # search-snippet summaries) before being written here. Confirmed
    # accessible: all 5 are MDPI open-access articles, verified by loading the
    # actual abstract page.
    # -------------------------------------------------------------------------
    {
        "id": "ref_jeddah_uhi_addas2023",
        "citation": "Addas (2023), Land (MDPI)",
        "title": "Machine Learning Techniques to Map the Impact of Urban Heat Island: "
                  "Investigating the City of Jeddah",
        "url": "https://doi.org/10.3390/land12061159",
        "mechanism": "urban_heat_island",
        "text": (
            "In the megacity of Jeddah, Saudi Arabia, from 2000 to 2021, the "
            "urban area enlarged by about 80%, while the urban heat island (UHI) "
            "effect increased overall. Impervious surfaces significantly impact "
            "the UHI effect, while vegetation and water bodies have negative "
            "implications for the UHI effect. More than 80% of the total parts "
            "in Jeddah have been classified by extremely high UHI conditions, "
            "with the megacity's south, north, and central-east parts "
            "categorized by very high UHI conditions."
        ),
    },
    {
        "id": "ref_sw_orographic_alharbi2026",
        "citation": "Alharbi (2026), Atmosphere (MDPI)",
        "title": "Development of Region-Specific Rainfall Design Storms Using Machine "
                  "Learning in Southwestern Saudi Arabia",
        "url": "https://doi.org/10.3390/atmos17050443",
        "mechanism": "orographic_lifting",
        "text": (
            "The mountainous southwest of Saudi Arabia exhibits complex, highly "
            "seasonal precipitation driven by Indian Ocean monsoon inflows and "
            "orographic lifting. Cluster analysis of 8972 rainfall events "
            "recorded at 151 gauges found two primary storm clusters -- one "
            "with early, intense peaks and another with later peak intensities "
            "-- broadly reflecting windward versus leeward storm behavior. High "
            "risk from hydrologic extremes has been reported in this region, "
            "including local flash floods and short-duration high-intensity "
            "convective storms."
        ),
    },
    {
        "id": "ref_dust_transport_alzaid2024",
        "citation": "Alzaid, Anil & Aga (2024), Atmosphere (MDPI)",
        "title": "Simulation and Assessment of Episodic Dust Storms in Eastern Saudi "
                  "Arabia Using HYSPLIT Trajectory Model and Satellite Observations",
        "url": "https://doi.org/10.3390/atmos15121515",
        "mechanism": "cross_border_dust_transport",
        "text": (
            "The 'dust belt' region extending from the western Sahara to the "
            "Gobi Desert frequently generates severe dust storms that cause "
            "hazardous air quality. Using the HYSPLIT trajectory model "
            "calibrated for eastern Saudi Arabia (R-squared of 0.9965 between "
            "measured and predicted values), source apportionment via air "
            "particle backward trajectories and frequency analyses indicated "
            "that the northern regions, specifically Iraq and Syria, were the "
            "primary sources of the severe dust storms observed in the "
            "receptor area of eastern Saudi Arabia."
        ),
    },
    {
        "id": "ref_sst_teleconnection_almaashi2024",
        "citation": "Almaashi, Hasanean & Labban (2024), Atmosphere (MDPI)",
        "title": "Long-Term Teleconnections Between Global Circulation Patterns and "
                  "Interannual Variability of Surface Air Temperature over Kingdom "
                  "of Saudi Arabia",
        "url": "https://doi.org/10.3390/atmos15111310",
        "mechanism": "sst_teleconnection",
        "text": (
            "Surface air temperature (SAT) variability over the Kingdom of Saudi "
            "Arabia reveals significant warming trends, particularly from 1994 "
            "onward. The first empirical orthogonal function mode explains more "
            "than 45% of the variance and is correlated with global sea surface "
            "temperature (SST) patterns. The Pacific Warm Pool (PWP) "
            "consistently enhances surface warming in both winter and summer "
            "seasons through upper air convergence, and the Tropical North "
            "Atlantic (TNA) SST contributes warming over central and northern "
            "parts of Saudi Arabia during both winter and summer seasons."
        ),
    },
    {
        "id": "ref_jeddah2022_flood_sofia2024",
        "citation": "Sofia et al. (2024), Water (MDPI)",
        "title": "A Nationwide Flood Forecasting System for Saudi Arabia: Insights "
                  "from the Jeddah 2022 Event",
        "url": "https://doi.org/10.3390/w16141939",
        "mechanism": "orographic_lifting",
        "text": (
            "Saudi Arabia is threatened by recurrent flash floods caused by "
            "extreme precipitation events. An extreme precipitation event "
            "struck Jeddah on 24 November 2022, recorded as the heaviest "
            "rainfall in the region's history, leading to widespread flash "
            "floods and highlighting the critical need for accurate and "
            "timely forecasting."
        ),
    },
]


def get_by_mechanism(mechanism_id):
    return [c for c in CORPUS if c["mechanism"] == mechanism_id]


if __name__ == "__main__":
    print(f"Corpus entries: {len(CORPUS)}")
    for c in CORPUS:
        print(f"  {c['id']:26s} [{c['mechanism']:20s}] {c['citation']}")
