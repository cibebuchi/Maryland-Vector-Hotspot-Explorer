import streamlit as st
import pandas as pd
import geopandas as gpd
import plotly.express as px
import numpy as np
from pathlib import Path
from fetch_and_prepare import (
    get_taxon_key,
    fetch_occurrences_for_taxon,
    assign_to_counties
)

# Point to your crown logo
logo_path = Path(__file__).parent / "logo.png"

# Use the crown image as the page icon
st.set_page_config(
    page_title="🐀🦟 Maryland Vector Hotspot Explorer",
    page_icon=str(logo_path),
    layout="wide"
)

st.title("🐀🦟 Maryland Vector Hotspot Explorer")

st.markdown(
    "This app fetches real-time GBIF data and visualizes spatial and temporal hotspots "
    "for disease-related vectors in Maryland."
)

# ---------------------- Sidebar ------------------------
species_list = [
    "Rattus norvegicus",
    "Aedes aegypti",
    "Aedes albopictus"
]
selected_species = st.sidebar.selectbox("Select Species", species_list)

year_range = st.sidebar.slider("Select Year Range", 2000, 2025, (2020, 2023))
show_monthly = st.sidebar.checkbox("Show Monthly Time Series", value=True)
compare_counties = st.sidebar.checkbox("Compare Counties")

# Initialize session_state for gdf_joined and county_selection
if "gdf_joined" not in st.session_state:
    st.session_state.gdf_joined = None
if "county_selection" not in st.session_state:
    st.session_state.county_selection = []

# Build county selector conditionally after data loads
county_selection = []
if compare_counties:
    if st.session_state.gdf_joined is None:
        st.sidebar.info("Run analysis first to load counties for comparison.")
    else:
        county_options = sorted(st.session_state.gdf_joined["NAME_2"].dropna().unique())
        county_selection = st.sidebar.multiselect(
            "Select Two Counties for Comparison",
            options=county_options,
            max_selections=2,
            default=st.session_state.county_selection
        )
        st.session_state.county_selection = county_selection

# -------------------- Main Execution -------------------
if st.sidebar.button("Run Analysis"):
    st.info("Fetching live GBIF data…")
    try:
        key = get_taxon_key(selected_species)
        df_occ = fetch_occurrences_for_taxon(
            taxon_key=key,
            country="US",
            state_province="Maryland",
            year_from=year_range[0],
            year_to=year_range[1]
        )
        df_occ["queried_scientificName"] = selected_species
    except Exception as e:
        st.error(f"Failed to fetch {selected_species}: {e}")
        st.stop()

    if df_occ.empty:
        st.warning("No records retrieved. Try adjusting filters.")
        st.stop()

    # Clean and prepare date fields
    df_occ = df_occ.dropna(subset=["year", "month"])
    df_occ["year"] = df_occ["year"].astype(int)
    df_occ["month"] = df_occ["month"].astype(int)
    df_occ["year_month"] = pd.to_datetime(
        dict(year=df_occ["year"], month=df_occ["month"], day=1),
        errors="coerce"
    )

    # Spatial join to counties
    gdf_joined = assign_to_counties(df_occ)
    st.session_state.gdf_joined = gdf_joined

    # Compute per-county counts & normalize to a 0–1 Risk Score
    county_counts = (
        gdf_joined
        .groupby(["NAME_2", "queried_scientificName"])
        .size()
        .reset_index(name="occurrence_count")
    )
    county_counts["Risk Score"] = county_counts["occurrence_count"].apply(np.log1p)
    county_counts["Risk Score"] /= county_counts["Risk Score"].max()
    county_counts["Risk Score"] = county_counts["Risk Score"].round(2)

    # If comparing two counties, show their scores up front
    if compare_counties and len(county_selection) == 2:
        scores = county_counts.set_index("NAME_2")["Risk Score"].to_dict()
        st.markdown(
            f"**{county_selection[0]} Risk Score:** {scores.get(county_selection[0], 0):.2f}  &nbsp;&nbsp; "
            f"**{county_selection[1]} Risk Score:** {scores.get(county_selection[1], 0):.2f}"
        )

    st.success(f"Fetched {len(df_occ)} occurrence records.")

    # -------------------- Hotspot Map --------------------
    st.subheader("Hotspot Map")
    county_geo = gpd.read_file("md_counties.geojson")
    merged = county_geo.merge(
        county_counts,
        on="NAME_2",
        how="left"
    ).fillna({"Risk Score": 0, "occurrence_count": 0})

    fig_map = px.choropleth_mapbox(
        merged,
        geojson=merged.geometry.__geo_interface__,
        locations=merged.index,
        color="Risk Score",
        hover_name="NAME_2",
        hover_data=["queried_scientificName", "occurrence_count"],
        mapbox_style="carto-positron",
        center={"lat": 39.0, "lon": -76.8},
        zoom=6.5,
        color_continuous_scale="Reds",
        opacity=0.6
    )
    fig_map.update_layout(margin={"r": 0, "t": 0, "l": 0, "b": 0})
    st.plotly_chart(fig_map, use_container_width=True)

    # -------------------- Monthly Chart --------------------
    if show_monthly:
        st.subheader("Monthly Presence Over Time")
        if compare_counties and len(county_selection) == 2:
            monthly_counts = (
                gdf_joined[gdf_joined["NAME_2"].isin(county_selection)]
                .groupby(["year_month", "queried_scientificName", "NAME_2"])
                .size()
                .reset_index(name="count")
            )
            fig_month = px.line(
                monthly_counts,
                x="year_month",
                y="count",
                color="NAME_2",
                line_dash="NAME_2",
                title="Monthly Occurrence Comparison by County",
                labels={"count": "Occurrences", "year_month": "Date"}
            )
        else:
            monthly_counts = (
                df_occ
                .groupby(["year_month", "queried_scientificName"])
                .size()
                .reset_index(name="count")
            )
            fig_month = px.line(
                monthly_counts,
                x="year_month",
                y="count",
                color="queried_scientificName",
                title="Monthly Occurrence Counts",
                labels={"count": "Occurrences", "year_month": "Date"}
            )
        fig_month.update_xaxes(type="date")
        st.plotly_chart(fig_month, use_container_width=True)

    # -------------------- Annual Chart --------------------
    st.subheader("Annual Presence Summary")
    if compare_counties and len(county_selection) == 2:
        annual_counts = (
            gdf_joined[gdf_joined["NAME_2"].isin(county_selection)]
            .groupby(["year", "NAME_2"])
            .size()
            .reset_index(name="count")
        )
        fig_annual = px.line(
            annual_counts,
            x="year",
            y="count",
            color="NAME_2",
            line_dash="NAME_2",
            markers=True,
            title="Annual Occurrence Comparison by County",
            labels={"count": "Occurrences", "year": "Year"}
        )
    else:
        annual_counts = (
            df_occ
            .groupby(["year", "queried_scientificName"])
            .size()
            .reset_index(name="count")
        )
        fig_annual = px.bar(
            annual_counts,
            x="year",
            y="count",
            color="queried_scientificName",
            barmode="group",
            title="Annual Occurrence Counts by Species",
            labels={"count": "Occurrences", "year": "Year"}
        )
    st.plotly_chart(fig_annual, use_container_width=True)
