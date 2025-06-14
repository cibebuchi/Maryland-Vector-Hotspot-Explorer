#!/usr/bin/env python3
import requests
import pandas as pd
import time
import geopandas as gpd
from shapely.geometry import Point

GBIF_BASE = "https://api.gbif.org/v1"
MD_COUNTIES_PATH = "md_counties.geojson"

def get_taxon_key(scientific_name: str) -> int:
    url = f"{GBIF_BASE}/species/match"
    params = {"name": scientific_name}
    resp = requests.get(url, params=params)
    resp.raise_for_status()
    result = resp.json()
    if "usageKey" in result:
        return result["usageKey"]
    else:
        raise ValueError(f"GBIF did not return a usageKey for '{scientific_name}'")

def fetch_occurrences_for_taxon(taxon_key: int, country: str, state_province: str,
                                 year_from: int = None, year_to: int = None,
                                 limit: int = 300, sleep_between_requests: float = 0.2) -> pd.DataFrame:
    all_records = []
    offset = 0
    while True:
        params = {
            "taxonKey": taxon_key,
            "country": country,
            "stateProvince": state_province,
            "hasCoordinate": "true",
            "limit": limit,
            "offset": offset
        }
        if year_from is not None and year_to is not None:
            params["year"] = f"{year_from},{year_to}"
        url = f"{GBIF_BASE}/occurrence/search"
        resp = requests.get(url, params=params)
        resp.raise_for_status()
        data = resp.json()
        results = data.get("results", [])
        if not results:
            break
        for rec in results:
            lat = rec.get("decimalLatitude")
            lon = rec.get("decimalLongitude")
            year = rec.get("year")
            month = rec.get("month")
            species_name = rec.get("species")
            event_date = rec.get("eventDate")
            all_records.append({
                "species": species_name,
                "decimalLatitude": lat,
                "decimalLongitude": lon,
                "year": year,
                "month": month,
                "eventDate": event_date
            })
        if len(results) < limit:
            break
        offset += limit
        time.sleep(sleep_between_requests)
    df = pd.DataFrame(all_records)
    return df

def assign_to_counties(df, counties_path=MD_COUNTIES_PATH):
    gdf = gpd.GeoDataFrame(
        df.dropna(subset=["decimalLatitude", "decimalLongitude"]).copy(),
        geometry=[Point(xy) for xy in zip(df["decimalLongitude"], df["decimalLatitude"])],
        crs="EPSG:4326"
    )
    counties = gpd.read_file(counties_path)
    joined = gpd.sjoin(gdf, counties, predicate="within", how="inner")
    return joined

def compute_hotspot_scores(joined_df):
    if joined_df.empty:
        return pd.DataFrame()
    grouped = joined_df.groupby(["NAME_2", "queried_scientificName"]).size().reset_index(name="occurrence_count")
    grouped["hotspot_score"] = grouped.groupby("queried_scientificName")["occurrence_count"].transform(
        lambda x: (x - x.min()) / (x.max() - x.min()) if x.max() != x.min() else 0)
    return grouped

def main():
    species_list = [
        "Rattus norvegicus",        
        "Aedes aegypti",     
        "Aedes albopictus",           
        "Culex pipiens"        
    ]
    YEAR_FROM = 2020
    YEAR_TO = 2023
    combined_dfs = []

    for sci_name in species_list:
        print(f"Resolving taxonKey for {sci_name}")
        try:
            key = get_taxon_key(sci_name)
        except Exception as e:
            print(f"Error resolving {sci_name}: {e}")
            continue
        df_occ = fetch_occurrences_for_taxon(
            taxon_key=key,
            country="US",
            state_province="Maryland",
            year_from=YEAR_FROM,
            year_to=YEAR_TO
        )
        df_occ["queried_scientificName"] = sci_name
        combined_dfs.append(df_occ)

    all_data = pd.concat(combined_dfs, ignore_index=True)
    all_data.to_csv("All_MD_occurrences.csv", index=False)

    joined = assign_to_counties(all_data)
    joined.to_csv("MD_occurrences_with_counties.csv", index=False)

    hotspots = compute_hotspot_scores(joined)
    hotspots.to_csv("MD_hotspot_scores.csv", index=False)

if __name__ == "__main__":
    main()
