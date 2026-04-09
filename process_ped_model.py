# This code was written with the assistance of Claude (Anthropic).

import json
import pandas as pd
import numpy as np
from pyproj import Transformer

INPUT_PATH = "NYC_Pedestrian_Model/NYC_pednetwork_estimates_counts_2018-2019.geojson"
OUTPUT_PATH = "nyc_ped_avg_annual_counts.csv"

# --- Read and parse GeoJSON ---
print("Reading GeoJSON (this may take a moment for 263 MB)...")
with open(INPUT_PATH, "r") as f:
    data = json.load(f)

features = data["features"]
print(f"Loaded {len(features):,} features")

# --- Set up coordinate transformer: EPSG:6538 -> WGS84 ---
transformer = Transformer.from_crs("EPSG:6538", "EPSG:4326", always_xy=True)

# --- Extract properties + midpoints ---
print("Extracting properties and computing midpoints...")
rows = []
for feat in features:
    props = feat["properties"]
    coords = feat["geometry"]["coordinates"]

    # Compute midpoint along the LineString (geometric midpoint of middle segment)
    n = len(coords)
    if n == 1:
        mx, my = coords[0][0], coords[0][1]
    elif n == 2:
        mx = (coords[0][0] + coords[1][0]) / 2.0
        my = (coords[0][1] + coords[1][1]) / 2.0
    else:
        # Walk cumulative distances to find the halfway point
        dists = [0.0]
        for i in range(1, n):
            dx = coords[i][0] - coords[i - 1][0]
            dy = coords[i][1] - coords[i - 1][1]
            dists.append(dists[-1] + (dx * dx + dy * dy) ** 0.5)
        half = dists[-1] / 2.0
        for i in range(1, n):
            if dists[i] >= half:
                frac = (half - dists[i - 1]) / (dists[i] - dists[i - 1]) if dists[i] != dists[i - 1] else 0.0
                mx = coords[i - 1][0] + frac * (coords[i][0] - coords[i - 1][0])
                my = coords[i - 1][1] + frac * (coords[i][1] - coords[i - 1][1])
                break

    # Reproject midpoint to lon/lat
    lon, lat = transformer.transform(mx, my)

    rows.append({
        "latitude": round(lat, 6),
        "longitude": round(lon, 6),
        "predwkdyAM": props.get("predwkdyAM", 0),
        "predwkdyMD": props.get("predwkdyMD", 0),
        "predwkdyPM": props.get("predwkdyPM", 0),
        "predwkndAM": props.get("predwkndAM", 0),
        "predwkndMD": props.get("predwkndMD", 0),
        "predwkndPM": props.get("predwkndPM", 0),
        "Shape_Leng": props.get("Shape_Leng", 0),
    })

df = pd.DataFrame(rows)

# --- Compute average daily pedestrian volume ---
# weekday_daily = AM + midday + PM;  weekend_daily = AM + midday + PM
# avg_daily = (5*weekday + 2*weekend) / 7;  avg_annual = avg_daily * 365
df["weekday_daily"] = df["predwkdyAM"] + df["predwkdyMD"] + df["predwkdyPM"]
df["weekend_daily"] = df["predwkndAM"] + df["predwkndMD"] + df["predwkndPM"]
df["avg_daily"] = ((5 * df["weekday_daily"] + 2 * df["weekend_daily"]) / 7.0).round(1)
df["avg_annual"] = (df["avg_daily"] * 365).astype(int)

df = df.sort_values("avg_annual", ascending=False).reset_index(drop=True)

print(f"\nSample rows:\n{df.head(10).to_string()}")
print(f"\nTotal segments: {len(df):,}")
print(f"Avg daily range: {df['avg_daily'].min():.0f} – {df['avg_daily'].max():.0f}")
print(f"Avg annual range: {df['avg_annual'].min():,} – {df['avg_annual'].max():,}")

df.to_csv(OUTPUT_PATH, index=False)
print(f"\nSaved to {OUTPUT_PATH}")
