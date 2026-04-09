# This code was written with the assistance of Claude (Anthropic).

import json
import csv
from pyproj import Transformer

INPUT_PATH = "NYC_Pedestrian_Model/NYC_pednetwork_estimates_counts_2018-2019.geojson"
OUTPUT_CSV = "nyc_ped_endpoints.csv"

transformer = Transformer.from_crs("EPSG:6538", "EPSG:4326", always_xy=True)

print("Reading GeoJSON...")
with open(INPUT_PATH, "r") as f:
    data = json.load(f)

features = data["features"]
print(f"Loaded {len(features):,} features")

print("Extracting endpoints...")
fieldnames = [
    "start_lat", "start_lon", "end_lat", "end_lon",
    "predwkdyAM", "predwkdyMD", "predwkdyPM",
    "predwkndAM", "predwkndMD", "predwkndPM",
    "avg_daily", "avg_annual",
]

rows = []
for feat in features:
    coords = feat["geometry"]["coordinates"]
    props = feat["properties"]

    if len(coords) < 2:
        continue

    # Start and end points in EPSG:6538
    sx, sy = coords[0][0], coords[0][1]
    ex, ey = coords[-1][0], coords[-1][1]

    # Reproject to WGS84
    s_lon, s_lat = transformer.transform(sx, sy)
    e_lon, e_lat = transformer.transform(ex, ey)

    # Compute daily/annual averages
    wkdy = (props.get("predwkdyAM", 0) or 0) + (props.get("predwkdyMD", 0) or 0) + (props.get("predwkdyPM", 0) or 0)
    wknd = (props.get("predwkndAM", 0) or 0) + (props.get("predwkndMD", 0) or 0) + (props.get("predwkndPM", 0) or 0)
    avg_daily = round((5 * wkdy + 2 * wknd) / 7.0, 1)
    avg_annual = int(avg_daily * 365)

    rows.append({
        "start_lat": round(s_lat, 6),
        "start_lon": round(s_lon, 6),
        "end_lat": round(e_lat, 6),
        "end_lon": round(e_lon, 6),
        "predwkdyAM": props.get("predwkdyAM", 0),
        "predwkdyMD": props.get("predwkdyMD", 0),
        "predwkdyPM": props.get("predwkdyPM", 0),
        "predwkndAM": props.get("predwkndAM", 0),
        "predwkndMD": props.get("predwkndMD", 0),
        "predwkndPM": props.get("predwkndPM", 0),
        "avg_daily": avg_daily,
        "avg_annual": avg_annual,
    })

with open(OUTPUT_CSV, "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(rows)

print(f"Extracted {len(rows):,} segments to {OUTPUT_CSV}")
