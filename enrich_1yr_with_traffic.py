#!/usr/bin/env python3
# This code was written with the assistance of Claude (Anthropic).

import os
import re
import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

BASE = "/scratch/gpfs/ALAINK/Suthi"
YOLO_DIR = os.path.join(BASE, "yolo_feature_detector")

# Existing 1-year YOLO features with crash counts
PER_IMAGE_CSV = os.path.join(YOLO_DIR, "feature_prevalence_full_m_per_image_with_actual_counts.csv")

# Traffic data sources
AADT_CSV = os.path.join(BASE, "nyc_aadt_2023.csv")
PED_CSV = os.path.join(BASE, "nyc_ped_avg_annual_counts.csv")

# Output
OUTPUT_CSV = os.path.join(YOLO_DIR, "feature_prevalence_1yr_with_traffic.csv")

# Thresholds (same as find_intersections_all_three.py)
AADT_THRESHOLD_M = 150
PED_THRESHOLD_M = 50

M_PER_DEG_LAT = 111_320
M_PER_DEG_LON = 111_320 * np.cos(np.radians(40.7))

COORD_RE = re.compile(r"chip_\d+_([-\d.]+)_([-\d.]+)\.\w+$", re.I)


def latlon_to_meters(lat, lon):
    x = (lon + 74.0) * M_PER_DEG_LON
    y = (lat - 40.5) * M_PER_DEG_LAT
    return x, y


def parse_coords(path):
    m = COORD_RE.search(os.path.basename(path))
    if not m:
        return None, None
    return float(m.group(1)), float(m.group(2))


def main():
    # 1. Load per-image YOLO features (1-year, already has crash_count_actual)
    print("Loading per-image YOLO features...")
    df = pd.read_csv(PER_IMAGE_CSV)
    print(f"  {len(df)} rows")

    df[["lat", "lon"]] = df["path"].apply(lambda p: pd.Series(parse_coords(p)))
    df = df.dropna(subset=["lat", "lon"])

    # Unique image locations
    unique_locs = df[["lat", "lon"]].drop_duplicates()
    print(f"  {len(unique_locs)} unique image locations")

    # 2. Load AADT
    print("Loading AADT...")
    aadt = pd.read_csv(AADT_CSV).dropna(subset=["latitude", "longitude", "aadt"])
    print(f"  {len(aadt)} AADT segments")

    # 3. Load pedestrian data
    print("Loading pedestrian data...")
    ped = pd.read_csv(PED_CSV).dropna(subset=["latitude", "longitude"])
    print(f"  {len(ped)} pedestrian segments")

    # 4. Spatial matching via KD-Trees
    print("Building spatial indices and matching...")
    aadt_xy = np.array([latlon_to_meters(r.latitude, r.longitude) for _, r in aadt.iterrows()])
    ped_xy = np.array([latlon_to_meters(r.latitude, r.longitude) for _, r in ped.iterrows()])
    chip_xy = np.array([latlon_to_meters(lat, lon) for lat, lon in unique_locs.values])

    tree_aadt = cKDTree(aadt_xy)
    tree_ped = cKDTree(ped_xy)

    dist_aadt, idx_aadt = tree_aadt.query(chip_xy)
    dist_ped, idx_ped = tree_ped.query(chip_xy)

    unique_locs = unique_locs.reset_index(drop=True)
    unique_locs["aadt_dist_m"] = dist_aadt
    unique_locs["ped_dist_m"] = dist_ped

    mask = (unique_locs["aadt_dist_m"] <= AADT_THRESHOLD_M) & \
           (unique_locs["ped_dist_m"] <= PED_THRESHOLD_M)

    print(f"  AADT within {AADT_THRESHOLD_M}m: {(unique_locs['aadt_dist_m'] <= AADT_THRESHOLD_M).sum()}")
    print(f"  Ped within {PED_THRESHOLD_M}m: {(unique_locs['ped_dist_m'] <= PED_THRESHOLD_M).sum()}")
    print(f"  Both: {mask.sum()}")

    matched_locs = unique_locs[mask].copy()

    # Attach traffic values
    aadt_reset = aadt.reset_index(drop=True)
    ped_reset = ped.reset_index(drop=True)
    matched_locs["AADT"] = aadt_reset.loc[idx_aadt[mask], "aadt"].values
    matched_locs["Ped_avg_daily"] = ped_reset.loc[idx_ped[mask], "avg_daily"].values

    # 5. Join traffic onto per-image features using coords
    #    Round coords for safe merge
    matched_locs["lat_r"] = matched_locs["lat"].round(6)
    matched_locs["lon_r"] = matched_locs["lon"].round(6)
    traffic_lookup = matched_locs[["lat_r", "lon_r", "AADT", "Ped_avg_daily"]].copy()

    df["lat_r"] = df["lat"].round(6)
    df["lon_r"] = df["lon"].round(6)

    merged = df.merge(traffic_lookup, on=["lat_r", "lon_r"], how="inner")
    merged = merged.drop(columns=["lat", "lon", "lat_r", "lon_r"])

    print(f"\nEnriched result: {len(merged)} images at {merged[['path']].apply(lambda r: parse_coords(r['path']), axis=1).nunique()} unique locations")
    print(f"  crash_class distribution:")
    print(merged["crash_class"].value_counts().to_string())
    print(f"\n  crash_count_actual distribution:")
    if "crash_count_actual" in merged.columns:
        print(merged["crash_count_actual"].value_counts().sort_index().to_string())
    print(f"\n  AADT range: {merged['AADT'].min():.0f} – {merged['AADT'].max():.0f}")
    print(f"  Ped range: {merged['Ped_avg_daily'].min():.0f} – {merged['Ped_avg_daily'].max():.0f}")

    merged.to_csv(OUTPUT_CSV, index=False)
    print(f"\nSaved: {OUTPUT_CSV}")


if __name__ == "__main__":
    main()
