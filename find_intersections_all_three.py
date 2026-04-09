# This code was written with the assistance of Claude (Anthropic).

import os
import re
import glob
import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

# === Config ===
CHIP_DIR = "ChippedImages_5Years"
CATEGORIES = ["safe", "1_crash", "2_crashes", "3plus_crashes"]
PED_CSV = "nyc_ped_avg_annual_counts.csv"
AADT_CSV = "nyc_aadt_2023.csv"
OUTPUT_CSV = "intersections_with_all_three.csv"

# Max distance in meters to consider a match
# AADT segments are road-level, so 150m is reasonable
# Ped segments are very dense, so 50m should work
AADT_THRESHOLD_M = 150
PED_THRESHOLD_M = 50

# Approximate meters per degree at NYC latitude (~40.7)
M_PER_DEG_LAT = 111_320
M_PER_DEG_LON = 111_320 * np.cos(np.radians(40.7))  # ~84,400

def latlon_to_meters(lat, lon, ref_lat=40.7):
    """Convert lat/lon to approximate local meter coords for distance calc."""
    x = (lon + 74.0) * M_PER_DEG_LON  # shift near 0
    y = (lat - 40.5) * M_PER_DEG_LAT
    return x, y


# ============================================================
# 1. Extract image chip locations from filenames
# ============================================================
print("Scanning image chips...")
chip_data = []
pattern = re.compile(r"chip_(\d+)_([-\d.]+)_([-\d.]+)\.tif")

for cat in CATEGORIES:
    cat_dir = os.path.join(CHIP_DIR, cat)
    if not os.path.isdir(cat_dir):
        print(f"  Warning: {cat_dir} not found, skipping")
        continue
    for fname in os.listdir(cat_dir):
        m = pattern.match(fname)
        if m:
            chip_id = m.group(1)
            lat = float(m.group(2))
            lon = float(m.group(3))
            chip_data.append({
                "chip_id": chip_id,
                "chip_lat": lat,
                "chip_lon": lon,
                "category": cat,
                "filename": fname,
            })

chips = pd.DataFrame(chip_data)

# Deduplicate by location (same intersection may appear in multiple categories — keep one)
# Actually, same chip_id in different categories means different intersections
# But same lat/lon could appear; let's keep unique lat/lon pairs
chips_unique = chips.drop_duplicates(subset=["chip_lat", "chip_lon"]).copy()
print(f"  Total chips: {len(chips):,}")
print(f"  Unique locations: {len(chips_unique):,}")

# ============================================================
# 2. Load vehicle AADT data
# ============================================================
print("Loading AADT data...")
aadt = pd.read_csv(AADT_CSV)
aadt = aadt.dropna(subset=["latitude", "longitude", "aadt"])
print(f"  AADT segments: {len(aadt):,}")

# ============================================================
# 3. Load pedestrian count data
# ============================================================
print("Loading pedestrian data...")
ped = pd.read_csv(PED_CSV)
ped = ped.dropna(subset=["latitude", "longitude"])
print(f"  Pedestrian segments: {len(ped):,}")

# ============================================================
# 4. Build KD-Trees for fast nearest-neighbor lookup
# ============================================================
print("Building spatial indices...")

# Convert to meter coordinates
aadt_xy = np.array([latlon_to_meters(r.latitude, r.longitude) for _, r in aadt.iterrows()])
ped_xy = np.array([latlon_to_meters(r.latitude, r.longitude) for _, r in ped.iterrows()])
chip_xy = np.array([latlon_to_meters(r.chip_lat, r.chip_lon) for _, r in chips_unique.iterrows()])

tree_aadt = cKDTree(aadt_xy)
tree_ped = cKDTree(ped_xy)

# ============================================================
# 5. For each chip, find nearest AADT and nearest ped segment
# ============================================================
print("Finding nearest matches for each chip...")

dist_aadt, idx_aadt = tree_aadt.query(chip_xy)
dist_ped, idx_ped = tree_ped.query(chip_xy)

chips_unique = chips_unique.reset_index(drop=True)
chips_unique["aadt_dist_m"] = dist_aadt
chips_unique["aadt_idx"] = idx_aadt
chips_unique["ped_dist_m"] = dist_ped
chips_unique["ped_idx"] = idx_ped

# ============================================================
# 6. Filter to matches within thresholds
# ============================================================
mask = (chips_unique["aadt_dist_m"] <= AADT_THRESHOLD_M) & \
       (chips_unique["ped_dist_m"] <= PED_THRESHOLD_M)

matched = chips_unique[mask].copy()
print(f"\nMatching results:")
print(f"  Chips with AADT within {AADT_THRESHOLD_M}m: {(chips_unique['aadt_dist_m'] <= AADT_THRESHOLD_M).sum():,}")
print(f"  Chips with ped within {PED_THRESHOLD_M}m: {(chips_unique['ped_dist_m'] <= PED_THRESHOLD_M).sum():,}")
print(f"  Chips with BOTH: {len(matched):,}")

# Join AADT fields
aadt_reset = aadt.reset_index(drop=True)
matched["aadt"] = aadt_reset.loc[matched["aadt_idx"].values, "aadt"].values
matched["aadt_route"] = aadt_reset.loc[matched["aadt_idx"].values, "route"].values
matched["aadt_description"] = aadt_reset.loc[matched["aadt_idx"].values, "description"].values
matched["aadt_county"] = aadt_reset.loc[matched["aadt_idx"].values, "county"].values
matched["aadt_functional_class"] = aadt_reset.loc[matched["aadt_idx"].values, "functional_class"].values

# Join pedestrian fields
ped_reset = ped.reset_index(drop=True)
matched["ped_avg_daily"] = ped_reset.loc[matched["ped_idx"].values, "avg_daily"].values
matched["ped_avg_annual"] = ped_reset.loc[matched["ped_idx"].values, "avg_annual"].values

# Also get all categories for each location
loc_cats = chips.groupby(["chip_lat", "chip_lon"])["category"].apply(lambda x: ";".join(sorted(set(x)))).reset_index()
loc_cats.columns = ["chip_lat", "chip_lon", "all_categories"]
matched = matched.merge(loc_cats, on=["chip_lat", "chip_lon"], how="left")

# ============================================================
# 7. Output
# ============================================================
out_cols = [
    "chip_lat", "chip_lon", "chip_id", "category", "all_categories", "filename",
    "aadt", "aadt_dist_m", "aadt_route", "aadt_description", "aadt_county", "aadt_functional_class",
    "ped_avg_daily", "ped_avg_annual", "ped_dist_m",
]

out = matched[out_cols].copy()
out["aadt_dist_m"] = out["aadt_dist_m"].round(1)
out["ped_dist_m"] = out["ped_dist_m"].round(1)
out = out.sort_values("aadt", ascending=False).reset_index(drop=True)

out.to_csv(OUTPUT_CSV, index=False)

print(f"\n=== Summary ===")
print(f"Intersections with image + vehicle + pedestrian data: {len(out):,}")
print(f"\nBy category:")
for cat in CATEGORIES:
    n = out["all_categories"].str.contains(cat).sum()
    print(f"  {cat}: {n:,}")
print(f"\nAADT range: {out['aadt'].min():,} – {out['aadt'].max():,}")
print(f"Ped avg daily range: {out['ped_avg_daily'].min():.0f} – {out['ped_avg_daily'].max():.0f}")
print(f"\nBy county:")
print(out["aadt_county"].value_counts().to_string())
print(f"\nTop 10 by vehicle AADT:")
for _, r in out.head(10).iterrows():
    print(f"  AADT={int(r['aadt']):>7}  Ped={int(r['ped_avg_daily']):>5}/day  {r['aadt_route']:<25}  ({r['chip_lat']:.4f}, {r['chip_lon']:.4f})")

print(f"\nSaved to {OUTPUT_CSV}")
