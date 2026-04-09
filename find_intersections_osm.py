# This code was written with the assistance of Claude (Anthropic).

import os
import re
import numpy as np
import pandas as pd
from scipy.spatial import cKDTree
import pyrosm

# === Config ===
OSM_PBF = "NY.osm.pbf"
NYC_BBOX = [-74.3, 40.49, -73.7, 40.92]  # [west, south, east, north]
CHIP_DIR = "ChippedImages_5Years"
CATEGORIES = ["safe", "1_crash", "2_crashes", "3plus_crashes"]
AADT_ENDPOINTS_CSV = "nyc_aadt_endpoints.csv"
PED_ENDPOINTS_CSV = "nyc_ped_endpoints.csv"
OUTPUT_CSV = "intersections_with_all_three.csv"

# Max snap distance for endpoint -> OSM node (meters)
SNAP_THRESHOLD_M = 100
# Max snap distance for chip -> OSM node (meters)
CHIP_SNAP_THRESHOLD_M = 100
# Max snap distance for ped endpoint -> OSM node (meters)
PED_SNAP_THRESHOLD_M = 100

# Approximate meters per degree at NYC latitude
M_PER_DEG_LAT = 111_320
M_PER_DEG_LON = 111_320 * np.cos(np.radians(40.7))


def to_xy(lat, lon):
    """Convert lat/lon arrays to approximate local meter coords for KD-tree."""
    x = np.asarray(lon) * M_PER_DEG_LON
    y = np.asarray(lat) * M_PER_DEG_LAT
    return np.column_stack([x, y])


# ============================================================
# 1. Load OSM intersection nodes
# ============================================================
print("Loading OSM driving network from PBF (NYC bounding box)...")
osm = pyrosm.OSM(OSM_PBF, bounding_box=NYC_BBOX)
nodes, edges = osm.get_network(network_type="driving", nodes=True)
print(f"  OSM nodes: {len(nodes):,}")
print(f"  OSM edges: {len(edges):,}")

# Extract node ID, lat, lon
if "lat" in nodes.columns and "lon" in nodes.columns:
    osm_nodes = nodes[["id", "lat", "lon"]].copy()
else:
    osm_nodes = nodes[["id"]].copy()
    osm_nodes["lon"] = nodes.geometry.x
    osm_nodes["lat"] = nodes.geometry.y

osm_nodes = osm_nodes.dropna(subset=["lat", "lon"]).reset_index(drop=True)
print(f"  OSM nodes with coords: {len(osm_nodes):,}")

# Build KD-tree for OSM nodes
osm_xy = to_xy(osm_nodes["lat"].values, osm_nodes["lon"].values)
tree_osm = cKDTree(osm_xy)

# ============================================================
# 2. Read AADT segment endpoints from pre-extracted CSV
# ============================================================
print("\nLoading AADT endpoint data...")

aadt_df = pd.read_csv(AADT_ENDPOINTS_CSV)
aadt_df = aadt_df.dropna(subset=["start_lat", "start_lon", "end_lat", "end_lon", "aadt"])
print(f"  AADT segments: {len(aadt_df):,}")

# Snap start & end endpoints to nearest OSM nodes
start_xy = to_xy(aadt_df["start_lat"].values, aadt_df["start_lon"].values)
end_xy = to_xy(aadt_df["end_lat"].values, aadt_df["end_lon"].values)

dist_start, idx_start = tree_osm.query(start_xy)
dist_end, idx_end = tree_osm.query(end_xy)

# Assign AADT to OSM nodes based on direction
#   Two-way (oneway != 'Y'): assign to BOTH start and end nodes
#   One-way (oneway == 'Y'): assign to END node only (downstream)
# node_aadt: { osm_node_id -> { 'aadt_sum': X, 'segments': [...] } }
node_aadt = {}

def add_to_node(osm_idx, dist_m, seg_info, endpoint_label):
    if dist_m > SNAP_THRESHOLD_M:
        return False
    nid = int(osm_nodes.iloc[osm_idx]["id"])
    if nid not in node_aadt:
        node_aadt[nid] = {"aadt_sum": 0, "segments": [], "su_truck_sum": 0, "cu_truck_sum": 0}
    node_aadt[nid]["aadt_sum"] += seg_info["aadt"]
    node_aadt[nid]["su_truck_sum"] += (seg_info["su_truck_aadt"] or 0)
    node_aadt[nid]["cu_truck_sum"] += (seg_info["cu_truck_aadt"] or 0)
    node_aadt[nid]["segments"].append({
        "route": seg_info["route"],
        "description": seg_info["description"],
        "county": seg_info["county"],
        "functional_class": seg_info["functional_class"],
        "stat_type": seg_info["stat_type"],
        "station_type": seg_info["station_type"],
        "aadt": seg_info["aadt"],
        "endpoint": endpoint_label,
        "snap_dist_m": round(dist_m, 1),
    })
    return True

assigned_start = 0
assigned_end = 0
for i, seg in aadt_df.iterrows():
    is_oneway = str(seg.get("oneway", "")).strip() == "Y"

    if is_oneway:
        # One-way: traffic flows start -> end, assign to END (downstream) node
        if add_to_node(idx_end[i], dist_end[i], seg, "end"):
            assigned_end += 1
    else:
        # Two-way: traffic passes through both intersections
        if add_to_node(idx_start[i], dist_start[i], seg, "start"):
            assigned_start += 1
        if add_to_node(idx_end[i], dist_end[i], seg, "end"):
            assigned_end += 1

print(f"  Endpoint-to-OSM-node assignments: {assigned_start} start + {assigned_end} end")
print(f"  Unique OSM nodes with AADT: {len(node_aadt):,}")

# Build AADT lookup dataframe keyed by OSM node ID
aadt_rows = []
for nid, info in node_aadt.items():
    # Use the highest-AADT segment's metadata as the representative description
    best_seg = max(info["segments"], key=lambda s: s["aadt"])
    aadt_rows.append({
        "osm_node_id": nid,
        "aadt": info["aadt_sum"],
        "su_truck_aadt": info["su_truck_sum"],
        "cu_truck_aadt": info["cu_truck_sum"],
        "num_aadt_segments": len(info["segments"]),
        "route": best_seg["route"],
        "description": best_seg["description"],
        "county": best_seg["county"],
        "functional_class": best_seg["functional_class"],
        "stat_type": best_seg["stat_type"],
        "station_type": best_seg["station_type"],
        "max_snap_dist_m": max(s["snap_dist_m"] for s in info["segments"]),
    })

aadt_at_nodes = pd.DataFrame(aadt_rows).set_index("osm_node_id")
print(f"  AADT at nodes range: {aadt_at_nodes['aadt'].min():,} - {aadt_at_nodes['aadt'].max():,}")

# ============================================================
# 3. Load and snap image chips to OSM nodes
# ============================================================
print("\nScanning image chips...")
chip_data = []
pattern = re.compile(r"chip_(\d+)_([-\d.]+)_([-\d.]+)\.tif")

for cat in CATEGORIES:
    cat_dir = os.path.join(CHIP_DIR, cat)
    if not os.path.isdir(cat_dir):
        print(f"  Warning: {cat_dir} not found")
        continue
    for fname in os.listdir(cat_dir):
        m = pattern.match(fname)
        if m:
            chip_data.append({
                "chip_id": m.group(1),
                "chip_lat": float(m.group(2)),
                "chip_lon": float(m.group(3)),
                "category": cat,
                "filename": fname,
            })

chips = pd.DataFrame(chip_data)
print(f"  Total chips: {len(chips):,}")

# Deduplicate by location
chips_unique = chips.drop_duplicates(subset=["chip_lat", "chip_lon"]).copy().reset_index(drop=True)
print(f"  Unique chip locations: {len(chips_unique):,}")

chip_xy = to_xy(chips_unique["chip_lat"].values, chips_unique["chip_lon"].values)
dist_chip, idx_chip = tree_osm.query(chip_xy)

chips_unique["osm_node_id"] = osm_nodes.iloc[idx_chip]["id"].values.astype(int)
chips_unique["chip_snap_dist_m"] = dist_chip

chips_snapped = chips_unique[chips_unique["chip_snap_dist_m"] <= CHIP_SNAP_THRESHOLD_M].copy()
print(f"  Chips snapped to OSM node (within {CHIP_SNAP_THRESHOLD_M}m): {len(chips_snapped):,}")

# ============================================================
# 4. Exact join on OSM node ID
# ============================================================
print("\nJoining on OSM node ID...")

matched = chips_snapped.merge(
    aadt_at_nodes, left_on="osm_node_id", right_index=True, how="inner"
)
print(f"  Chips matched to AADT via OSM node: {len(matched):,}")

# ============================================================
# 5. Snap pedestrian segment endpoints to OSM nodes
# ============================================================
print("\nLoading pedestrian endpoint data...")
ped_df = pd.read_csv(PED_ENDPOINTS_CSV)
ped_df = ped_df.dropna(subset=["start_lat", "start_lon", "end_lat", "end_lon"])
print(f"  Ped segments: {len(ped_df):,}")

ped_start_xy = to_xy(ped_df["start_lat"].values, ped_df["start_lon"].values)
ped_end_xy = to_xy(ped_df["end_lat"].values, ped_df["end_lon"].values)

dist_ped_start, idx_ped_start = tree_osm.query(ped_start_xy)
dist_ped_end, idx_ped_end = tree_osm.query(ped_end_xy)

# Assign ped volume to BOTH endpoint nodes (pedestrian segments are bidirectional)
node_ped = {}  # osm_node_id -> { avg_daily_sum, avg_annual_sum, num_segments }

def add_ped_to_node(osm_idx, dist_m, row):
    if dist_m > PED_SNAP_THRESHOLD_M:
        return False
    nid = int(osm_nodes.iloc[osm_idx]["id"])
    if nid not in node_ped:
        node_ped[nid] = {"avg_daily_sum": 0, "avg_annual_sum": 0, "num_segments": 0}
    node_ped[nid]["avg_daily_sum"] += row["avg_daily"]
    node_ped[nid]["avg_annual_sum"] += row["avg_annual"]
    node_ped[nid]["num_segments"] += 1
    return True

ped_assigned = 0
for i, row in ped_df.iterrows():
    a = add_ped_to_node(idx_ped_start[i], dist_ped_start[i], row)
    b = add_ped_to_node(idx_ped_end[i], dist_ped_end[i], row)
    if a or b:
        ped_assigned += 1

print(f"  Ped segments assigned to nodes: {ped_assigned:,}")
print(f"  Unique OSM nodes with ped data: {len(node_ped):,}")

# Build ped lookup
ped_at_nodes = pd.DataFrame([
    {
        "osm_node_id": nid,
        "ped_avg_daily": round(info["avg_daily_sum"], 1),
        "ped_avg_annual": int(info["avg_annual_sum"]),
        "num_ped_segments": info["num_segments"],
    }
    for nid, info in node_ped.items()
]).set_index("osm_node_id")

# Join ped to matched (chips + AADT)
matched = matched.merge(ped_at_nodes, left_on="osm_node_id", right_index=True, how="inner")
print(f"  With pedestrian data via OSM node: {len(matched):,}")

# ============================================================
# 6. Add all categories per location
# ============================================================
loc_cats = chips.groupby(["chip_lat", "chip_lon"])["category"].apply(
    lambda x: ";".join(sorted(set(x)))
).reset_index()
loc_cats.columns = ["chip_lat", "chip_lon", "all_categories"]
matched = matched.merge(loc_cats, on=["chip_lat", "chip_lon"], how="left")

# ============================================================
# 7. Output
# ============================================================
out_cols = [
    "osm_node_id", "chip_lat", "chip_lon",
    "chip_id", "category", "all_categories", "filename",
    "aadt", "num_aadt_segments", "max_snap_dist_m",
    "route", "description", "county", "functional_class",
    "su_truck_aadt", "cu_truck_aadt", "stat_type", "station_type",
    "ped_avg_daily", "ped_avg_annual", "num_ped_segments",
    "chip_snap_dist_m",
]

out = matched[out_cols].copy()
out["max_snap_dist_m"] = out["max_snap_dist_m"].round(1)
out["chip_snap_dist_m"] = out["chip_snap_dist_m"].round(1)
out = out.sort_values("aadt", ascending=False).reset_index(drop=True)

out.to_csv(OUTPUT_CSV, index=False)

print(f"\n{'='*60}")
print(f"INTERSECTIONS WITH IMAGE + VEHICLE + PED DATA: {len(out):,}")
print(f"{'='*60}")
print(f"\nBy crash category:")
for cat in CATEGORIES:
    n = out["all_categories"].str.contains(cat).sum()
    print(f"  {cat}: {n:,}")
print(f"\nAADT range: {out['aadt'].min():,} - {out['aadt'].max():,}")
print(f"Ped avg daily range: {out['ped_avg_daily'].min():.0f} - {out['ped_avg_daily'].max():.0f}")
print(f"\nBy county:")
print(out["county"].value_counts().to_string())
print(f"\nSnap distance stats (AADT endpoint -> OSM node):")
print(out["max_snap_dist_m"].describe().to_string())
print(f"\nTop 10 by AADT:")
for _, r in out.head(10).iterrows():
    print(f"  AADT={int(r['aadt']):>7}  Ped={int(r['ped_avg_daily']):>5}/day  "
          f"Node={r['osm_node_id']}  {r['route']}")

print(f"\nSaved to {OUTPUT_CSV}")
