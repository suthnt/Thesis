# This code was written with the assistance of Claude (Anthropic).

import json
import pandas as pd
import numpy as np
from osgeo import ogr
from pyproj import Transformer

COUNTS_CSV = "Automated_Traffic_Volume_Counts.csv"
LION_GDB = "nyclion/lion/lion.gdb"
OUTPUT_PATH = "nyc_traffic_intersection_counts.csv"

# ============================================================
# 1. Read and aggregate traffic counts
# ============================================================
print("Reading traffic counts...")
df = pd.read_csv(COUNTS_CSV)
df["Vol"] = pd.to_numeric(df["Vol"], errors="coerce").fillna(0).astype(int)

# Each row is a 15-min count. Aggregate to daily totals per segment+direction+date.
df["date"] = pd.to_datetime(df[["Yr", "M", "D"]].rename(columns={"Yr": "year", "M": "month", "D": "day"}))
daily = df.groupby(["SegmentID", "Direction", "date"])["Vol"].sum().reset_index()
daily.rename(columns={"Vol": "daily_vol"}, inplace=True)

# Average daily volume per segment+direction (across all observed days)
avg_daily = daily.groupby(["SegmentID", "Direction"])["daily_vol"].mean().reset_index()
avg_daily.rename(columns={"daily_vol": "avg_daily_vol"}, inplace=True)

# Also compute per-segment total (sum of all directions)
seg_avg = avg_daily.groupby("SegmentID")["avg_daily_vol"].sum().reset_index()
seg_avg.rename(columns={"avg_daily_vol": "avg_daily_total"}, inplace=True)
seg_avg["avg_annual_total"] = (seg_avg["avg_daily_total"] * 365).astype(int)

print(f"  Unique segments with counts: {seg_avg['SegmentID'].nunique()}")
print(f"  Avg daily total range: {seg_avg['avg_daily_total'].min():.0f} - {seg_avg['avg_daily_total'].max():.0f}")

# ============================================================
# 2. Read LION segment layer -> SegmentID, NodeIDFrom, NodeIDTo
# ============================================================
print("Reading LION segments...")
ds = ogr.Open(LION_GDB)
lion_layer = ds.GetLayerByName("lion")

seg_to_nodes = {}
for feat in lion_layer:
    sid = feat.GetField("SegmentID")
    nf = feat.GetField("NodeIDFrom")
    nt = feat.GetField("NodeIDTo")
    if sid is not None:
        # SegmentID in LION is zero-padded string, CSV has int
        try:
            sid_int = int(sid)
        except (ValueError, TypeError):
            continue
        seg_to_nodes[sid_int] = (nf, nt)

print(f"  LION segments loaded: {len(seg_to_nodes):,}")

# ============================================================
# 3. Read LION node layer -> NodeID, x, y
# ============================================================
print("Reading LION nodes...")
node_layer = ds.GetLayerByName("node")

node_coords = {}
for feat in node_layer:
    nid = feat.GetField("NODEID")
    geom = feat.GetGeometryRef()
    if geom is not None and nid is not None:
        node_coords[str(nid).zfill(7)] = (geom.GetX(), geom.GetY())

print(f"  LION nodes loaded: {len(node_coords):,}")

ds = None  # close GDB

# ============================================================
# 4. Map segment counts to intersection nodes
# ============================================================
print("Mapping segments to intersections...")

# For each segment with traffic data, split its volume equally to from/to nodes
node_volumes = {}  # node_id -> list of volumes attributed to it
matched = 0
unmatched_segs = []

for _, row in seg_avg.iterrows():
    sid = int(row["SegmentID"])
    vol = row["avg_daily_total"]

    if sid in seg_to_nodes:
        nf, nt = seg_to_nodes[sid]
        matched += 1
        # Attribute full segment volume to both intersections
        # (each intersection sees the traffic passing through)
        for node_id in [nf, nt]:
            if node_id and node_id.strip():
                node_volumes.setdefault(node_id, []).append(vol)
    else:
        unmatched_segs.append(sid)

print(f"  Matched segments: {matched}/{len(seg_avg)}")
if unmatched_segs:
    print(f"  Unmatched segment IDs: {unmatched_segs[:10]}...")

# ============================================================
# 5. Aggregate and reproject
# ============================================================
print("Aggregating intersection volumes...")

# CRS: LION uses NAD83 / New York Long Island (ftUS) - EPSG:2263
transformer = Transformer.from_crs("EPSG:2263", "EPSG:4326", always_xy=True)

int_rows = []
for node_id, vols in node_volumes.items():
    total_vol = sum(vols)
    num_segments = len(vols)

    if node_id in node_coords:
        x, y = node_coords[node_id]
        lon, lat = transformer.transform(x, y)
        int_rows.append({
            "node_id": node_id,
            "latitude": round(lat, 6),
            "longitude": round(lon, 6),
            "num_segments": num_segments,
            "avg_daily_traffic": round(total_vol, 0),
            "avg_annual_traffic": int(total_vol * 365),
        })
    else:
        print(f"  Warning: node {node_id} not found in node layer")

result = pd.DataFrame(int_rows)
result = result.sort_values("avg_annual_traffic", ascending=False).reset_index(drop=True)

print(f"\nTotal intersections with traffic data: {len(result)}")
print(f"\nTop 10 intersections:")
print(result.head(10).to_string())
print(f"\nAvg daily traffic range: {result['avg_daily_traffic'].min():.0f} - {result['avg_daily_traffic'].max():.0f}")
print(f"Avg annual traffic range: {result['avg_annual_traffic'].min():,} - {result['avg_annual_traffic'].max():,}")

result.to_csv(OUTPUT_PATH, index=False)
print(f"\nSaved to {OUTPUT_PATH}")
