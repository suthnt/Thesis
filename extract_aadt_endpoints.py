# This code was written with the assistance of Claude (Anthropic).

from osgeo import ogr
from pyproj import Transformer
import csv

AADT_GDB = "NYSDOT_data/AADT_2023.gdb"
OUTPUT_CSV = "nyc_aadt_endpoints.csv"

# Transformer: EPSG:26918 (NAD83 UTM 18N) -> WGS84
transformer = Transformer.from_crs("EPSG:26918", "EPSG:4326", always_xy=True)

print("Reading AADT segments from GDB (Region 11 only)...")
ds = ogr.Open(AADT_GDB)
layer = ds.GetLayer(0)

rows = []
for feat in layer:
    if feat.GetField("Traffic_Station_Locations_RG") != 11:
        continue

    aadt_val = feat.GetField("AADT_Stats_2023_Table_AADT")
    if aadt_val is None:
        continue

    geom = feat.GetGeometryRef()
    if geom is None:
        continue

    ls = geom.GetGeometryRef(0)
    if ls is None or ls.GetPointCount() < 2:
        continue

    n_pts = ls.GetPointCount()
    sx, sy = ls.GetX(0), ls.GetY(0)
    ex, ey = ls.GetX(n_pts - 1), ls.GetY(n_pts - 1)
    s_lon, s_lat = transformer.transform(sx, sy)
    e_lon, e_lat = transformer.transform(ex, ey)

    oneway = (feat.GetField("Traffic_Station_Locations_Oneway") or "").strip()

    rows.append({
        "aadt": aadt_val,
        "start_lat": round(s_lat, 6),
        "start_lon": round(s_lon, 6),
        "end_lat": round(e_lat, 6),
        "end_lon": round(e_lon, 6),
        "oneway": oneway,
        "route": feat.GetField("AADT_Stats_2023_Table_Route") or "",
        "description": feat.GetField("AADT_Stats_2023_Table_Description") or "",
        "county": feat.GetField("AADT_Stats_2023_Table_County") or "",
        "functional_class": feat.GetField("AADT_Stats_2023_Table_Functional_Class") or "",
        "su_truck_aadt": feat.GetField("AADT_Stats_2023_Table_Single_Unit_Truck_AADT") or 0,
        "cu_truck_aadt": feat.GetField("AADT_Stats_2023_Table_Combo_Unit_Truck_AADT") or 0,
        "stat_type": feat.GetField("AADT_Stats_2023_Table_Stat_Type") or "",
        "station_type": feat.GetField("AADT_Stats_2023_Table_Station_Type") or "",
    })

ds = None

fieldnames = [
    "aadt", "start_lat", "start_lon", "end_lat", "end_lon", "oneway",
    "route", "description", "county", "functional_class",
    "su_truck_aadt", "cu_truck_aadt", "stat_type", "station_type",
]

with open(OUTPUT_CSV, "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(rows)

print(f"Extracted {len(rows):,} R11 segments to {OUTPUT_CSV}")
oneway_count = sum(1 for r in rows if r["oneway"] == "Y")
print(f"  One-way: {oneway_count:,}")
print(f"  Two-way: {len(rows) - oneway_count:,}")
