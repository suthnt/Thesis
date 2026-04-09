# This code was written with the assistance of Claude (Anthropic).

from osgeo import ogr
import csv

GDB_PATH = "NYSDOT_data/AADT_2023.gdb"
OUTPUT_PATH = "nyc_aadt_2023.csv"

print("Reading AADT 2023 GDB...")
ds = ogr.Open(GDB_PATH)
layer = ds.GetLayer(0)

# Field name shortcuts (the GDB has very long prefixed names)
FIELDS = {
    "rc_station": "Traffic_Station_Locations_RC_STATION",
    "dot_id": "Traffic_Station_Locations_DOT_ID",
    "region": "Traffic_Station_Locations_RG",
    "speed_limit": "Traffic_Station_Locations_Speed",
    "oneway": "Traffic_Station_Locations_Oneway",
    "lanes": "Traffic_Station_Locations_Lanes",
    "description": "AADT_Stats_2023_Table_Description",
    "county": "AADT_Stats_2023_Table_County",
    "route": "AADT_Stats_2023_Table_Route",
    "functional_class": "AADT_Stats_2023_Table_Functional_Class",
    "latitude": "AADT_Stats_2023_Table_Latitude",
    "longitude": "AADT_Stats_2023_Table_Longitude",
    "year": "AADT_Stats_2023_Table_Year",
    "stat_type": "AADT_Stats_2023_Table_Stat_Type",
    "aadt": "AADT_Stats_2023_Table_AADT",
    "aawdt": "AADT_Stats_2023_Table_AAWDT",
    "su_truck_aadt": "AADT_Stats_2023_Table_Single_Unit_Truck_AADT",
    "cu_truck_aadt": "AADT_Stats_2023_Table_Combo_Unit_Truck_AADT",
    "k_factor": "AADT_Stats_2023_Table_K_Factor",
    "d_factor": "AADT_Stats_2023_Table_D_Factor",
    "station_type": "AADT_Stats_2023_Table_Station_Type",
    "shape_length": "Shape_Length",
}

OUT_COLS = [
    "latitude", "longitude", "aadt", "aawdt",
    "su_truck_aadt", "cu_truck_aadt",
    "county", "route", "description",
    "functional_class", "stat_type", "station_type",
    "speed_limit", "oneway", "lanes",
    "k_factor", "d_factor",
    "rc_station", "dot_id", "shape_length",
]

rows = []
skipped = 0
for feat in layer:
    rg = feat.GetField(FIELDS["region"])
    if rg != 11:
        continue

    row = {}
    for out_name in OUT_COLS:
        val = feat.GetField(FIELDS[out_name])
        row[out_name] = val if val is not None else ""
    
    # Skip rows without lat/lon or AADT
    if not row["latitude"] or not row["longitude"] or row["aadt"] == "":
        skipped += 1
        continue

    rows.append(row)

ds = None

# Sort by AADT descending
rows.sort(key=lambda r: int(r["aadt"]) if r["aadt"] else 0, reverse=True)

# Write CSV
with open(OUTPUT_PATH, "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=OUT_COLS)
    writer.writeheader()
    writer.writerows(rows)

print(f"Region 11 segments extracted: {len(rows):,}")
print(f"Skipped (missing lat/lon/AADT): {skipped}")
print()

# Summary stats
aadts = [int(r["aadt"]) for r in rows if r["aadt"]]
print(f"AADT range: {min(aadts):,} - {max(aadts):,}")
print(f"AADT median: {sorted(aadts)[len(aadts)//2]:,}")
print()

# County breakdown
counties = {}
for r in rows:
    c = r["county"] or "Unknown"
    counties[c] = counties.get(c, 0) + 1
print("By county:")
for k, v in sorted(counties.items(), key=lambda x: -x[1]):
    print(f"  {k}: {v:,}")

print()
# Show top 10
print("Top 10 by AADT:")
for r in rows[:10]:
    print(f"  AADT={r['aadt']:>7}  {r['route']:<30} {r['description'][:50]}")

print(f"\nSaved to {OUTPUT_PATH}")
