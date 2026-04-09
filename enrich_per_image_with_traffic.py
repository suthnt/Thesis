#!/usr/bin/env python3
# This code was written with the assistance of Claude (Anthropic).

import os
import re
import pandas as pd

BASE = "/scratch/gpfs/ALAINK/Suthi"
INTERSECTIONS = os.path.join(BASE, "intersections_with_all_three.csv")

INPUT_CSVS = [
    os.path.join(BASE, "yolo_feature_detector", "feature_prevalence_full_m_per_image.csv"),
    os.path.join(BASE, "yolo_feature_detector", "ground_truth_prevalence_per_image_with_actual_counts.csv"),
    os.path.join(BASE, "yolo_feature_detector", "feature_area_coverage_exp2_m_per_image.csv"),
]

def extract_chip_id(path):
    m = re.search(r'chip_(\d+)', os.path.basename(path))
    return int(m.group(1)) if m else None

def main():
    inter = pd.read_csv(INTERSECTIONS)
    traffic = inter.groupby("chip_id").agg({
        "aadt": "first",
        "ped_avg_daily": "first",
    }).reset_index()
    traffic = traffic.rename(columns={"chip_id": "chip_id_int", "aadt": "AADT", "ped_avg_daily": "Ped_avg_daily"})
    print(f"Traffic lookup: {len(traffic)} unique chips")

    for csv_path in INPUT_CSVS:
        if not os.path.isfile(csv_path):
            print(f"  SKIP (not found): {csv_path}")
            continue
        df = pd.read_csv(csv_path)
        df["chip_id_int"] = df["path"].apply(extract_chip_id)
        n0 = len(df)
        merged = df.merge(traffic, on="chip_id_int", how="inner")
        merged = merged.drop(columns=["chip_id_int"])

        base, ext = os.path.splitext(csv_path)
        out_path = base + "_with_traffic" + ext
        merged.to_csv(out_path, index=False)
        print(f"  {os.path.basename(csv_path)}: {n0} -> {len(merged)} rows (matched) -> {out_path}")

if __name__ == "__main__":
    main()
