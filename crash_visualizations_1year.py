#!/usr/bin/env python3
# This code was written with the assistance of Claude (Anthropic).

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

RAW_CSV = "/scratch/gpfs/ALAINK/Suthi/NYC_MVC_2024_BikePed.csv"
OUTPUT_DIR = Path("/scratch/gpfs/ALAINK/Suthi/crash_visualizations_1year")
OUTPUT_DIR.mkdir(exist_ok=True)

# Style
plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["axes.spines.top"] = False
plt.rcParams["axes.spines.right"] = False


def load_raw_data():
    """Load raw 1-year crash data."""
    df = pd.read_csv(RAW_CSV)
    # Parse date and time
    df["CRASH DATE"] = pd.to_datetime(df["CRASH DATE"], format="%m/%d/%Y", errors="coerce")
    df = df.dropna(subset=["CRASH DATE"])
    # Parse time ("12:25", "5:15", "0:00" -> hour as float)
    def parse_time(s):
        if pd.isna(s):
            return np.nan
        s = str(s).strip()
        parts = s.split(":")
        if len(parts) >= 2:
            try:
                return int(parts[0]) + int(parts[1]) / 60.0
            except ValueError:
                return np.nan
        return np.nan

    df["hour"] = df["CRASH TIME"].apply(parse_time)
    df["day_of_year"] = df["CRASH DATE"].dt.dayofyear
    return df


def plot_time_of_day(df):
    """Crashes by hour of day (0-24)."""
    valid = df["hour"].dropna()
    if valid.empty:
        print("No valid time data for time-of-day plot")
        return
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.hist(valid, bins=24, range=(0, 24), color="#e74c3c", alpha=0.85, edgecolor="white", linewidth=0.5)
    ax.set_xlabel("Hour of day")
    ax.set_ylabel("Number of crashes")
    ax.set_title("NYC Bike/Ped Crashes by Time of Day (2024, raw data)")
    ax.set_xticks(range(0, 25, 2))
    ax.set_xlim(0, 24)
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "crashes_by_hour.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {OUTPUT_DIR / 'crashes_by_hour.png'}")


def plot_day_of_year(df):
    """Crashes by day of year (Jan 1 - Dec 31)."""
    fig, ax = plt.subplots(figsize=(12, 5))
    counts = df.groupby("day_of_year").size()
    ax.bar(counts.index, counts.values, color="#3498db", alpha=0.8, width=0.9, edgecolor="none")
    ax.set_xlabel("Day of year (Jan 1 = 1, Dec 31 = 366)")
    ax.set_ylabel("Number of crashes")
    ax.set_title("NYC Bike/Ped Crashes by Day of Year (2024, raw data)")
    # Add month labels
    month_pos = [1, 32, 60, 91, 121, 152, 182, 213, 244, 274, 305, 335]
    month_names = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
    ax.set_xticks(month_pos)
    ax.set_xticklabels(month_names)
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "crashes_by_day_of_year.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {OUTPUT_DIR / 'crashes_by_day_of_year.png'}")


def plot_borough_heatmap(df):
    """Bar chart by borough with heat-style coloring."""
    borough_counts = df["BOROUGH"].value_counts()
    borough_counts = borough_counts[borough_counts.index != ""]  # drop empty
    if borough_counts.empty:
        print("No borough data for borough plot")
        return
    fig, ax = plt.subplots(figsize=(8, 5))
    colors = plt.cm.YlOrRd(np.linspace(0.3, 0.95, len(borough_counts)))
    bars = ax.barh(borough_counts.index, borough_counts.values, color=colors, edgecolor="white", linewidth=0.5)
    ax.set_xlabel("Number of crashes")
    ax.set_ylabel("Borough")
    ax.set_title("NYC Bike/Ped Crashes by Borough (2024, raw data)")
    ax.invert_yaxis()  # so Manhattan/Bronx at top
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "crashes_by_borough.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {OUTPUT_DIR / 'crashes_by_borough.png'}")


def plot_map(df):
    """Scatter plot of crash locations on NYC map."""
    # Need valid lat/lon
    df_map = df.dropna(subset=["LATITUDE", "LONGITUDE"]).copy()
    df_map["LATITUDE"] = pd.to_numeric(df_map["LATITUDE"], errors="coerce")
    df_map["LONGITUDE"] = pd.to_numeric(df_map["LONGITUDE"], errors="coerce")
    df_map = df_map.dropna(subset=["LATITUDE", "LONGITUDE"])
    # NYC bounds roughly
    df_map = df_map[
        (df_map["LATITUDE"] >= 40.5) & (df_map["LATITUDE"] <= 40.95)
        & (df_map["LONGITUDE"] >= -74.35) & (df_map["LONGITUDE"] <= -73.7)
    ]
    if df_map.empty:
        print("No valid lat/lon for map")
        return

    try:
        import folium
        # Create folium map
        m = folium.Map(
            location=[40.7128, -74.0060],
            zoom_start=10,
            tiles="CartoDB positron",
        )
        # Sample if too many points for performance (folium handles ~10k ok)
        to_plot = df_map if len(df_map) <= 15000 else df_map.sample(10000, random_state=42)
        for _, row in to_plot.iterrows():
            folium.CircleMarker(
                location=[row["LATITUDE"], row["LONGITUDE"]],
                radius=2,
                color="#e74c3c",
                fill=True,
                fill_color="#e74c3c",
                fill_opacity=0.6,
                weight=0,
            ).add_to(m)
        m.save(str(OUTPUT_DIR / "crashes_map.html"))
        print(f"Saved: {OUTPUT_DIR / 'crashes_map.html'} (open in browser)")
    except ImportError:
        # Fallback: matplotlib scatter (no basemap)
        fig, ax = plt.subplots(figsize=(10, 12))
        ax.scatter(
            df_map["LONGITUDE"], df_map["LATITUDE"],
            s=2, alpha=0.4, c="#e74c3c", rasterized=True
        )
        ax.set_xlabel("Longitude")
        ax.set_ylabel("Latitude")
        ax.set_title("NYC Bike/Ped Crash Locations (2024, raw data)")
        ax.set_aspect("equal")
        ax.set_xlim(-74.35, -73.7)
        ax.set_ylim(40.5, 40.95)
        fig.tight_layout()
        fig.savefig(OUTPUT_DIR / "crashes_map.png", dpi=150, bbox_inches="tight")
        plt.close()
        print(f"Saved: {OUTPUT_DIR / 'crashes_map.png'} (install folium for interactive HTML map)")


def main():
    print("Loading raw 1-year crash data...")
    df = load_raw_data()
    print(f"  Loaded {len(df)} crash records")
    print()
    plot_time_of_day(df)
    plot_day_of_year(df)
    plot_borough_heatmap(df)
    plot_map(df)
    print(f"\nAll outputs in: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
