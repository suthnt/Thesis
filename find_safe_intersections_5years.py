#!/usr/bin/env python
# coding: utf-8
# This code was written with the assistance of Claude (Anthropic).

import os
import sys
import time
import json
import osmnx as ox
import geopandas as gpd
import pandas as pd
import numpy as np
from shapely.geometry import Point
from pathlib import Path
from datetime import datetime, timedelta
from osmnx._errors import InsufficientResponseError

# === CONFIGURATION ===
INPUT_CSV = '/scratch/gpfs/ALAINK/Suthi/CleanedCrashes_5Years_filtered.csv'
OUTPUT_DIR = '/scratch/gpfs/ALAINK/Suthi/safe_intersections_5years'
CHECKPOINT_FILE = os.path.join(OUTPUT_DIR, 'checkpoint.json')
RESULTS_FILE = os.path.join(OUTPUT_DIR, 'safe_intersections_partial.geojson')
FINAL_OUTPUT = '/scratch/gpfs/ALAINK/Suthi/SafeIntersections_5Years.geojson'

# Processing parameters
SEARCH_RADIUS_FT = 1200  # feet - search radius around each crash point
BATCH_SIZE = 100         # Save checkpoint every N points
MIN_DELAY = 1.0          # Minimum seconds between API calls
MAX_DELAY = 5.0          # Max delay on rate limit
START_INDEX = 0          # Override to start from specific index (0 = use checkpoint)

# OSMnx settings
ox.settings.use_cache = True
ox.settings.cache_folder = os.path.join(OUTPUT_DIR, 'osmnx_cache')
ox.settings.timeout = 300
ox.settings.overpass_rate_limit = True


def setup_directories():
    """Create output directories if needed."""
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(ox.settings.cache_folder, exist_ok=True)
    print(f"Output directory: {OUTPUT_DIR}")
    print(f"Cache directory: {ox.settings.cache_folder}")


def load_checkpoint():
    """Load checkpoint to resume from last position."""
    if os.path.exists(CHECKPOINT_FILE):
        with open(CHECKPOINT_FILE, 'r') as f:
            checkpoint = json.load(f)
        print(f"Resuming from checkpoint: index {checkpoint['last_index'] + 1}")
        return checkpoint
    return {'last_index': -1, 'processed': 0, 'errors': 0, 'skipped': 0}


def save_checkpoint(checkpoint, results_gdf=None):
    """Save checkpoint and partial results."""
    with open(CHECKPOINT_FILE, 'w') as f:
        json.dump(checkpoint, f, indent=2)
    
    if results_gdf is not None and len(results_gdf) > 0:
        results_gdf.to_file(RESULTS_FILE, driver='GeoJSON')
        print(f"  Checkpoint saved: index {checkpoint['last_index']}, {len(results_gdf)} intersections found")


def find_intersections(lon, lat, radius_ft=800, network_type="drive"):
    """
    Find road intersections within radius of a point.
    
    Args:
        lon, lat: Coordinates in WGS84
        radius_ft: Search radius in feet
        network_type: OSMnx network type
    
    Returns:
        GeoDataFrame of intersection points with street_count >= 3
    """
    radius_m = float(radius_ft) * 0.3048

    def grab(lat, lon, dist_m, net):
        try:
            G = ox.graph_from_point((lat, lon), dist=dist_m, network_type=net, simplify=True)
            if G.number_of_edges() == 0:
                raise InsufficientResponseError("empty graph")
            return G
        except InsufficientResponseError:
            # try bigger + 'all'
            G = ox.graph_from_point((lat, lon), dist=dist_m*2.5, network_type="all", simplify=True)
            if G.number_of_edges() == 0:
                raise InsufficientResponseError("still empty after fallback")
            return G

    G = grab(float(lat), float(lon), radius_m, network_type)

    # Project before consolidating (meters!)
    Gp = ox.projection.project_graph(G)
    Gc = ox.consolidate_intersections(Gp, tolerance=10, rebuild_graph=True, dead_ends=False)

    if Gc.number_of_edges() == 0:
        # if consolidation nuked everything, skip consolidation
        Gc = Gp

    # Nodes & street counts on the same graph
    nodes_proj, _ = ox.graph_to_gdfs(Gc)
    sc = ox.stats.count_streets_per_node(Gc)
    nodes_proj["street_count"] = nodes_proj.index.map(sc)

    # Build projected point, filter by radius (meters)
    pt_proj = gpd.GeoSeries([Point(lon, lat)], crs=4326).to_crs(nodes_proj.crs).iloc[0]
    hits = nodes_proj[(nodes_proj["street_count"] >= 3) &
                      (nodes_proj.geometry.distance(pt_proj) <= radius_m)]

    # Return lon/lat
    return hits.to_crs(4326)


def process_crashes(start_from=0):
    """
    Main processing loop with checkpointing and rate limiting.
    """
    setup_directories()
    
    # Load data
    print(f"Loading crash data from: {INPUT_CSV}")
    pts = pd.read_csv(INPUT_CSV)
    total_points = len(pts)
    print(f"Total crash points: {total_points}")
    
    # Load checkpoint or start fresh
    if start_from > 0:
        checkpoint = {'last_index': start_from - 1, 'processed': 0, 'errors': 0, 'skipped': 0}
        print(f"Starting from index {start_from} (manual override)")
    else:
        checkpoint = load_checkpoint()
    
    start_idx = checkpoint['last_index'] + 1
    
    # Load existing results if resuming
    results = []
    if os.path.exists(RESULTS_FILE) and start_idx > 0:
        try:
            existing = gpd.read_file(RESULTS_FILE)
            results = [existing]
            print(f"Loaded {len(existing)} existing intersections from partial results")
        except Exception as e:
            print(f"Warning: Could not load existing results: {e}")
    
    # Track timing for rate limiting
    last_request_time = 0
    consecutive_errors = 0
    
    # Progress tracking
    start_time = time.time()
    points_this_session = 0
    
    print(f"\n{'='*60}")
    print(f"Starting processing at index {start_idx} of {total_points}")
    print(f"{'='*60}\n")
    
    for i in range(start_idx, total_points):
        row = pts.iloc[i]
        lon, lat = row['LONGITUDE'], row['LATITUDE']
        
        # Rate limiting
        elapsed = time.time() - last_request_time
        if elapsed < MIN_DELAY:
            time.sleep(MIN_DELAY - elapsed)
        
        # Extra delay if we've had errors
        if consecutive_errors > 0:
            delay = min(MIN_DELAY * (2 ** consecutive_errors), MAX_DELAY)
            time.sleep(delay)
        
        try:
            last_request_time = time.time()
            gdf = find_intersections(lon, lat, radius_ft=SEARCH_RADIUS_FT, network_type="drive")
            
            if len(gdf) > 0:
                gdf["crash_point_id"] = i
                gdf["crash_lon"] = lon
                gdf["crash_lat"] = lat
                gdf["crash_count"] = row.get('count', 1)
                results.append(gdf)
            
            checkpoint['processed'] += 1
            consecutive_errors = 0
            points_this_session += 1
            
            # Progress output
            if points_this_session % 10 == 0:
                elapsed_session = time.time() - start_time
                rate = points_this_session / elapsed_session * 3600  # per hour
                remaining = total_points - i - 1
                eta_hours = remaining / rate if rate > 0 else 0
                
                total_ints = sum(len(r) for r in results)
                print(f"[{i+1}/{total_points}] {points_this_session} processed this session, "
                      f"{total_ints} intersections found, "
                      f"Rate: {rate:.0f}/hr, ETA: {eta_hours:.1f}h")
        
        except InsufficientResponseError:
            checkpoint['skipped'] += 1
            consecutive_errors = 0  # This is expected, not a real error
            if points_this_session % 50 == 0:
                print(f"[{i+1}] Skipped - no road data at ({lon:.4f}, {lat:.4f})")
        
        except Exception as e:
            checkpoint['errors'] += 1
            consecutive_errors += 1
            print(f"[{i+1}] Error at ({lon:.4f}, {lat:.4f}): {e}")
            
            if consecutive_errors >= 5:
                print("Too many consecutive errors! Saving checkpoint and exiting.")
                checkpoint['last_index'] = i
                if results:
                    combined = gpd.GeoDataFrame(pd.concat(results, ignore_index=True), crs="EPSG:4326")
                    save_checkpoint(checkpoint, combined)
                return
        
        checkpoint['last_index'] = i
        
        # Save checkpoint periodically
        if (i + 1) % BATCH_SIZE == 0:
            if results:
                combined = gpd.GeoDataFrame(pd.concat(results, ignore_index=True), crs="EPSG:4326")
                save_checkpoint(checkpoint, combined)
    
    # Final save
    print(f"\n{'='*60}")
    print("PROCESSING COMPLETE")
    print(f"{'='*60}")
    
    if results:
        final_gdf = gpd.GeoDataFrame(pd.concat(results, ignore_index=True), crs="EPSG:4326")
        
        # Remove duplicates (same intersection found from multiple crash points)
        # Round coordinates to identify duplicates
        final_gdf['lon_round'] = final_gdf.geometry.x.round(5)
        final_gdf['lat_round'] = final_gdf.geometry.y.round(5)
        
        print(f"Total intersections before dedup: {len(final_gdf)}")
        final_gdf_dedup = final_gdf.drop_duplicates(subset=['lon_round', 'lat_round'])
        final_gdf_dedup = final_gdf_dedup.drop(columns=['lon_round', 'lat_round'])
        print(f"Unique intersections after dedup: {len(final_gdf_dedup)}")
        
        # Save final output
        final_gdf_dedup.to_file(FINAL_OUTPUT, driver='GeoJSON')
        print(f"Saved to: {FINAL_OUTPUT}")
        
        # Also save CSV version
        csv_output = FINAL_OUTPUT.replace('.geojson', '.csv')
        final_gdf_dedup['longitude'] = final_gdf_dedup.geometry.x
        final_gdf_dedup['latitude'] = final_gdf_dedup.geometry.y
        final_gdf_dedup.drop(columns=['geometry']).to_csv(csv_output, index=False)
        print(f"CSV saved to: {csv_output}")
    
    print(f"\nSummary:")
    print(f"  Processed: {checkpoint['processed']}")
    print(f"  Skipped (no road data): {checkpoint['skipped']}")
    print(f"  Errors: {checkpoint['errors']}")
    
    # Clean up checkpoint on successful completion
    if os.path.exists(CHECKPOINT_FILE):
        os.rename(CHECKPOINT_FILE, CHECKPOINT_FILE + '.completed')
        print(f"\nCheckpoint marked as complete.")


if __name__ == "__main__":
    # Allow starting from a specific index via command line
    start = START_INDEX
    if len(sys.argv) > 1:
        try:
            start = int(sys.argv[1])
            print(f"Starting from index {start} (command line argument)")
        except ValueError:
            print(f"Invalid start index: {sys.argv[1]}")
            sys.exit(1)
    
    process_crashes(start_from=start)
