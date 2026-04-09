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
from datetime import datetime
from osmnx._errors import InsufficientResponseError

# === CONFIGURATION ===
INPUT_CSV = '/scratch/gpfs/ALAINK/Suthi/CleanedCrashes_5Years_filtered.csv'
OUTPUT_DIR = '/scratch/gpfs/ALAINK/Suthi/safe_intersections_parallel'

# Processing parameters
SEARCH_RADIUS_FT = 1200  # feet - search radius around each crash point
BATCH_SIZE = 50          # Save checkpoint every N points
MIN_DELAY = 1.0          # Minimum seconds between API calls
MAX_DELAY = 5.0          # Max delay on rate limit

# OSMnx settings
ox.settings.use_cache = True
ox.settings.timeout = 300
ox.settings.overpass_rate_limit = True


def setup_directories(chunk_id):
    """Create output directories if needed."""
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    cache_dir = os.path.join(OUTPUT_DIR, f'cache_chunk_{chunk_id}')
    os.makedirs(cache_dir, exist_ok=True)
    ox.settings.cache_folder = cache_dir
    print(f"Output directory: {OUTPUT_DIR}")
    print(f"Cache directory: {cache_dir}")


def get_chunk_files(chunk_id):
    """Get file paths for this chunk."""
    return {
        'checkpoint': os.path.join(OUTPUT_DIR, f'checkpoint_chunk_{chunk_id}.json'),
        'results': os.path.join(OUTPUT_DIR, f'results_chunk_{chunk_id}.csv'),
        'final': os.path.join(OUTPUT_DIR, f'final_chunk_{chunk_id}.csv'),
    }


def load_checkpoint(checkpoint_file):
    """Load checkpoint to resume from last position."""
    if os.path.exists(checkpoint_file):
        with open(checkpoint_file, 'r') as f:
            checkpoint = json.load(f)
        print(f"Resuming from checkpoint: processed {checkpoint['processed_in_chunk']}")
        return checkpoint
    return {'last_idx_in_chunk': -1, 'processed_in_chunk': 0, 'errors': 0, 'skipped': 0}


def save_checkpoint(checkpoint, checkpoint_file, results_gdf=None, results_file=None):
    """Save checkpoint and partial results."""
    with open(checkpoint_file, 'w') as f:
        json.dump(checkpoint, f, indent=2)
    
    if results_gdf is not None and len(results_gdf) > 0 and results_file:
        # Save as CSV to avoid fiona/pyogrio dependency
        df = results_gdf.copy()
        df['longitude'] = df.geometry.x
        df['latitude'] = df.geometry.y
        df = df.drop(columns=['geometry'])
        df.to_csv(results_file, index=False)
        print(f"  Checkpoint saved: {checkpoint['processed_in_chunk']} processed, {len(df)} intersections")


def find_intersections(lon, lat, radius_ft=800, network_type="drive"):
    """
    Find road intersections within radius of a point.
    """
    radius_m = float(radius_ft) * 0.3048

    def grab(lat, lon, dist_m, net):
        try:
            G = ox.graph_from_point((lat, lon), dist=dist_m, network_type=net, simplify=True)
            if G.number_of_edges() == 0:
                raise InsufficientResponseError("empty graph")
            return G
        except InsufficientResponseError:
            G = ox.graph_from_point((lat, lon), dist=dist_m*2.5, network_type="all", simplify=True)
            if G.number_of_edges() == 0:
                raise InsufficientResponseError("still empty after fallback")
            return G

    G = grab(float(lat), float(lon), radius_m, network_type)

    Gp = ox.projection.project_graph(G)
    Gc = ox.consolidate_intersections(Gp, tolerance=10, rebuild_graph=True, dead_ends=False)

    if Gc.number_of_edges() == 0:
        Gc = Gp

    nodes_proj, _ = ox.graph_to_gdfs(Gc)
    sc = ox.stats.count_streets_per_node(Gc)
    nodes_proj["street_count"] = nodes_proj.index.map(sc)

    pt_proj = gpd.GeoSeries([Point(lon, lat)], crs=4326).to_crs(nodes_proj.crs).iloc[0]
    hits = nodes_proj[(nodes_proj["street_count"] >= 3) &
                      (nodes_proj.geometry.distance(pt_proj) <= radius_m)]

    return hits.to_crs(4326)


def process_chunk(chunk_id, total_chunks):
    """
    Process a specific chunk of the data.
    """
    setup_directories(chunk_id)
    files = get_chunk_files(chunk_id)
    
    # Load data
    print(f"Loading crash data from: {INPUT_CSV}")
    pts = pd.read_csv(INPUT_CSV)
    total_points = len(pts)
    
    # Calculate chunk boundaries
    chunk_size = total_points // total_chunks
    start_idx = chunk_id * chunk_size
    end_idx = start_idx + chunk_size if chunk_id < total_chunks - 1 else total_points
    
    chunk_data = pts.iloc[start_idx:end_idx].reset_index(drop=True)
    chunk_len = len(chunk_data)
    
    print(f"\n{'='*60}")
    print(f"CHUNK {chunk_id} of {total_chunks}")
    print(f"Processing rows {start_idx} to {end_idx-1} ({chunk_len} points)")
    print(f"{'='*60}\n")
    
    # Load checkpoint
    checkpoint = load_checkpoint(files['checkpoint'])
    start_in_chunk = checkpoint['last_idx_in_chunk'] + 1
    
    # Load existing results if resuming
    results = []
    if os.path.exists(files['results']) and start_in_chunk > 0:
        try:
            existing = pd.read_csv(files['results'])
            # Convert back to GeoDataFrame
            existing = gpd.GeoDataFrame(
                existing,
                geometry=gpd.points_from_xy(existing.longitude, existing.latitude),
                crs="EPSG:4326"
            )
            results = [existing]
            print(f"Loaded {len(existing)} existing intersections")
        except Exception as e:
            print(f"Warning: Could not load existing results: {e}")
    
    # Track timing
    last_request_time = 0
    consecutive_errors = 0
    start_time = time.time()
    points_this_session = 0
    
    for i in range(start_in_chunk, chunk_len):
        row = chunk_data.iloc[i]
        global_idx = start_idx + i
        lon, lat = row['LONGITUDE'], row['LATITUDE']
        
        # Rate limiting
        elapsed = time.time() - last_request_time
        if elapsed < MIN_DELAY:
            time.sleep(MIN_DELAY - elapsed)
        
        if consecutive_errors > 0:
            delay = min(MIN_DELAY * (2 ** consecutive_errors), MAX_DELAY)
            time.sleep(delay)
        
        try:
            last_request_time = time.time()
            gdf = find_intersections(lon, lat, radius_ft=SEARCH_RADIUS_FT, network_type="drive")
            
            if len(gdf) > 0:
                gdf["crash_point_id"] = global_idx
                gdf["crash_lon"] = lon
                gdf["crash_lat"] = lat
                gdf["crash_count"] = row.get('count', 1)
                results.append(gdf)
            
            checkpoint['processed_in_chunk'] += 1
            consecutive_errors = 0
            points_this_session += 1
            
            # Progress output
            if points_this_session % 10 == 0:
                elapsed_session = time.time() - start_time
                rate = points_this_session / elapsed_session * 3600
                remaining = chunk_len - i - 1
                eta_hours = remaining / rate if rate > 0 else 0
                
                total_ints = sum(len(r) for r in results)
                print(f"[Chunk {chunk_id}] {i+1}/{chunk_len} ({points_this_session} this session), "
                      f"{total_ints} ints, {rate:.0f}/hr, ETA: {eta_hours:.1f}h")
        
        except InsufficientResponseError:
            checkpoint['skipped'] += 1
            consecutive_errors = 0
        
        except Exception as e:
            checkpoint['errors'] += 1
            consecutive_errors += 1
            print(f"[Chunk {chunk_id}] Error at idx {i}: {e}")
            
            if consecutive_errors >= 5:
                print("Too many consecutive errors! Saving and exiting.")
                checkpoint['last_idx_in_chunk'] = i
                if results:
                    combined = gpd.GeoDataFrame(pd.concat(results, ignore_index=True), crs="EPSG:4326")
                    save_checkpoint(checkpoint, files['checkpoint'], combined, files['results'])
                sys.exit(1)
        
        checkpoint['last_idx_in_chunk'] = i
        
        # Save checkpoint periodically
        if (i + 1) % BATCH_SIZE == 0:
            if results:
                combined = gpd.GeoDataFrame(pd.concat(results, ignore_index=True), crs="EPSG:4326")
                save_checkpoint(checkpoint, files['checkpoint'], combined, files['results'])
    
    # Final save
    print(f"\n{'='*60}")
    print(f"CHUNK {chunk_id} COMPLETE")
    print(f"{'='*60}")
    
    if results:
        final_gdf = gpd.GeoDataFrame(pd.concat(results, ignore_index=True), crs="EPSG:4326")
        # Save as CSV
        final_df = final_gdf.copy()
        final_df['longitude'] = final_df.geometry.x
        final_df['latitude'] = final_df.geometry.y
        final_df = final_df.drop(columns=['geometry'])
        final_df.to_csv(files['final'], index=False)
        print(f"Saved {len(final_df)} intersections to: {files['final']}")
    
    print(f"\nSummary:")
    print(f"  Processed: {checkpoint['processed_in_chunk']}")
    print(f"  Skipped: {checkpoint['skipped']}")
    print(f"  Errors: {checkpoint['errors']}")
    
    # Mark complete
    if os.path.exists(files['checkpoint']):
        os.rename(files['checkpoint'], files['checkpoint'] + '.completed')


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python find_safe_intersections_parallel.py <chunk_id> <total_chunks>")
        print("Example: python find_safe_intersections_parallel.py 0 10")
        sys.exit(1)
    
    chunk_id = int(sys.argv[1])
    total_chunks = int(sys.argv[2])
    
    if chunk_id < 0 or chunk_id >= total_chunks:
        print(f"Error: chunk_id must be between 0 and {total_chunks-1}")
        sys.exit(1)
    
    process_chunk(chunk_id, total_chunks)
