#!/usr/bin/env python
# coding: utf-8
# This code was written with the assistance of Claude (Anthropic).

import os
import sys
import glob
import geopandas as gpd
import pandas as pd

OUTPUT_DIR = '/scratch/gpfs/ALAINK/Suthi/safe_intersections_parallel'
FINAL_OUTPUT = '/scratch/gpfs/ALAINK/Suthi/SafeIntersections_5Years.csv'


def merge_results(total_chunks=10):
    """Merge all chunk results into a single file."""
    
    print(f"Looking for chunk results in: {OUTPUT_DIR}")
    
    # Find all final chunk files
    chunk_files = []
    missing_chunks = []
    
    for i in range(total_chunks):
        final_file = os.path.join(OUTPUT_DIR, f'final_chunk_{i}.csv')
        results_file = os.path.join(OUTPUT_DIR, f'results_chunk_{i}.csv')
        
        if os.path.exists(final_file):
            chunk_files.append((i, final_file))
        elif os.path.exists(results_file):
            # Use partial results if final doesn't exist
            chunk_files.append((i, results_file))
            print(f"  Warning: Chunk {i} using partial results (not complete)")
        else:
            missing_chunks.append(i)
    
    if missing_chunks:
        print(f"\n⚠️  Missing chunks: {missing_chunks}")
        print("  These chunks may still be running or failed.")
    
    if not chunk_files:
        print("No chunk files found!")
        return
    
    print(f"\nFound {len(chunk_files)} chunk files to merge:")
    for i, f in chunk_files:
        print(f"  Chunk {i}: {f}")
    
    # Load and merge
    all_results = []
    for chunk_id, filepath in chunk_files:
        try:
            df = pd.read_csv(filepath)
            print(f"  Chunk {chunk_id}: {len(df)} intersections")
            all_results.append(df)
        except Exception as e:
            print(f"  Chunk {chunk_id}: Error loading - {e}")
    
    if not all_results:
        print("No data loaded!")
        return
    
    # Combine all results
    combined = pd.concat(all_results, ignore_index=True)
    print(f"\nTotal intersections before dedup: {len(combined)}")
    
    # Remove duplicates (same intersection found from multiple crash points)
    combined['lon_round'] = combined['longitude'].round(5)
    combined['lat_round'] = combined['latitude'].round(5)
    deduped = combined.drop_duplicates(subset=['lon_round', 'lat_round'])
    deduped = deduped.drop(columns=['lon_round', 'lat_round'])
    
    print(f"Unique intersections after dedup: {len(deduped)}")
    
    # Save CSV
    deduped.to_csv(FINAL_OUTPUT, index=False)
    print(f"\n✅ Saved to: {FINAL_OUTPUT}")
    
    # Summary stats
    print(f"\n{'='*50}")
    print("SUMMARY")
    print(f"{'='*50}")
    print(f"Total unique safe intersections: {len(deduped)}")
    if 'street_count' in deduped.columns:
        print(f"Street count distribution:")
        print(deduped['street_count'].value_counts().sort_index())
    print(f"\nColumns: {list(deduped.columns)}")


if __name__ == "__main__":
    total = 10
    if len(sys.argv) > 1:
        total = int(sys.argv[1])
    
    merge_results(total)
