# -*- coding: utf-8 -*-
# This code was written with the assistance of Claude (Anthropic).

import os
import glob
import subprocess
import rasterio
from rasterio.windows import Window
from rasterio.crs import CRS
from pyproj import Transformer, CRS as PyprojCRS
import pandas as pd
import numpy as np
from pathlib import Path
import zipfile


def get_crs_units_per_meter(crs):
    """Get the conversion factor from meters to the CRS linear units.
    
    Returns:
        float: Number of CRS units per meter (e.g., 3.28084 for US survey feet)
        str: Name of the unit
    """
    try:
        pyproj_crs = PyprojCRS.from_user_input(crs)
        axis_info = pyproj_crs.axis_info
        
        if axis_info:
            unit_name = axis_info[0].unit_name
            # Get the conversion factor (meters per CRS unit)
            unit_conversion = axis_info[0].unit_conversion_factor
            
            if unit_conversion and unit_conversion != 0:
                # unit_conversion is meters per CRS unit
                # We want CRS units per meter
                units_per_meter = 1.0 / unit_conversion
                return units_per_meter, unit_name
        
        # Default to meters if we can't determine
        return 1.0, "metre"
    except Exception as e:
        print(f"Warning: Could not determine CRS units ({e}), assuming meters")
        return 1.0, "metre"

# === CONFIGURATION ===
TILE_FOLDERS = ["/scratch/gpfs/ALAINK/Suthi/BronxImages", "/scratch/gpfs/ALAINK/Suthi/QueensImages", "/scratch/gpfs/ALAINK/Suthi/BrooklynImages", "/scratch/gpfs/ALAINK/Suthi/ManhattanImages", "/scratch/gpfs/ALAINK/Suthi/StatenIslandImages"]  # your tile directories
CSV_PATH = "/scratch/gpfs/ALAINK/Suthi/CleanedCrashes_5Years.csv"
OUTPUT_DIR = "/scratch/gpfs/ALAINK/Suthi/DangerousIntersectionImages_5Years"
CHIP_SIZE_METERS = 15  # 15m x 15m

# Edge case settings
SKIP_PARTIAL_CHIPS = True      # Skip chips that would be cut off at edges
NODATA_THRESHOLD = 0.1         # Skip chips with >10% nodata pixels
SAVE_METADATA_CSV = True       # Save a CSV with chip metadata for tracking

# === STEP 1: Create a Virtual Raster (VRT) from all tiles ===
def create_vrt(tile_folders, vrt_path="mosaic.vrt", file_extension=".jp2"):
    """Build a VRT that virtually stitches all tiles together.

    Args:
        tile_folders: List of directories containing image tiles
        vrt_path: Output path for the VRT file
        file_extension: Only include files with this extension (e.g., '.jp2', '.tif')
    """
    # Find ONLY the image files we want (e.g., .jp2), not .aux, .tab, .j2w, etc.
    all_images = []
    for folder in tile_folders:
        # Use case-insensitive matching for the extension
        pattern = os.path.join(folder, "**", f"*{file_extension}")
        found = glob.glob(pattern, recursive=True)

        # Extra filter to exclude files like .jp2.aux (which would match *.jp2)
        found = [f for f in found if f.lower().endswith(file_extension.lower())
                 and not f.lower().endswith('.aux')]

        all_images.extend(found)

    print(f"Found {len(all_images)} {file_extension} files")

    if len(all_images) == 0:
        raise ValueError(f"No {file_extension} files found in the specified folders!")

    # Show a sample of what we found
    print(f"Sample files:")
    for f in all_images[:5]:
        print(f"  - {f}")
    if len(all_images) > 5:
        print(f"  ... and {len(all_images) - 5} more")

    # Write file list for gdalbuildvrt
    filelist_path = "tile_list.txt"
    with open(filelist_path, "w") as f:
        f.write("\n".join(all_images))

    # Build VRT using GDAL
    cmd = ["gdalbuildvrt", "-input_file_list", filelist_path, vrt_path]
    subprocess.run(cmd, check=True)
    print(f"Created VRT: {vrt_path}")

    return vrt_path

# === STEP 2: Extract chips with edge case handling ===
def extract_chips(vrt_path, csv_path, output_dir, chip_meters=15,
                  skip_partial=True, nodata_threshold=0.1, save_metadata=True):
    """Extract chips at each lat/lon location with robust edge case handling.

    Args:
        vrt_path: Path to the VRT (or any raster)
        csv_path: CSV with 'longitude' and 'latitude' columns
        output_dir: Where to save the chips
        chip_meters: Size of chips in meters
        skip_partial: If True, skip chips that would extend beyond image edges
        nodata_threshold: Skip chips with more than this fraction of nodata (0-1)
        save_metadata: If True, save a CSV tracking all chip extractions
    """
    os.makedirs(output_dir, exist_ok=True)

    df = pd.read_csv(csv_path)

    # Metadata tracking
    metadata_records = []

    with rasterio.open(vrt_path) as src:
        # Get image dimensions
        img_height, img_width = src.height, src.width

        # Get the resolution in CRS units per pixel
        res_x = abs(src.transform.a)  # pixel width in CRS units
        res_y = abs(src.transform.e)  # pixel height in CRS units

        # Detect CRS units and get conversion factor
        units_per_meter, unit_name = get_crs_units_per_meter(src.crs)
        
        # Convert chip size from meters to CRS units
        chip_size_crs_units = chip_meters * units_per_meter

        # Calculate chip size in pixels
        chip_width_px = int(np.ceil(chip_size_crs_units / res_x))
        chip_height_px = int(np.ceil(chip_size_crs_units / res_y))
        half_w = chip_width_px // 2
        half_h = chip_height_px // 2

        print(f"Image dimensions: {img_width} x {img_height} pixels")
        print(f"Image CRS units: {unit_name} ({units_per_meter:.4f} per meter)")
        print(f"Image resolution: {res_x:.4f} x {res_y:.4f} {unit_name}/pixel")
        print(f"Chip size: {chip_width_px} x {chip_height_px} pixels ({chip_meters}m = {chip_size_crs_units:.2f} {unit_name})")

        # Get nodata value if it exists
        nodata_val = src.nodata
        print(f"NoData value: {nodata_val}")

        # Set up coordinate transformer if needed (WGS84 lat/lon -> image CRS)
        image_crs = src.crs
        if image_crs != CRS.from_epsg(4326):
            transformer = Transformer.from_crs("EPSG:4326", image_crs, always_xy=True)
            print(f"Coordinate transform: WGS84 -> {image_crs}")
        else:
            transformer = None
            print("No coordinate transform needed (image is in WGS84)")

        # Get image bounds for validation
        bounds = src.bounds
        print(f"Image bounds: {bounds}")

        # Counters
        success_count = 0
        skip_outside = 0
        skip_edge = 0
        skip_nodata = 0
        skip_error = 0

        for idx, row in df.iterrows():
            lon, lat = row['LONGITUDE'], row['LATITUDE']
            status = "unknown"
            chip_path = None

            # Transform coordinates if necessary
            if transformer:
                x, y = transformer.transform(lon, lat)
            else:
                x, y = lon, lat

            # === EDGE CASE 1: Point outside image bounds ===
            if not (bounds.left <= x <= bounds.right and bounds.bottom <= y <= bounds.top):
                status = "outside_bounds"
                skip_outside += 1
                metadata_records.append({
                    **row.to_dict(),  # Include all original CSV columns
                    'idx': idx, 'status': status, 'chip_path': None
                })
                continue

            # Convert to pixel coordinates
            py, px = src.index(x, y)

            # Calculate window bounds
            col_start = px - half_w
            row_start = py - half_h
            col_end = col_start + chip_width_px
            row_end = row_start + chip_height_px

            # === EDGE CASE 2: Chip would extend beyond image edges ===
            is_edge_chip = (col_start < 0 or row_start < 0 or
                           col_end > img_width or row_end > img_height)

            if is_edge_chip and skip_partial:
                status = "edge_partial"
                skip_edge += 1
                metadata_records.append({
                    **row.to_dict(),  # Include all original CSV columns
                    'idx': idx, 'status': status, 'chip_path': None
                })
                continue

            # Clamp window to valid bounds (if we're not skipping partials)
            col_start_clamped = max(0, col_start)
            row_start_clamped = max(0, row_start)
            actual_width = min(chip_width_px, img_width - col_start_clamped)
            actual_height = min(chip_height_px, img_height - row_start_clamped)

            window = Window(col_start_clamped, row_start_clamped, actual_width, actual_height)

            try:
                chip = src.read(window=window)

                # === EDGE CASE 3: Too much NoData ===
                if nodata_val is not None:
                    nodata_fraction = np.mean(chip == nodata_val)
                else:
                    # Also check for zeros or very low values as potential nodata
                    nodata_fraction = np.mean(chip == 0)

                if nodata_fraction > nodata_threshold:
                    status = f"too_much_nodata_{nodata_fraction:.1%}"
                    skip_nodata += 1
                    metadata_records.append({
                        **row.to_dict(),  # Include all original CSV columns
                        'idx': idx, 'status': status, 'chip_path': None
                    })
                    continue

                # === EDGE CASE 4: Pad partial chips to full size (if not skipping) ===
                if chip.shape[1] != chip_height_px or chip.shape[2] != chip_width_px:
                    # Create a full-size array filled with nodata/zeros
                    full_chip = np.zeros((chip.shape[0], chip_height_px, chip_width_px),
                                        dtype=chip.dtype)
                    if nodata_val is not None:
                        full_chip.fill(nodata_val)

                    # Calculate where to paste the actual data
                    paste_row = max(0, -row_start)
                    paste_col = max(0, -col_start)
                    full_chip[:, paste_row:paste_row+chip.shape[1],
                             paste_col:paste_col+chip.shape[2]] = chip
                    chip = full_chip
                    status = "success_padded"
                else:
                    status = "success"

                # Save the chip
                chip_path = os.path.join(output_dir, f"chip_{idx:06d}_{lat:.6f}_{lon:.6f}.tif")

                # Get the transform for this chip (using unclamped position for correct georef)
                chip_transform = rasterio.transform.from_bounds(
                    x - (half_w * res_x), y - (half_h * res_y),
                    x + (half_w * res_x), y + (half_h * res_y),
                    chip_width_px, chip_height_px
                )

                with rasterio.open(
                    chip_path, 'w',
                    driver='GTiff',
                    height=chip.shape[1],
                    width=chip.shape[2],
                    count=chip.shape[0],
                    dtype=chip.dtype,
                    crs=src.crs,
                    transform=chip_transform,
                    nodata=nodata_val
                ) as dst:
                    dst.write(chip)

                success_count += 1
                if success_count % 100 == 0:
                    print(f"  Processed {success_count} chips...")

            except Exception as e:
                status = f"error: {str(e)}"
                skip_error += 1

            metadata_records.append({
                **row.to_dict(),  # Include all original CSV columns
                'idx': idx, 'status': status, 'chip_path': chip_path
            })

        # Summary
        print(f"\n{'='*50}")
        print(f"EXTRACTION COMPLETE")
        print(f"{'='*50}")
        print(f"  ✓ Success:          {success_count}")
        print(f"  ✗ Outside bounds:   {skip_outside}")
        print(f"  ✗ Edge/partial:     {skip_edge}")
        print(f"  ✗ Too much nodata:  {skip_nodata}")
        print(f"  ✗ Errors:           {skip_error}")
        print(f"  Total processed:    {len(df)}")

        # Save metadata CSV
        if save_metadata:
            meta_df = pd.DataFrame(metadata_records)
            meta_path = os.path.join(output_dir, "crash_chip_extraction5years_metadata.csv")
            meta_df.to_csv(meta_path, index=False)
            print(f"\nMetadata saved to: {meta_path}")

# === MAIN ===
if __name__ == "__main__":
    # Create VRT from .jp2 files only
    vrt_path = create_vrt(TILE_FOLDERS, file_extension=".jp2")

    # Extract chips with edge case handling
    extract_chips(
        vrt_path,
        CSV_PATH,
        OUTPUT_DIR,
        chip_meters=CHIP_SIZE_METERS,
        skip_partial=SKIP_PARTIAL_CHIPS,
        nodata_threshold=NODATA_THRESHOLD,
        save_metadata=SAVE_METADATA_CSV
    )