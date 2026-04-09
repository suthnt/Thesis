#!/usr/bin/env python3
# This code was written with the assistance of Claude (Anthropic).

import argparse
import json
import os
from typing import Optional

# DOCUMENT_ROOT for Label Studio - must match env var on Della
DOCUMENT_ROOT = "/scratch/gpfs/ALAINK/Suthi"
IMAGE_EXT = (".tif", ".tiff", ".png", ".jpg", ".jpeg")

SOURCES = {
    "binary_test": {
        "path": "OrganizedDatasetBalancedBinary/test",
        "classes": ["safe", "dangerous"],
    },
    "binary_train": {
        "path": "OrganizedDatasetBalancedBinary/train",
        "classes": ["safe", "dangerous"],
    },
    "multiclass_test": {
        "path": "OrganizedDatasetMultiClass/test",
        "classes": ["0_safe", "1_crash", "2_crashes", "3plus_crashes"],
    },
    "multiclass_train": {
        "path": "OrganizedDatasetMultiClass/train",
        "classes": ["0_safe", "1_crash", "2_crashes", "3plus_crashes"],
    },
    "chipped_5years": {
        "path": "ChippedImages_5Years",
        "classes": ["safe", "dangerous", "1_crash", "2_crashes", "3plus_crashes"],
    },
}


def collect_tasks(source_key: str, limit_per_class: Optional[int] = None) -> list:
    """Collect tasks for Label Studio import."""
    cfg = SOURCES[source_key]
    base = os.path.join(DOCUMENT_ROOT, cfg["path"])
    tasks = []

    for cls in cfg["classes"]:
        folder = os.path.join(base, cls)
        if not os.path.isdir(folder):
            print(f"  [skip] {folder} (not found)")
            continue

        files = [
            f for f in os.listdir(folder)
            if f.lower().endswith(IMAGE_EXT)
        ]
        if limit_per_class is not None:
            files = files[:limit_per_class]

        rel_base = os.path.join(cfg["path"], cls)
        for f in files:
            rel_path = os.path.join(rel_base, f).replace("\\", "/")
            tasks.append({
                "data": {
                    "image": f"/data/local-files/?d={rel_path}",
                    "class_folder": cls,  # metadata for reference
                },
            })

        print(f"  {cls}: {len(files)} images")

    return tasks


def main():
    ap = argparse.ArgumentParser(description="Create Label Studio import JSON for Della chips")
    ap.add_argument("--source", choices=list(SOURCES), default="binary_test",
                    help="Which chip source to use")
    ap.add_argument("--limit", type=int, default=None,
                    help="Max images per class (for quick tests)")
    ap.add_argument("--output", "-o", default=None,
                    help="Output JSON file (default: label_studio_<source>.json)")
    args = ap.parse_args()

    out_file = args.output or f"label_studio_{args.source}.json"

    print(f"Source: {args.source}")
    print(f"Path: {SOURCES[args.source]['path']}")
    if args.limit:
        print(f"Limit: {args.limit} per class")
    print("Collecting tasks...")

    tasks = collect_tasks(args.source, args.limit)
    print(f"\nTotal tasks: {len(tasks)}")

    with open(out_file, "w") as f:
        json.dump(tasks, f, indent=2)

    print(f"Wrote: {out_file}")
    print("\n=== Label Studio setup ===")
    print("1. Set env vars before starting Label Studio:")
    print("   export LABEL_STUDIO_LOCAL_FILES_SERVING_ENABLED=true")
    print(f"   export LABEL_STUDIO_LOCAL_FILES_DOCUMENT_ROOT={DOCUMENT_ROOT}")
    print("2. In Label Studio: Project > Import > Upload Files")
    print(f"   Upload {out_file}")
    print("3. Or use API: POST /api/projects/{id}/import with the JSON")


if __name__ == "__main__":
    main()
