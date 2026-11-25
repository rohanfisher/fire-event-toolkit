#!/usr/bin/env python3
"""
Fire event identification and size statistics.

Splits a day-of-burning (DOY) raster into individual fire events using
region growing, then converts the events to polygons and reports their
size distribution.

Method
------
Starting from each unlabelled burnt pixel (DOY > 0), the region grows into
neighbouring burnt pixels whose DOY differs from the *current* pixel by no
more than ``--tolerance`` days. Because the comparison is pixel-to-pixel
rather than pixel-to-seed, a fire that spreads steadily over many days is
kept as one event, while adjacent burns separated by a jump in date are split.

Outputs (per input raster, prefixed with the raster name)
--------------------------------------------------------
regions_<name>.tif        event ID raster (int32, 0 = unburnt)
<name>_GT<min>ha.shp      events larger than --min-area-ha
<name>_top<N>.shp         the N largest events
<name>_area_stats.csv     summary statistics and size-class table
<name>_merged.shp         all events dissolved into one footprint

Reference: Fisher, R. et al. (2026) A method for characterising fire events in
high fire frequency landscapes. Int. J. Wildland Fire 35(10), WF26071.
https://doi.org/10.1071/WF26071

Author: Rohan Fisher. License: MIT.

Requirements: rasterio, geopandas, pandas, numpy.
Run ``python fire_event_identification.py --help`` for options.
"""

import argparse
import glob
import os
import sys
from collections import deque

import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio
from rasterio.features import shapes

NEIGHBOURS_8 = [(-1, -1), (-1, 0), (-1, 1), (0, -1),
                (0, 1), (1, -1), (1, 0), (1, 1)]
NEIGHBOURS_4 = [(-1, 0), (0, -1), (0, 1), (1, 0)]

# Size classes in hectares: (lower exclusive, upper inclusive, label)
SIZE_CLASSES_HA = [
    (0, 1, "0 - 1 ha"),
    (1, 10, "1 - 10 ha"),
    (10, 100, "10 - 100 ha"),
    (100, 1_000, "100 - 1,000 ha"),
    (1_000, 10_000, "1,000 - 10,000 ha"),
    (10_000, 100_000, "10,000 - 100,000 ha"),
    (100_000, 1_000_000, "100,000 - 1,000,000 ha"),
    (1_000_000, float("inf"), ">1,000,000 ha"),
]


# ---------------------------------------------------------------------------
# Region growing
# ---------------------------------------------------------------------------

def label_fire_events(data, valid, tolerance, connectivity=8):
    """Label connected fire events in a DOY array.

    Parameters
    ----------
    data : 2-D array of DOY values.
    valid : 2-D bool array, True where a pixel is burnt.
    tolerance : max DOY difference between neighbouring pixels in one event.
    connectivity : 4 or 8.

    Returns an int32 array of event IDs (0 = not burnt).
    """
    neighbours = NEIGHBOURS_8 if connectivity == 8 else NEIGHBOURS_4
    rows, cols = data.shape
    labels = np.zeros(data.shape, dtype=np.int32)
    region_id = 0

    for y0, x0 in zip(*np.nonzero(valid)):
        if labels[y0, x0]:
            continue
        region_id += 1
        labels[y0, x0] = region_id
        queue = deque([(y0, x0)])
        while queue:
            cy, cx = queue.popleft()
            current = data[cy, cx]
            for dy, dx in neighbours:
                ny, nx = cy + dy, cx + dx
                if (0 <= ny < rows and 0 <= nx < cols
                        and valid[ny, nx] and not labels[ny, nx]
                        and abs(data[ny, nx] - current) <= tolerance):
                    labels[ny, nx] = region_id
                    queue.append((ny, nx))
    return labels


# ---------------------------------------------------------------------------
# Vector outputs and statistics
# ---------------------------------------------------------------------------

def events_to_geodataframe(labels_tif):
    with rasterio.open(labels_tif) as src:
        labels = src.read(1)
        features = (
            {"properties": {"region_id": int(v)}, "geometry": geom}
            for geom, v in shapes(labels, mask=labels > 0, transform=src.transform)
        )
        return gpd.GeoDataFrame.from_features(list(features), crs=src.crs)


def add_area(gdf, area_crs):
    """Add area_ha / area_km2 columns, using an equal-area CRS if geographic."""
    if gdf.crs is None:
        print("Warning: raster has no CRS; areas are in squared pixel units.")
        area_m2 = gdf.geometry.area
    elif gdf.crs.is_geographic:
        area_m2 = gdf.geometry.to_crs(area_crs).area
    else:
        area_m2 = gdf.geometry.area
    gdf["area_ha"] = area_m2 / 1e4
    gdf["area_km2"] = area_m2 / 1e6
    return gdf


def size_statistics(gdf, min_area_ha, top_n, noise_ha):
    big = gdf[gdf["area_ha"] > min_area_ha].sort_values("area_ha", ascending=False)
    rows = [
        [f"Total area of events >{min_area_ha:g} ha", "", big["area_ha"].sum()],
        [f"Combined area of top {top_n} events", "", big.head(top_n)["area_ha"].sum()],
        [f"Number of events >{min_area_ha:g} ha", len(big), ""],
    ]
    if len(big):
        rows.append(["Area of largest event", "", big.iloc[0]["area_ha"]])
        for i, area in enumerate(big.head(top_n)["area_ha"], start=1):
            rows.append([f"Area of event #{i}", "", area])

    for lo, hi, label in SIZE_CLASSES_HA:
        sel = (gdf["area_ha"] > lo) & (gdf["area_ha"] <= hi)
        rows.append([f"Size class {label}", int(sel.sum()), gdf.loc[sel, "area_ha"].sum()])

    rows.append(["Total area (all events)", len(gdf), gdf["area_ha"].sum()])
    rows.append([f"Total area (events >{noise_ha:g} ha)", "",
                 gdf.loc[gdf["area_ha"] > noise_ha, "area_ha"].sum()])
    return big, pd.DataFrame(rows, columns=["statistic", "count", "area_ha"])


# ---------------------------------------------------------------------------
# Per-file driver
# ---------------------------------------------------------------------------

def process_raster(input_tif, out_dir, args):
    name = os.path.splitext(os.path.basename(input_tif))[0]
    print(f"\nProcessing {input_tif}")

    with rasterio.open(input_tif) as src:
        data = src.read(1)
        meta = src.meta.copy()
        nodata = src.nodata

    valid = np.isfinite(data) & (data > 0)
    if nodata is not None:
        valid &= data != nodata

    labels = label_fire_events(data, valid, args.tolerance, args.connectivity)
    print(f"  {labels.max()} fire events identified")

    labels_tif = os.path.join(out_dir, f"regions_{name}.tif")
    meta.update(driver="GTiff", dtype="int32", nodata=0, count=1, compress="LZW")
    with rasterio.open(labels_tif, "w", **meta) as dst:
        dst.write(labels, 1)
    print(f"  Event raster: {labels_tif}")

    gdf = events_to_geodataframe(labels_tif)
    if gdf.empty:
        print("  No burnt pixels found; skipping vector outputs.")
        return
    gdf = add_area(gdf, args.area_crs)

    big, stats = size_statistics(gdf, args.min_area_ha, args.top_n, args.noise_ha)

    stats_csv = os.path.join(out_dir, f"{name}_area_stats.csv")
    stats.to_csv(stats_csv, index=False)
    print(f"  Statistics:   {stats_csv}")

    if not big.empty:
        big_shp = os.path.join(out_dir, f"{name}_GT{args.min_area_ha:g}ha.shp")
        big.to_file(big_shp)
        top_shp = os.path.join(out_dir, f"{name}_top{args.top_n}.shp")
        big.head(args.top_n).to_file(top_shp)
        print(f"  Large events: {big_shp}")
        print(f"  Top {args.top_n}:       {top_shp}")

    if not args.no_merged:
        merged_shp = os.path.join(out_dir, f"{name}_merged.shp")
        gdf[["geometry"]].dissolve().to_file(merged_shp)
        print(f"  Footprint:    {merged_shp}")

    print(f"  Events >{args.min_area_ha:g} ha: {len(big)}, "
          f"total {big['area_ha'].sum():,.1f} ha"
          + (f", largest {big.iloc[0]['area_ha']:,.1f} ha" if len(big) else ""))


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def collect_inputs(paths):
    files = []
    for p in paths:
        if os.path.isdir(p):
            files += sorted(glob.glob(os.path.join(p, "*.tif")))
        else:
            files += sorted(glob.glob(p)) or ([p] if os.path.isfile(p) else [])
    return files


def parse_args(argv=None):
    p = argparse.ArgumentParser(
        description="Identify individual fire events in day-of-burning rasters "
                    "and summarise their sizes.")
    p.add_argument("inputs", nargs="+",
                   help="DOY GeoTIFF(s), glob pattern(s) or folder(s) of .tif files.")
    p.add_argument("--out-dir", required=True, help="Output folder.")
    p.add_argument("--tolerance", type=float, default=1.0,
                   help="Max DOY difference between neighbouring pixels in the "
                        "same event (default: 1).")
    p.add_argument("--connectivity", type=int, choices=(4, 8), default=8,
                   help="Pixel neighbourhood (default: 8).")
    p.add_argument("--min-area-ha", type=float, default=10.0,
                   help="Threshold for the 'large events' outputs (default: 10 ha).")
    p.add_argument("--top-n", type=int, default=10,
                   help="Number of largest events to export (default: 10).")
    p.add_argument("--noise-ha", type=float, default=0.2,
                   help="Events at or below this size are excluded from the "
                        "'Total area (events >X ha)' statistic (default: 0.2 ha).")
    p.add_argument("--area-crs", default="EPSG:6933",
                   help="Equal-area CRS for area calculation when the input is in "
                        "geographic coordinates (default: EPSG:6933).")
    p.add_argument("--no-merged", action="store_true",
                   help="Skip the dissolved footprint shapefile.")
    return p.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    files = collect_inputs(args.inputs)
    if not files:
        sys.exit("No input rasters found.")
    os.makedirs(args.out_dir, exist_ok=True)
    for f in files:
        process_raster(f, args.out_dir, args)
    print("\nAll files processed.")


if __name__ == "__main__":
    main()
