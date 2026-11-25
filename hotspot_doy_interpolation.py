#!/usr/bin/env python3
"""
Hotspot day-of-year (DOY) interpolation.

Builds a continuous "day of burning" raster from satellite active-fire
hotspot points, constrained to mapped burnt-area (fire scar) polygons.

Workflow
--------
1. For each calendar month, select hotspots whose DOY falls within that month
   (plus a few days of overlap before the month starts).
2. Interpolate the selected hotspot DOY values to a grid with SAGA GIS
   Multilevel B-Spline interpolation (``grid_spline`` tool 4).
3. Clip each monthly grid to the burnt-area polygons mapped for that month.
4. Mosaic the twelve clipped grids into a single int32 DOY raster
   (0 = unburnt / no data, 1-366 = interpolated day of burning).

Reference: Fisher, R. et al. (2026) A method for characterising fire events in
high fire frequency landscapes. Int. J. Wildland Fire 35(10), WF26071.
https://doi.org/10.1071/WF26071

Author: Rohan Fisher. License: MIT.

Requirements: SAGA GIS (``saga_cmd``), geopandas, rasterio, numpy.
Run ``python hotspot_doy_interpolation.py --help`` for options.
"""

import argparse
import calendar
import glob
import os
import shutil
import subprocess
import sys
from datetime import date, datetime

import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio
from rasterio.mask import mask
from rasterio.merge import merge


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def monthly_doy_windows(year, overlap_days=3):
    """Return {month: (doy_min_exclusive, doy_max_inclusive)} for a year.

    Each window starts ``overlap_days`` before the first day of the month so
    the spline has support at the month boundary. December always runs to 366.
    """
    windows = {}
    for month in range(1, 13):
        start = date(year, month, 1).timetuple().tm_yday
        end = start + calendar.monthrange(year, month)[1] - 1
        lo = max(0, start - 1 - overlap_days)
        hi = 366 if month == 12 else end
        windows[month] = (lo, hi)
    return windows


def resolve_single_file(pattern):
    """Accept a path or a glob pattern and return one existing file."""
    if os.path.isfile(pattern):
        return pattern
    matches = sorted(glob.glob(pattern))
    if not matches:
        raise FileNotFoundError(f"No file matches: {pattern}")
    if len(matches) > 1:
        print(f"Multiple matches for {pattern}; using {matches[0]}")
    return matches[0]


def load_hotspots(path, doy_field, date_field, year):
    """Read hotspots and make sure a numeric DOY column exists."""
    gdf = gpd.read_file(path)
    if date_field:
        dates = pd.to_datetime(gdf[date_field], errors="coerce")
        gdf = gdf[dates.dt.year == year].copy()
        gdf[doy_field] = dates[dates.dt.year == year].dt.dayofyear.astype(int)
    elif doy_field not in gdf.columns:
        raise KeyError(
            f"Field '{doy_field}' not found in {path}. "
            "Use --doy-field or --date-field."
        )
    gdf[doy_field] = pd.to_numeric(gdf[doy_field], errors="coerce")
    return gdf.dropna(subset=[doy_field])


# ---------------------------------------------------------------------------
# Processing steps
# ---------------------------------------------------------------------------

def interpolate_doy(saga_cmd, points, doy_field, doy_min, doy_max,
                    points_shp, out_grid, extent, cell_size,
                    epsilon, level_max):
    """Filter hotspots to a DOY window and run SAGA B-spline interpolation.

    Returns True if a grid was produced.
    """
    subset = points[(points[doy_field] > doy_min) & (points[doy_field] <= doy_max)]
    if subset.empty:
        print(f"  No hotspots for DOY {doy_min}-{doy_max}; skipping.")
        return False
    subset.to_file(points_shp)

    xmin, ymin, xmax, ymax = extent
    cmd = [
        saga_cmd, "grid_spline", "4",
        f"-SHAPES={points_shp}",
        f"-TARGET_OUT_GRID={out_grid}",
        f"-FIELD={doy_field}",
        "-TARGET_DEFINITION=0",
        f"-TARGET_USER_SIZE={cell_size}",
        f"-TARGET_USER_XMIN={xmin}",
        f"-TARGET_USER_XMAX={xmax}",
        f"-TARGET_USER_YMIN={ymin}",
        f"-TARGET_USER_YMAX={ymax}",
        "-TARGET_USER_FLAT=1",
        "-TARGET_USER_FITS=0",
        f"-EPSILON={epsilon}",
        f"-LEVEL_MAX={level_max}",
    ]
    try:
        subprocess.run(cmd, check=True)
    except subprocess.CalledProcessError as err:
        print(f"  SAGA failed for DOY {doy_min}-{doy_max}: {err}")
        return False
    print(f"  Interpolated {len(subset)} hotspots (DOY {doy_min}-{doy_max}).")
    return os.path.exists(out_grid)


def clip_to_month(doy_grid, polygons, month_field, month, out_path):
    """Clip a DOY grid to the burnt-area polygons mapped for ``month``."""
    month_polys = polygons[polygons[month_field] == month]
    if month_polys.empty:
        print(f"  No burnt-area polygons for month {month}; skipping clip.")
        return False

    with rasterio.open(doy_grid) as src:
        if src.crs and month_polys.crs and month_polys.crs != src.crs:
            month_polys = month_polys.to_crs(src.crs)
        out_image, out_transform = mask(src, list(month_polys.geometry), crop=True)
        meta = src.meta.copy()

    meta.update(driver="GTiff", height=out_image.shape[1],
                width=out_image.shape[2], transform=out_transform)
    with rasterio.open(out_path, "w", **meta) as dst:
        dst.write(out_image)
    print(f"  Clipped to month {month}: {out_path}")
    return True


def mosaic_grids(grids, extent, out_path):
    """Mosaic clipped monthly grids into one int32 DOY raster (0 = no data)."""
    sources = [rasterio.open(g) for g in grids]
    try:
        xmin, ymin, xmax, ymax = extent
        mosaic, transform = merge(sources, bounds=(xmin, ymin, xmax, ymax))
        meta = sources[0].meta.copy()
        crs = sources[0].crs
    finally:
        for src in sources:
            src.close()

    # Anything outside a valid day-of-year range (nodata fill, overshoot from
    # the spline, NaN) becomes 0.
    mosaic = np.where(np.isfinite(mosaic), np.rint(mosaic), 0)
    mosaic[(mosaic < 1) | (mosaic > 366)] = 0
    mosaic = mosaic.astype(np.int32)

    meta.update(driver="GTiff", height=mosaic.shape[1], width=mosaic.shape[2],
                transform=transform, crs=crs, dtype="int32", nodata=0,
                compress="LZW")
    with rasterio.open(out_path, "w", **meta) as dst:
        dst.write(mosaic)
    print(f"Mosaic saved: {out_path}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args(argv=None):
    p = argparse.ArgumentParser(
        description="Interpolate a day-of-burning (DOY) raster from fire "
                    "hotspots, clipped to monthly burnt-area polygons.")
    p.add_argument("--hotspots", required=True,
                   help="Hotspot point file, or a glob pattern (e.g. 'data/HotSpot*.shp').")
    p.add_argument("--burnt-areas", required=True,
                   help="Burnt-area / fire-scar polygon file with a month attribute.")
    p.add_argument("--out-dir", required=True, help="Output folder.")
    p.add_argument("--year", type=int, default=datetime.now().year,
                   help="Year being mapped (default: current year).")
    p.add_argument("--doy-field", default="DOY",
                   help="Hotspot field holding day of year (default: DOY).")
    p.add_argument("--date-field", default=None,
                   help="Optional hotspot date field; if given, DOY is computed "
                        "from it and points are limited to --year.")
    p.add_argument("--month-field", default="month",
                   help="Burnt-area field holding month 1-12 (default: month).")
    p.add_argument("--bbox", type=float, nargs=4,
                   metavar=("XMIN", "YMIN", "XMAX", "YMAX"),
                   help="Output extent in the hotspot CRS (default: hotspot extent).")
    p.add_argument("--cell-size", type=float, default=0.0026,
                   help="Output cell size in CRS units (default: 0.0026 deg, ~250-290 m).")
    p.add_argument("--overlap-days", type=int, default=3,
                   help="Days of overlap before each month window (default: 3).")
    p.add_argument("--epsilon", type=float, default=0.0001,
                   help="SAGA B-spline threshold error (default: 0.0001).")
    p.add_argument("--level-max", type=int, default=11,
                   help="SAGA B-spline maximum level (default: 11).")
    p.add_argument("--saga-cmd", default=os.environ.get("SAGA_CMD", "saga_cmd"),
                   help="Path to saga_cmd (default: $SAGA_CMD or saga_cmd on PATH).")
    p.add_argument("--output-name", default=None,
                   help="Mosaic filename (default: DOY_<year>_<MonDD>.tif).")
    p.add_argument("--keep-working", action="store_true",
                   help="Keep intermediate monthly grids.")
    return p.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)

    if shutil.which(args.saga_cmd) is None and not os.path.isfile(args.saga_cmd):
        sys.exit(f"saga_cmd not found: {args.saga_cmd} (set --saga-cmd or $SAGA_CMD)")

    hotspot_path = resolve_single_file(args.hotspots)
    print(f"Hotspots:    {hotspot_path}")
    print(f"Burnt areas: {args.burnt_areas}")

    points = load_hotspots(hotspot_path, args.doy_field, args.date_field, args.year)
    polygons = gpd.read_file(args.burnt_areas)
    polygons[args.month_field] = pd.to_numeric(polygons[args.month_field], errors="coerce")

    extent = tuple(args.bbox) if args.bbox else tuple(points.total_bounds)

    os.makedirs(args.out_dir, exist_ok=True)
    working_dir = os.path.join(args.out_dir, "working")
    os.makedirs(working_dir, exist_ok=True)

    output_name = args.output_name or f"DOY_{args.year}_{datetime.now():%b%d}.tif"
    output_path = os.path.join(args.out_dir, output_name)

    clipped = []
    for month, (doy_min, doy_max) in monthly_doy_windows(args.year, args.overlap_days).items():
        print(f"\nMonth {month:02d} (DOY >{doy_min} to {doy_max})")
        pts_shp = os.path.join(working_dir, f"DOY_points_{month:02d}.shp")
        grid = os.path.join(working_dir, f"DOY_grid_{month:02d}.tif")
        clip = os.path.join(working_dir, f"DOY_grid_{month:02d}_clipped.tif")

        if not interpolate_doy(args.saga_cmd, points, args.doy_field, doy_min, doy_max,
                               pts_shp, grid, extent, args.cell_size,
                               args.epsilon, args.level_max):
            continue
        if clip_to_month(grid, polygons, args.month_field, month, clip):
            clipped.append(clip)

    if not clipped:
        sys.exit("No monthly grids were produced; nothing to mosaic.")

    print()
    mosaic_grids(clipped, extent, output_path)

    if not args.keep_working:
        shutil.rmtree(working_dir, ignore_errors=True)


if __name__ == "__main__":
    main()
